"""
eval_variational.py — the experiments that justify the *variational* year code
(things StarGAN / StarGAN-v2 conditioning can't give): a structured, ordered,
interpolable year manifold and intra-year diversity.

Metrics (on a variational YearCycleGAN checkpoint + the independent oracle):
  * latent ordering   — Spearman rho between the 1-D principal coordinate of the
                        learned year means {mu_y} and the true year. High => the
                        latent recovered the chronological axis.
  * interpolation     — translate with w interpolated between the first and last
    monotonicity        bucket over t in [0,1]; the oracle's predicted year should
                        rise monotonically. Report Spearman(t, y_hat) and the MAE
                        to the linearly-expected year.
  * intra-year        — for a fixed image and year, sample K stochastic styles and
    diversity           measure mean pairwise distance of the outputs (a model
                        with a real distribution per year is diverse; a
                        deterministic one is not).
Also writes a qualitative interpolation strip.

  python eval_variational.py --ckpt runs/yearcgan_centroids_v2/ckpt_latest.pth \
      --oracle runs/oracle_convnext/oracle.pth --objects Car House --limit 300
"""

import argparse, os
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from scipy.stats import spearmanr

from eval_cyclegan import DateOracle
import eval_color as C


def load_model(ckpt, device):
    from gan_models import YearCycleGAN
    ck = torch.load(ckpt, map_location=device); a = ck.get("args", {})
    m = YearCycleGAN(n_years=len(ck.get("avlabels", range(14))) if "avlabels" in ck else 14,
                     ngf=a.get("ngf", 64), ndf=a.get("ndf", 64), n_blocks=a.get("n_blocks", 9),
                     style_dim=a.get("style_dim", 64), latent_dim=a.get("latent_dim", 64),
                     style_mode=a.get("style_mode", "variational")).to(device)
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); m.eval()
    return m, a


@torch.no_grad()
def latent_ordering(model):
    mu = model.style.mu.weight.detach().cpu().numpy()      # (K, d)
    mu = mu - mu.mean(0)
    # leading principal coordinate
    u, s, vt = np.linalg.svd(mu, full_matrices=False)
    pc1 = mu @ vt[0]
    rho, _ = spearmanr(pc1, np.arange(len(mu)))
    return abs(float(rho))


@torch.no_grad()
def interpolation(model, oracle, imgs, device, K, steps=11):
    a, b = 0, K - 1
    preds = []                                             # (steps,) mean predicted bucket
    ts = np.linspace(0, 1, steps)
    for t in ts:
        w = model.style.interpolate_w(a, b, float(t)).to(device)   # (1, style_dim)
        out = model.G(imgs.to(device), w.expand(len(imgs), -1))
        preds.append(oracle.predict(out).float().mean().item())
    preds = np.array(preds)
    rho, _ = spearmanr(ts, preds)
    expected = ts * (K - 1)
    return abs(float(rho)), float(np.mean(np.abs(preds - expected)))


@torch.no_grad()
def diversity(model, imgs, device, K, n_samples=6):
    """Mean pairwise pixel distance among stochastic samples for a fixed (image,
    year), averaged over a few years. ~0 for deterministic/free-latent style
    that ignores the per-year variance."""
    div = []
    for y in range(0, K, max(1, K // 4)):
        years = torch.full((len(imgs),), y, device=device, dtype=torch.long)
        samps = [model.translate(imgs.to(device), years, stochastic=True)[0] for _ in range(n_samples)]
        s = torch.stack(samps)                              # (n, B, 3, H, W)
        d = 0.0; c = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d += (s[i] - s[j]).abs().mean().item(); c += 1
        div.append(d / c)
    return float(np.mean(div))


@torch.no_grad()
def interp_strip(model, img, device, K, steps=14, path="interp_strip.png"):
    row = [(img.cpu() + 1) / 2]
    for t in np.linspace(0, 1, steps):
        w = model.style.interpolate_w(0, K - 1, float(t)).to(device)
        out = model.G(img.unsqueeze(0).to(device), w)[0].cpu()
        row.append((out.clamp(-1, 1) + 1) / 2)
    save_image(make_grid(torch.stack(row), nrow=steps + 1, padding=2), path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--objects", nargs="+", default=["Car"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--out", default="runs/variational")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    from train_year_cyclegan import build_dataset
    ds, avlabels = build_dataset(args.objects, args.img_size, args.limit)
    K = len(avlabels)
    imgs = torch.stack([ds[i][0] for i in range(min(64, len(ds)))])

    model, a = load_model(args.ckpt, device)
    ock = torch.load(args.oracle, map_location=device)
    oracle = DateOracle(K, backbone=ock.get("backbone", "resnet")).to(device)
    oracle.load_state_dict(ock["state_dict"]); oracle.eval()

    print(f"== variational eval | {os.path.basename(args.ckpt)} | style_mode="
          f"{a.get('style_mode','variational')} | K={K} ==", flush=True)
    rho_ord = latent_ordering(model)
    rho_int, mae_int = interpolation(model, oracle, imgs, device, K)
    div = diversity(model, imgs, device, K)
    print(f"[latent ordering]   Spearman rho(mu PC1, year) = {rho_ord:.3f}")
    print(f"[interpolation]     Spearman(t, y_hat) = {rho_int:.3f} | MAE to linear = {mae_int:.2f} buckets")
    print(f"[intra-year divers] mean pairwise |.|_1 = {div:.4f}  (0 => no diversity)")
    interp_strip(model, imgs[0], device, K, path=os.path.join(args.out, "interp_strip.png"))
    print(f"strip -> {args.out}/interp_strip.png")


if __name__ == "__main__":
    main()
