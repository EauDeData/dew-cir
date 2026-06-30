"""
eval_interp_color.py — is the year manifold *metrically* aligned with time?

Generating with a style code interpolated t of the way from bucket a to bucket b
should yield images whose COLOUR signature matches real photos of the in-between
year ~ (1-t)*year_a + t*year_b. We test linear vs spherical (slerp) latent
interpolation, the earliest<->latest sweep, the farthest->centre case, and
adjacent (colour-rich) pairs. Colour distance = colour-FID over the 11-D
photographic-colour signature (eval_color).

  python eval_interp_color.py --ckpt runs/yearcgan_centroids_v2/ckpt_latest.pth \
      --objects Car House --limit 1500
"""

import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

import eval_color as C
from eval_variational import load_model


def slerp(t, v0, v1):
    v0n, v1n = v0 / v0.norm(), v1 / v1.norm()
    dot = torch.clamp((v0n * v1n).sum(), -1, 1)
    omega = torch.arccos(dot)
    if omega < 1e-4:
        return (1 - t) * v0 + t * v1
    so = torch.sin(omega)
    return torch.sin((1 - t) * omega) / so * v0 + torch.sin(t * omega) / so * v1


@torch.no_grad()
def gen_interp(model, src, a, b, t, device, mode="lerp", bs=64):
    mu = model.style.mu.weight.detach()
    v0, v1 = mu[a], mu[b]
    z = slerp(t, v0, v1) if mode == "slerp" else (1 - t) * v0 + t * v1
    w = model.style.mlp(z.unsqueeze(0).to(device))               # (1, style_dim)
    out = []
    for i in range(0, len(src), bs):
        xb = src[i:i + bs].to(device)
        out.append(model.G(xb, w.expand(len(xb), -1)).cpu())
    return torch.cat(out)


def nearest_bucket(gen_imgs, real_np):
    gi = C.to_numpy_imgs(gen_imgs)
    d = np.array([C.colour_fid(gi, real_np[k]) if real_np[k] is not None else np.inf
                  for k in range(len(real_np))])
    return d, int(np.nanargmin(d))


def sweep(model, src, real_np, device, K, freq, mode):
    errs, preds, exps = [], [], []
    for t in np.linspace(0, 1, 11):
        _, pred = nearest_bucket(gen_interp(model, src, 0, K - 1, t, device, mode), real_np)
        exp = t * (K - 1)
        errs.append(abs(pred - exp) * freq); preds.append(pred); exps.append(exp)
    rho, _ = spearmanr(exps, preds)
    return np.mean(errs), abs(rho), preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--objects", nargs="+", default=["Car", "House"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--cap", type=int, default=150)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from train_year_cyclegan import build_dataset
    ds, avlabels = build_dataset(args.objects, args.img_size, args.limit)
    K = len(avlabels); freq = avlabels[1] - avlabels[0]
    real = [[] for _ in range(K)]
    for i in range(len(ds)):
        it = ds[i]; b = int(it[3])
        if len(real[b]) < args.cap:
            real[b].append(it[0])
    real_np = [C.to_numpy_imgs(torch.stack(r)) if len(r) >= 2 else None for r in real]
    src = torch.stack([ds[i][0] for i in range(min(64, len(ds)))])
    model, _ = load_model(args.ckpt, device)
    print(f"== interp-colour | {args.ckpt.split('/')[-1]} | K={K} ==", flush=True)

    # (1) linear vs slerp sweep, earliest<->latest
    for mode in ("lerp", "slerp"):
        mae, rho, preds = sweep(model, src, real_np, device, K, freq, mode)
        print(f"(1) {mode:5} sweep: mean|err|={mae:4.1f} yr  Spearman={rho:.3f}  pred={preds}", flush=True)

    # (2) farthest -> centre, both modes
    for mode in ("lerp", "slerp"):
        d, pred = nearest_bucket(gen_interp(model, src, 0, K - 1, 0.5, device, mode), real_np)
        centre = (K - 1) / 2
        print(f"(2) {mode:5} extremes@0.5 -> bucket {pred} (year {avlabels[pred]}); "
              f"centre~{avlabels[int(round(centre))]} | err={abs(pred-centre)*freq:.1f} yr", flush=True)

    # (3) adjacent colour-rich pairs (late buckets): midpoint should sit between
    print("(3) adjacent late pairs, interp@0.5 (expect nearest in {a,b} or between):")
    for a, b in [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]:
        for mode in ("lerp", "slerp"):
            _, pred = nearest_bucket(gen_interp(model, src, a, b, 0.5, device, mode), real_np)
            mid = (a + b) / 2
            print(f"   {avlabels[a]}<->{avlabels[b]} [{mode}] -> bucket {pred} "
                  f"(year {avlabels[pred]}) | |err to mid|={abs(pred-mid)*freq:.1f} yr", flush=True)


if __name__ == "__main__":
    main()
