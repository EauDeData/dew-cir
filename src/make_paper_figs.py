"""
make_paper_figs.py — regenerate the paper's qualitative figures directly from a
trained checkpoint (EMA weights), replacing the old wandb exports.

  --mode grid    -> results_grid.png : rows = source crops, cols = source + the
                    K=14 five-year buckets (deterministic mean style mu_y), with
                    decade column headers.
  --mode interp  -> interp_strip.png : several source crops re-rendered along the
                    latent interpolation from the earliest to the latest bucket,
                    with arrowheads marking the year extrema.

Run on a free GPU, e.g.
  CUDA_VISIBLE_DEVICES=3 python make_paper_figs.py --mode grid \
      --ckpt runs/yearcgan/ckpt_latest.pth --out paper_gan/figures/results_grid.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from core_datautils import df as data_complete
from train_experts_dataloader import SpecialistDataloaderWithClass
from gan_models import YearCycleGAN

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("ema", ck["model"])
    ld = sd["style.mu.weight"].shape[1]          # latent dim baked into the ckpt
    model = YearCycleGAN(n_years=14, latent_dim=ld).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, ck.get("epoch")


def source_crops(objects, img_size, per_object=1, seed=0):
    """One representative crop per object (deterministic)."""
    tfm = transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    g = torch.Generator().manual_seed(seed)
    imgs, labels = [], []
    for obj in objects:
        ds = SpecialistDataloaderWithClass(data_complete, obj, transforms=tfm)
        idxs = torch.randperm(len(ds), generator=g)[:per_object].tolist()
        for i in idxs:
            x, _, _ = ds.get_one_sample(i)
            imgs.append(x); labels.append(obj)
    return torch.stack(imgs), labels


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


def to_img(t):
    return denorm(t).permute(1, 2, 0).cpu().numpy()


YEARS = [1930 + 5 * k for k in range(14)]


@torch.no_grad()
def make_grid_fig(model, sources, labels, out, device):
    K = 14
    nrow, ncol = len(sources), K + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 0.85, nrow * 0.92))
    if nrow == 1:
        axes = axes[None, :]
    for r in range(nrow):
        x = sources[r:r + 1].to(device)
        axes[r, 0].imshow(to_img(x[0]))
        if r == 0:
            axes[r, 0].set_title("source", fontsize=9)
        axes[r, 0].set_ylabel(labels[r], fontsize=8, rotation=0,
                              ha="right", va="center")
        for k in range(K):
            out_img = model.translate_to_year(x, k, stochastic=False)
            axes[r, k + 1].imshow(to_img(out_img[0]))
            if r == 0:
                axes[r, k + 1].set_title(f"{YEARS[k]}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    fig.tight_layout(pad=0.25)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


@torch.no_grad()
def make_interp_fig(model, sources, labels, out, device, steps=9):
    ts = np.linspace(0, 1, steps)
    nrow, ncol = len(sources), steps
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 0.95, nrow * 0.95 + 0.6))
    if nrow == 1:
        axes = axes[None, :]
    for r in range(nrow):
        x = sources[r:r + 1].to(device)
        for c, t in enumerate(ts):
            w = model.style.interpolate_w(0, 13, float(t)).to(device)
            out_img = model.G(x, w)
            axes[r, c].imshow(to_img(out_img[0]))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_linewidth(0.4)
        axes[r, 0].set_ylabel(labels[r], fontsize=8, rotation=0,
                              ha="right", va="center")
    fig.tight_layout(pad=0.25, rect=[0, 0.10, 1, 1])
    # year-extrema arrow spanning the strip, under the bottom row
    ax0 = axes[-1, 0].get_position()
    axN = axes[-1, -1].get_position()
    y = ax0.y0 - 0.055
    fig.add_artist(plt.Line2D([ax0.x0, axN.x1], [y, y], color="black", lw=1.0))
    fig.text(ax0.x0, y, r"$\blacktriangleleft$", ha="center", va="center", fontsize=11)
    fig.text(axN.x1, y, r"$\blacktriangleright$", ha="center", va="center", fontsize=11)
    fig.text(ax0.x0, y - 0.035, "1930", ha="center", va="center", fontsize=9)
    fig.text(axN.x1, y - 0.035, "1995", ha="center", va="center", fontsize=9)
    fig.text((ax0.x0 + axN.x1) / 2, y - 0.035,
             r"interpolated style code  $\mathrm{MLP}((1-t)\,\mu_{1930}+t\,\mu_{1995})$",
             ha="center", va="center", fontsize=9, style="italic")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["grid", "interp"], required=True)
    p.add_argument("--objects", nargs="+",
                   default=["Car", "House", "Man", "Woman", "Boy", "Person"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} (epoch {epoch}) on {device}")
    sources, labels = source_crops(args.objects, args.img_size, seed=args.seed)
    if args.mode == "grid":
        make_grid_fig(model, sources, labels, args.out, device)
    else:
        make_interp_fig(model, sources, labels, args.out, device, args.steps)


if __name__ == "__main__":
    main()
