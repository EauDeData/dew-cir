"""
deep_dream.py

Feature-visualization of a composed (object, year) representation, using MACO
(MAgnitude Constrained Optimization — Fel et al., NeurIPS 2023).

In the evaluation protocol a query image representation is composed as

        r = μ^c  +  μ^y                       (LABEL + LABEL query, evaluation.py)

where μ^c is the object/category proxy (contrastive_loss_fn.proxies[c]) and
μ^y is the year proxy (year_loss_fn.proxies[y]). This script synthesizes an
image x whose encoder feature f(x) matches r.

Why MACO instead of plain Lucid-style optimization
---------------------------------------------------
Standard pixel/Fourier optimization keeps finding high-frequency *adversarial*
solutions: the cosine to the target shoots toward 1.0 while the image stays
noisy. MACO removes that failure mode by decomposing the Fourier spectrum into
polar form and **fixing the magnitude to a natural-image spectrum, optimizing
only the phase**. Because the magnitude is constrained to a plausible 1/f
profile, every iterate is already natural in the frequency domain — so there is
no adversarial high-frequency escape and no need for TV / scaling hyperparams.
(Fel et al. 2023; reference impl. serre-lab/Horama.)

Usage
-----
CUDA_VISIBLE_DEVICES=0 python deep_dream.py \
    --ckpt_folder /data/113-2/users/amolina/cir_date/2e4f3a40 \
    --epoch 10 --model resnet \
    --object Car --year_idx 0 \
    --out dream_car_yr0.png
"""

import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluation import load_everything

MODEL_INPUT = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Color-correlation (SVD-sqrt) matrix used by MACO/Horama for decorrelated color.
COLOR_CORRELATION_SVD_SQRT = torch.tensor([
    [0.56282854,  0.58447580,  0.58447580],
    [0.19482528,  0.00000000, -0.19482528],
    [0.04329450, -0.10823626,  0.06494176],
])


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MACO deep-dream of a composed (object, year) representation.")

    p.add_argument("--ckpt_folder", required=True)
    p.add_argument("--epoch",       type=int, required=True)
    p.add_argument("--model",       type=str, default=None,
                   choices=["convnext", "vgg", "vit", "resnet"])
    p.add_argument("--objects",     type=str, nargs="+", default=None,
                   help="Full ordered object list (default: from ckpt args.json)")

    # What to dream
    p.add_argument("--object",    type=str, required=True, help="Category name, e.g. Car")
    p.add_argument("--use_image", type=int, default=None,
                   help="Index of an image (from the selected object's dataloader) to "
                        "use as the seed for the dream. If unset, start from random phase.")
    p.add_argument("--year",      type=int, default=None,  help="Actual year, e.g. 1955")
    p.add_argument("--year_idx",  type=int, default=None,  help="Year proxy index (overrides --year)")
    p.add_argument("--year_weight", type=float, default=1.0,
                   help="Weight on the year proxy in the target (0 = pure object, 1 = object+year)")
    p.add_argument("--objective", type=str, default="cosine", choices=["cosine", "mse"])

    # MACO optimization
    p.add_argument("--steps",       type=int,   default=1000)
    p.add_argument("--lr",          type=float, default=1.0, help="NAdam LR (works at 1.0 thanks to phase z-scoring)")
    p.add_argument("--canvas",      type=int,   default=448, help="Optimized image resolution (crops are resized to 224)")
    p.add_argument("--mag_decay",   type=float, default=1.0, help="1/f^decay magnitude profile (natural≈1.0)")
    p.add_argument("--magnitude_npy", type=str, default=None,
                   help="Optional precomputed natural-image magnitude spectrum (.npy); else 1/f is used")
    p.add_argument("--transforms",  type=int,   default=8,   help="Augmented crops averaged per step")
    p.add_argument("--crop_min",    type=float, default=0.20)
    p.add_argument("--crop_max",    type=float, default=0.80)
    p.add_argument("--center_bias", type=float, default=0.0, help="0=uniform crops, 1=always centered")
    p.add_argument("--noise",       type=float, default=0.05, help="Augmentation noise level")
    p.add_argument("--seed",        type=int,   default=42)

    p.add_argument("--out", type=str, default="dream.png")
    return p.parse_args()


def read_ckpt_objects(ckpt_folder):
    with open(os.path.join(ckpt_folder, "args.json")) as f:
        cfg = json.load(f)
    return cfg.get("objects"), cfg.get("model", "convnext")


# ── MACO image parametrization ───────────────────────────────────────────────

def standardize(t):
    """Z-score normalization (Horama)."""
    return (t - t.mean()) / (t.std() + 1e-4)


def rfft2d_freqs(h, w):
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def build_magnitude(h, w, decay, device, npy_path=None):
    """Fixed spectrum magnitude: a precomputed natural-image spectrum, or 1/f."""
    wf = w // 2 + 1
    if npy_path is not None:
        mag = np.load(npy_path).astype("float32")
        mag = torch.tensor(mag)
        if mag.dim() == 2:
            mag = mag[None, None]
        elif mag.dim() == 3:
            mag = mag[None]
        mag = F.interpolate(mag, size=(h, wf), mode="bilinear", align_corners=False)[0]
    else:
        freqs = rfft2d_freqs(h, w)                               # (h, wf)
        mag = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay
        mag = torch.tensor(mag, dtype=torch.float32)[None]      # (1, h, wf)
    return mag.to(device)


class MacoImage:
    """
    Image parametrized by the *phase* of its Fourier spectrum; the magnitude is
    fixed to a natural-image profile. Reconstruction:
        phase  ← z-score(phase)
        spec   ← magnitude · e^{i·phase}
        img    ← z-score(irfft2(spec))
        img    ← sigmoid(recorrelate_color(img))   ∈ [0,1]
    """

    def __init__(self, h, w, device, decay=1.0, npy_path=None, init_image=None):
        self.h, self.w = h, w
        self.device = device
        self.magnitude = build_magnitude(h, w, decay, device, npy_path)   # (1|3, h, wf)
        if init_image is not None:
            # Seed the phase from a real image so the dream starts from its
            # structural layout. The magnitude stays the fixed 1/f profile, so
            # only the *phase* (which carries spatial structure) is transferred.
            img = init_image.to(device)
            if img.shape[-2:] != (h, w):
                img = F.interpolate(img.unsqueeze(0), size=(h, w),
                                    mode="bilinear", align_corners=False)[0]
            spec = torch.fft.rfft2(img - img.mean(), s=(h, w))            # (3, h, wf)
            self.phase = spec.angle().detach().clone().requires_grad_(True)
        else:
            self.phase = (torch.randn(3, h, w // 2 + 1, device=device) * 0.01).requires_grad_(True)
        self.color = COLOR_CORRELATION_SVD_SQRT.to(device)

    def parameters(self):
        return [self.phase]

    def _recorrelate(self, img):  # img (3,H,W)
        flat = img.permute(1, 2, 0).reshape(-1, 3) @ self.color
        return flat.view(self.h, self.w, 3).permute(2, 0, 1)

    def image(self):
        phase = standardize(self.phase)
        spectrum = torch.complex(torch.cos(phase) * self.magnitude,
                                 torch.sin(phase) * self.magnitude)
        img = torch.fft.irfft2(spectrum, s=(self.h, self.w))   # (3,H,W)
        img = standardize(img)
        img = torch.sigmoid(self._recorrelate(img))
        return img.unsqueeze(0)                                 # (1,3,H,W) in [0,1]


# ── Augmentation (random resized crops + noise), batched ─────────────────────

def maco_augment(img, n_views, crop_min, crop_max, noise, device, center_bias=0.0):
    """Sample `n_views` random crops of `img`, resize to 224, add noise.

    `center_bias` ∈ [0,1] shrinks how far crop centers can stray from the middle
    (1 = always centered). Biasing toward the center concentrates a single subject
    instead of tiling the concept across the canvas.
    """
    s  = torch.empty(n_views, device=device).uniform_(crop_min, crop_max)   # crop scale
    mt = (1.0 - s) * (1.0 - center_bias)                                    # max center offset
    cx = (torch.rand(n_views, device=device) * 2 - 1) * mt
    cy = (torch.rand(n_views, device=device) * 2 - 1) * mt
    theta = torch.zeros(n_views, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 2] = s, cx
    theta[:, 1, 1], theta[:, 1, 2] = s, cy
    grid = F.affine_grid(theta, (n_views, 3, MODEL_INPUT, MODEL_INPUT), align_corners=False)
    crops = F.grid_sample(img.expand(n_views, -1, -1, -1), grid,
                          align_corners=False, padding_mode="reflection")
    return crops + torch.randn_like(crops) * noise


# ── Core ─────────────────────────────────────────────────────────────────────

def resolve_year_idx(args, avlabels):
    if args.year_idx is not None:
        assert 0 <= args.year_idx < len(avlabels), "year_idx out of range"
        return args.year_idx
    if args.year is None:
        raise ValueError("Provide either --year or --year_idx")
    freq = avlabels[1] - avlabels[0] if len(avlabels) > 1 else 1
    bucket = args.year - (args.year % freq)
    if bucket not in avlabels:
        raise ValueError(f"Year {args.year} → bucket {bucket} not in {avlabels}")
    return avlabels.index(bucket)


def deep_dream(model, target, args, device, init_image=None):
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)

    param = MacoImage(args.canvas, args.canvas, device,
                      decay=args.mag_decay, npy_path=args.magnitude_npy,
                      init_image=init_image)
    opt = torch.optim.NAdam(param.parameters(), lr=args.lr)

    target = target.to(device)
    target_unit = F.normalize(target, dim=0)

    for step in range(args.steps):
        opt.zero_grad()
        img = param.image()                                              # (1,3,C,C) in [0,1]
        crops = maco_augment(img, args.transforms, args.crop_min,
                             args.crop_max, args.noise, device,
                             center_bias=args.center_bias)               # (V,3,224,224)
        x_norm = (crops - mean) / std
        feat, _ = model(x_norm, None)                                    # (V, 1024)

        if args.objective == "cosine":
            match = -(F.normalize(feat, dim=-1) * target_unit).sum(-1).mean()
        else:
            match = F.mse_loss(feat, target.unsqueeze(0).expand_as(feat))

        match.backward()
        opt.step()

        if step % 100 == 0 or step == args.steps - 1:
            cos = (F.normalize(feat.detach(), dim=-1) * target_unit).sum(-1).mean().item()
            print(f"  step {step:4d} | loss {match.item():.4f} | mean-cos {cos:.4f}")

    return param.image().detach()


def save_image(img, path):
    arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)
    print(f"Saved → {path}")


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_objects, ckpt_model = read_ckpt_objects(args.ckpt_folder)
    objects    = args.objects or ckpt_objects
    model_type = args.model   or ckpt_model
    eval_args  = SimpleNamespace(model=model_type)

    model, contrastive_loss_fn, year_loss_fn, avlabels = load_everything(
        args.ckpt_folder, objects, args.epoch, device, eval_args
    )
    cat_proxies  = contrastive_loss_fn.proxies.detach().to(device)
    year_proxies = year_loss_fn.proxies.detach().to(device)

    if args.object not in objects:
        raise ValueError(f"Object '{args.object}' not in {objects}")
    obj_idx  = objects.index(args.object)
    year_idx = resolve_year_idx(args, avlabels)
    print(f"Object '{args.object}' (idx {obj_idx}) | year bucket {avlabels[year_idx]} (idx {year_idx})")

    # Composed target representation:  r = μ^c + w·μ^y   (w = --year_weight)
    target = cat_proxies[obj_idx] + args.year_weight * year_proxies[year_idx]

    # Optional: seed the dream from a real image of the selected object.
    init_image = None
    if args.use_image is not None:
        from torchvision import transforms as T
        from core_datautils import df as data_complete
        from train_experts_dataloader import SpecialistDataloaderWithClass

        seed_ds = SpecialistDataloaderWithClass(
            data_complete, args.object, transforms=T.ToTensor()
        )
        if not 0 <= args.use_image < len(seed_ds):
            raise ValueError(f"--use_image {args.use_image} out of range "
                             f"[0, {len(seed_ds)}) for object '{args.object}'")
        init_image, seed_year_idx, _ = seed_ds.get_one_sample(args.use_image)   # (3,224,224) in [0,1]
        print(f"Seeding from image #{args.use_image} of '{args.object}' "
              f"(its year bucket idx {seed_year_idx})")

    img = deep_dream(model, target, args, device, init_image=init_image)
    save_image(img, args.out)


if __name__ == "__main__":
    main()
