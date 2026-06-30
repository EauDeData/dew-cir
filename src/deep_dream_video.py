"""
deep_dream_video.py

Latent-space exploration videos via MACO deep dream.

We interpolate the *target representation* through proxy space and render a frame
at each step. The trick for a smooth video is to **warm-start each frame's phase
from the previous frame** — instead of re-optimizing from scratch, the dream
continuously deforms as the target drifts, giving fluid morphing.

Modes
-----
  year_sweep : fixed object, interpolate the year proxy 1930→1995 (object ages
               through the eras).
  morph      : interpolate the object proxy through a list of categories at a
               fixed year (e.g. Boy→Girl→Woman→Man→Boy).

Usage
-----
CUDA_VISIBLE_DEVICES=0 python deep_dream_video.py --mode year_sweep \
    --ckpt_folder /data/113-2/users/amolina/cir_date/9bf3ef52 --epoch 19 --model vgg \
    --object Boy --frames_per_seg 12 --out videos/boy_years.mp4
"""

import argparse
import os
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import deep_dream as dd
from evaluation import load_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_folder", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--objects", type=str, nargs="+", default=None)

    p.add_argument("--mode", choices=["year_sweep", "morph"], required=True)
    p.add_argument("--object", type=str, default="Boy", help="Object for year_sweep")
    p.add_argument("--morph_objects", type=str, nargs="+",
                   default=["Boy", "Girl", "Woman", "Man"], help="Sequence for morph mode")
    p.add_argument("--year_idx", type=int, default=0, help="Fixed year for morph mode")
    p.add_argument("--year_weight", type=float, default=1.0)
    p.add_argument("--loop", action="store_true", help="Loop back to the start (seamless)")

    # MACO / render config (defaults = the good config from the sweep)
    p.add_argument("--canvas", type=int, default=416)
    p.add_argument("--transforms", type=int, default=14)
    p.add_argument("--mag_decay", type=float, default=1.0)
    p.add_argument("--crop_min", type=float, default=0.30)
    p.add_argument("--crop_max", type=float, default=0.70)
    p.add_argument("--center_bias", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=99)

    # Video schedule
    p.add_argument("--frames_per_seg", type=int, default=12, help="Interp frames between key targets")
    p.add_argument("--warmup_steps", type=int, default=900, help="Steps to optimize the first frame")
    p.add_argument("--step_per_frame", type=int, default=55, help="Warm-start steps per subsequent frame")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--label", action="store_true", help="Burn era/object label onto frames")

    p.add_argument("--out", type=str, default="dream.mp4")
    return p.parse_args()


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def font(sz=22):
    for pth in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(pth):
            return ImageFont.truetype(pth, sz)
    return ImageFont.load_default()


def build_targets(args, cat_proxies, year_proxies, objects, avlabels):
    """Return a list of (target_vector, label_string)."""
    targets = []
    if args.mode == "year_sweep":
        obj = cat_proxies[objects.index(args.object)]
        keys = list(range(year_proxies.size(0)))
        if args.loop:
            keys = keys + keys[::-1][1:]
        for i in range(len(keys) - 1):
            a, b = year_proxies[keys[i]], year_proxies[keys[i + 1]]
            ya, yb = avlabels[keys[i]], avlabels[keys[i + 1]]
            for k in range(args.frames_per_seg):
                t = k / args.frames_per_seg
                targets.append((obj + args.year_weight * lerp(a, b, t),
                                f"{args.object} · {int(lerp(ya, yb, t))}"))
    else:  # morph
        yp = year_proxies[args.year_idx]
        seq = [objects.index(o) for o in args.morph_objects]
        if args.loop:
            seq = seq + [seq[0]]
        for i in range(len(seq) - 1):
            a, b = cat_proxies[seq[i]], cat_proxies[seq[i + 1]]
            na, nb = args.morph_objects[i], args.morph_objects[(i + 1) % len(args.morph_objects)]
            for k in range(args.frames_per_seg):
                t = k / args.frames_per_seg
                lbl = na if t < 0.5 else nb
                targets.append((lerp(a, b, t) + args.year_weight * yp,
                                f"{lbl} · {avlabels[args.year_idx]}"))
    return targets


def optimize_toward(param, opt, model, target_unit, args, n_steps, mean, std, device):
    for _ in range(n_steps):
        opt.zero_grad()
        img = param.image()
        crops = dd.maco_augment(img, args.transforms, args.crop_min, args.crop_max,
                                args.noise, device, center_bias=args.center_bias)
        feat, _ = model((crops - mean) / std, None)
        loss = -(F.normalize(feat, dim=-1) * F.normalize(target_unit, dim=0)).sum(-1).mean()
        loss.backward()
        opt.step()
    return param.image().detach()


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import json
    cfg = json.load(open(os.path.join(args.ckpt_folder, "args.json")))
    objects = args.objects or cfg["objects"]
    model_type = args.model or cfg.get("model", "convnext")

    model, contrastive, year_loss, avlabels = load_everything(
        args.ckpt_folder, objects, args.epoch, device, SimpleNamespace(model=model_type))
    cat_proxies = contrastive.proxies.detach().to(device)
    year_proxies = year_loss.proxies.detach().to(device)

    targets = build_targets(args, cat_proxies, year_proxies, objects, avlabels)
    print(f"{len(targets)} frames | mode={args.mode}")

    mean, std = dd.IMAGENET_MEAN.to(device), dd.IMAGENET_STD.to(device)
    param = dd.MacoImage(args.canvas, args.canvas, device, decay=args.mag_decay)
    opt = torch.optim.NAdam(param.parameters(), lr=args.lr)

    tmp = tempfile.mkdtemp(prefix="ddvid_")
    fnt = font(max(16, args.canvas // 20))

    # Warm up on the first target so frame 0 is already converged.
    optimize_toward(param, opt, model, targets[0][0], args, args.warmup_steps, mean, std, device)

    for fi, (tgt, label) in enumerate(targets):
        img = optimize_toward(param, opt, model, tgt, args, args.step_per_frame, mean, std, device)
        arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        frame = Image.fromarray(arr)
        if args.label:
            d = ImageDraw.Draw(frame)
            d.rectangle([0, 0, args.canvas, 34], fill=(0, 0, 0))
            d.text((8, 6), label, fill=(255, 255, 255), font=fnt)
        frame.save(os.path.join(tmp, f"f{fi:05d}.png"))
        if fi % 10 == 0:
            print(f"  frame {fi:4d}/{len(targets)} | {label}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", args.out,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved → {args.out}  ({len(targets)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
