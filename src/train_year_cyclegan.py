"""
train_year_cyclegan.py

Train the year-conditional CycleGAN (gan_models.YearCycleGAN) to translate an
image's year-bucket while preserving content, via cycle consistency + a
variational year-style latent.

SOTA training practices
-----------------------
  * Adam(betas=0.5,0.999), TTUR (D faster than G).
  * Linear LR decay over the second half of training (CycleGAN schedule).
  * EMA of the generator for sampling (much cleaner grids).
  * Image-buffer in the D step (inside YearCycleGAN).
  * Periodic translation grids: each source row -> all N year buckets.

Data convention: tanh generator -> images in [-1, 1].

Usage
-----
CUDA_VISIBLE_DEVICES=0 python train_year_cyclegan.py \
    --objects Boy Girl Woman Man Person House Car \
    --img_size 128 --batch_size 16 --epochs 100 --out_dir runs/yearcgan

Quick offline smoke test:
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train_year_cyclegan.py \
    --objects Boy --img_size 64 --batch_size 4 --limit 64 --max_steps 3 --no_wandb
"""

import argparse
import copy
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from core_datautils import df as data_complete
from train_experts_dataloader import SpecialistDataloaderWithClass
from gan_models import YearCycleGAN, load_year_proxies


# ── data ──────────────────────────────────────────────────────────────────────

def build_dataset(objects, img_size, limit=None, seed=0):
    """Combine several object datasets; images normalized to [-1, 1]."""
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = SpecialistDataloaderWithClass(data_complete, objects[0], transforms=tfm)
    avlabels = ds.available_labels
    for obj in objects[1:]:
        extra = SpecialistDataloaderWithClass(data_complete, obj, transforms=tfm)
        ds = ds + extra
        ds.available_labels = avlabels
    if limit is not None and limit < len(ds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:limit].tolist()
        ds = Subset(ds, idx)
    return ds, avlabels


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


# ── sampling grid ─────────────────────────────────────────────────────────────

@torch.no_grad()
def translation_grid(gen, sources, n_years, device):
    """Row per source image: [source | year_0 | ... | year_{N-1}]."""
    gen.eval()
    rows = []
    for img in sources:
        img = img.unsqueeze(0).to(device)
        row = [denorm(img.cpu())]
        for y in range(n_years):
            out = gen.translate_to_year(img, y, stochastic=False)
            row.append(denorm(out.cpu()))
        rows.append(torch.cat(row, 0))
    gen.train()
    grid = make_grid(torch.cat(rows, 0), nrow=n_years + 1, padding=2)
    return grid


# ── lr schedule (constant, then linear decay to 0) ────────────────────────────

def make_scheduler(opt, epochs, decay_start):
    def rule(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, epochs - decay_start))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=rule)


# ── EMA ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ema_update(ema_model, model, decay):
    for pe, pm in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="+",
                   default=["Boy", "Girl", "Woman", "Man", "Person", "House", "Car"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--decay_start", type=int, default=50,
                   help="Epoch at which LR begins linear decay to 0")

    # model
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=9)
    p.add_argument("--style_dim", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--style_mode", choices=["variational", "deterministic", "freelatent"],
                   default="variational",
                   help="year-conditioning: ours (variational) | StarGAN-style "
                        "(deterministic) | StarGAN-v2-style (freelatent)")
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_idt", type=float, default=5.0)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.01)

    # modular colorimetric add-ons (default off → original model/behaviour)
    p.add_argument("--color_head", action="store_true",
                   help="Add a year-conditioned colour/tone/grain transform to G's output")
    p.add_argument("--color_classifier", action="store_true",
                   help="Add a year classifier that sees only global colour statistics")
    p.add_argument("--lambda_color", type=float, default=1.0,
                   help="Weight of the colour-classifier loss (D on reals, G on fakes)")

    # initialize the year means from learned representation-learning centroids
    p.add_argument("--init_centroids", type=str, default=None,
                   help="Path to a year_loss_epoch*.pth (or its folder) whose 'proxies' "
                        "seed the year latent means; sets latent_dim to the proxy dim")
    p.add_argument("--centroid_epoch", type=int, default=None,
                   help="Which epoch's proxies to use (default: latest in the folder)")
    p.add_argument("--centroid_norm", choices=["unit", "std", "none"], default="unit",
                   help="How to rescale the centroids before seeding mu (default: unit)")
    p.add_argument("--disc_oracle_init", type=str, default=None,
                   help="Path to a DateOracle checkpoint (e.g. runs/oracle/oracle.pth). "
                        "Warm-starts the discriminator's year-classifier branch from this "
                        "pretrained oracle and keeps training it jointly. To avoid "
                        "circularity, evaluate faithfulness with a DIFFERENT oracle.")

    # optimization (TTUR)
    p.add_argument("--g_lr", type=float, default=1e-4)
    p.add_argument("--d_lr", type=float, default=4e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # logging / io
    p.add_argument("--out_dir", type=str, default="runs/yearcgan")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--ckpt_every_steps", type=int, default=0,
                   help="Also checkpoint every N steps (0=off). Use for runs whose "
                        "epochs are huge so a loadable ckpt appears quickly.")
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--use_wandb", action="store_true",
                   help="Explicitly enable Weights & Biases logging (on by default "
                        "unless --no_wandb).")
    p.add_argument("--common_representation", type=str, default=None,
                   help="Path to an already-trained YearCycleGAN checkpoint. Every "
                        "input image is first mapped (online, frozen) to a common "
                        "year via this generator, so all inputs share one colour "
                        "profile and the model must learn structural (not just "
                        "colorimetric) year changes. Origin-year labels are kept.")
    p.add_argument("--common_year", type=int, default=None,
                   help="Year bucket all inputs are mapped to (default: the centre "
                        "bucket n_years//2, to avoid an early/late bias).")
    p.add_argument("--max_steps", type=int, default=None, help="Stop early (smoke test)")
    p.add_argument("--limit", type=int, default=None, help="Subsample dataset (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold_out_target", type=int, default=None,
                   help="Never sample this year bucket as a translation target "
                        "(extrapolation: reach it at test time only by interpolation).")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint. Pass a .pth path, or 'auto' to "
                        "use the most recent checkpoint in --out_dir. Restores "
                        "G/D/EMA weights (and optimizer+step if the checkpoint "
                        "has them) and continues the epoch/LR schedule.")
    return p.parse_args()


def find_latest_ckpt(out_dir):
    """Most recent checkpoint in out_dir: prefer ckpt_latest.pth, else the
    ckpt_epoch*.pth with the highest (epoch, step)."""
    latest = os.path.join(out_dir, "ckpt_latest.pth")
    if os.path.exists(latest):
        return latest
    import glob, re
    cands = glob.glob(os.path.join(out_dir, "ckpt_epoch*.pth"))
    if not cands:
        return None
    def key(p):
        m = re.search(r"epoch(\d+)(?:_s(\d+))?", os.path.basename(p))
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (-1, -1)
    return max(cands, key=key)


def load_frozen_generator(path, n_years, device):
    """Load an already-trained YearCycleGAN (its EMA weights) as a frozen,
    eval-mode generator used to map inputs to a common year. Robust to checkpoints
    that carry extra branches (colour head, oracle classifier): only G + style are
    needed by translate_to_year."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("ema", ck["model"])
    cargs = ck.get("args", {}) or {}
    ld = sd["style.mu.weight"].shape[1]
    gen = YearCycleGAN(n_years=n_years, latent_dim=ld,
                       color_head=cargs.get("color_head", False),
                       style_mode=cargs.get("style_mode", "variational")).to(device)
    missing, _ = gen.load_state_dict(sd, strict=False)
    assert not any(k.startswith(("G.", "style.")) for k in missing), \
        f"common-representation generator missing G/style weights: {missing[:5]}"
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    use_wandb = args.use_wandb or not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project="year-cyclegan", config=vars(args))

    ds, avlabels = build_dataset(args.objects, args.img_size, args.limit, args.seed)
    n_years = len(avlabels)
    print(f"Dataset: {len(ds)} images | {n_years} year buckets {avlabels}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    # common-representation: frozen generator that maps every input to one year
    common_gen, common_year = None, None
    if args.common_representation is not None:
        common_gen = load_frozen_generator(args.common_representation, n_years, device)
        common_year = args.common_year if args.common_year is not None else n_years // 2
        print(f"Common representation: mapping all inputs to bucket {common_year} "
              f"({avlabels[common_year]}) via {args.common_representation} "
              f"(frozen); origin-year labels kept.", flush=True)

    init_centroids = None
    if args.init_centroids is not None:
        init_centroids = load_year_proxies(args.init_centroids, args.centroid_epoch,
                                           normalize=args.centroid_norm)
        assert init_centroids.size(0) == n_years, \
            f"{init_centroids.size(0)} centroids but {n_years} year buckets"
        args.latent_dim = init_centroids.size(1)        # match latent dim to proxy dim
        print(f"Seeding year means from {args.init_centroids} "
              f"({tuple(init_centroids.shape)}, norm={args.centroid_norm}) → latent_dim={args.latent_dim}")

    oracle_cls = None
    if args.disc_oracle_init is not None:
        from eval_cyclegan import DateOracle
        ock = torch.load(args.disc_oracle_init, map_location="cpu", weights_only=False)
        backbone = ock.get("backbone", "resnet")
        oracle_cls = DateOracle(n_years, backbone=backbone)
        oracle_cls.load_state_dict(ock["state_dict"] if "state_dict" in ock else ock)
        print(f"Warm-starting D's year-classifier from {args.disc_oracle_init} "
              f"(backbone={backbone}); it will keep training jointly.")

    model = YearCycleGAN(
        n_years=n_years, ngf=args.ngf, ndf=args.ndf, n_blocks=args.n_blocks,
        style_dim=args.style_dim, latent_dim=args.latent_dim,
        lambda_cyc=args.lambda_cyc, lambda_idt=args.lambda_idt,
        lambda_cls=args.lambda_cls, lambda_kl=args.lambda_kl,
        color_head=args.color_head, color_classifier=args.color_classifier,
        lambda_color=args.lambda_color, init_centroids=init_centroids,
        style_mode=args.style_mode, oracle_cls=oracle_cls,
    ).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(model.generator_parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator_parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    sched_g = make_scheduler(opt_g, args.epochs, args.decay_start)
    sched_d = make_scheduler(opt_d, args.epochs, args.decay_start)

    # fixed source images for consistent sample grids
    fixed = torch.stack([ds[i][0] for i in range(min(args.n_samples, len(ds)))], 0)
    if common_gen is not None:                          # show grids from common-year sources
        with torch.no_grad():
            fixed = common_gen.translate_to_year(fixed.to(device), common_year,
                                                 stochastic=False).cpu()

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch, start_step = 0, 0
    if args.resume:
        ckpt_path = find_latest_ckpt(args.out_dir) if args.resume == "auto" else args.resume
        if ckpt_path and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            if ck.get("opt_g") is not None:
                opt_g.load_state_dict(ck["opt_g"])
                opt_d.load_state_dict(ck["opt_d"])
            else:
                print("  (checkpoint has no optimizer state — warm restart)")
            ck_epoch = ck.get("epoch", -1)
            start_epoch = ck_epoch + 1                    # epoch ckpt = epoch finished
            start_step = ck.get("step") or start_epoch * len(loader)
            for _ in range(start_epoch):                  # fast-forward LR schedule
                sched_g.step(); sched_d.step()
            print(f"Resumed from {ckpt_path}: epoch {ck_epoch} → start at epoch "
                  f"{start_epoch}, step {start_step}, g_lr {sched_g.get_last_lr()[0]:.2e}")
        else:
            print(f"--resume given but no checkpoint found in {args.out_dir}; "
                  f"starting from scratch.")

    step = start_step
    for epoch in range(start_epoch, args.epochs):
        for image_A, _condition_A, _image_B, condition_B, _category in loader:
            # image_A's true origin year is condition_B (dataloader naming);
            # target year is sampled uniformly.
            x = image_A.to(device)
            y_origin = condition_B.to(device)
            if common_gen is not None:                  # map every input to one year
                with torch.no_grad():
                    x = common_gen.translate_to_year(x, common_year, stochastic=False)
            y_target = torch.randint(0, n_years, (x.size(0),), device=device)
            if args.hold_out_target is not None:        # extrapolation: never target it
                m = y_target == args.hold_out_target
                while m.any():
                    y_target[m] = torch.randint(0, n_years, (int(m.sum()),), device=device)
                    m = y_target == args.hold_out_target

            # ── D step ──
            opt_d.zero_grad(set_to_none=True)
            d_loss, d_logs = model.d_losses(x, y_origin, y_target)
            d_loss.backward()
            opt_d.step()

            # ── G step ──
            opt_g.zero_grad(set_to_none=True)
            g_loss, g_logs = model.g_losses(x, y_origin, y_target)
            g_loss.backward()
            opt_g.step()

            ema_update(ema, model, args.ema_decay)

            if step % args.log_every == 0:
                logs = {**d_logs, **g_logs,
                        "epoch": epoch, "g_lr": sched_g.get_last_lr()[0]}
                print(f"e{epoch} s{step} | " + " ".join(f"{k}:{v:.3f}"
                      for k, v in {**d_logs, **g_logs}.items()))
                if use_wandb:
                    wandb.log(logs, step=step)

            if step % args.sample_every == 0:
                grid = translation_grid(ema, fixed, n_years, device)
                path = os.path.join(args.out_dir, f"grid_s{step:07d}.png")
                save_image(grid, path)
                if use_wandb:
                    import wandb
                    wandb.log({"translations": wandb.Image(path)}, step=step)

            # step-based checkpoint (so a run is loadable long before the first
            # epoch boundary; epochs over the full set are ~24k steps)
            if args.ckpt_every_steps and step > 0 and step % args.ckpt_every_steps == 0:
                _save(model, ema, args, epoch, step=step, opt_g=opt_g, opt_d=opt_d)

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                print("max_steps reached — stopping.")
                _save(model, ema, args, epoch, opt_g=opt_g, opt_d=opt_d)
                return

        sched_g.step()
        sched_d.step()
        if (epoch + 1) % args.save_every == 0:
            _save(model, ema, args, epoch, opt_g=opt_g, opt_d=opt_d)

    _save(model, ema, args, args.epochs - 1, opt_g=opt_g, opt_d=opt_d)
    print("Training complete.")


def _save(model, ema, args, epoch, step=None, opt_g=None, opt_d=None):
    tag = f"epoch{epoch}" if step is None else f"epoch{epoch}_s{step:07d}"
    path = os.path.join(args.out_dir, f"ckpt_{tag}.pth")
    blob = {"model": model.state_dict(), "ema": ema.state_dict(),
            "epoch": epoch, "step": step, "args": vars(args),
            "opt_g": opt_g.state_dict() if opt_g is not None else None,
            "opt_d": opt_d.state_dict() if opt_d is not None else None}
    torch.save(blob, path)
    # keep a stable pointer to the most recent checkpoint
    torch.save(blob, os.path.join(args.out_dir, "ckpt_latest.pth"))
    print(f"Saved → {path}", flush=True)


if __name__ == "__main__":
    main()
