"""
run_tstr.py

Train-on-Synthetic / Test-on-Real (TSTR) evaluation of the year-translation GAN.
The question: if we train a date estimator ONLY on GAN images whose year was
swapped to a known target, does it work on REAL images? If the temporal
appearance is faithful, accuracy should approach a model trained on real data.

Four estimators, identical architecture / budget / test set, differing only in
training data (all tested on the SAME real held-out split):

  real      : real crops, real labels                 -> TRTR upper bound
  synth     : GAN(x -> random year y'), label = y'     -> TSTR (the metric)
  identity  : x unchanged,  label = random y'          -> CONTROL (must be ~chance;
                                                          proves the GAN, not
                                                          leftover content, carries
                                                          the year signal)
  augment   : real + synth                             -> is synthetic useful data?

A small (real - synth) gap = faithful re-dating. identity ~ chance = the metric
is not passing trivially.

  CUDA_VISIBLE_DEVICES=6 python run_tstr.py --ckpt runs/yearcgan/ckpt_epoch99.pth \
      --objects Boy Girl Woman Man Person House Car --train_limit 15000 --epochs 5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
from torchvision import transforms

from core_datautils import df as data_complete
from train_experts_dataloader import ObjectSpecificDateLoader
from eval_cyclegan import DateOracle


def build(objects, img_size, evaluate, flip):
    t = [transforms.Resize((img_size, img_size))]
    if flip:
        t.append(transforms.RandomHorizontalFlip())
    t += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    tfm = transforms.Compose(t)
    parts = [ObjectSpecificDateLoader(data_complete, o, transforms=tfm, evaluate=evaluate)
             for o in objects]
    return ConcatDataset(parts), parts[0].available_labels


def subset(ds, n, seed=0):
    if n is None or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


class XY(torch.utils.data.Dataset):
    """Expose (img, year) only, so real crops can be concatenated with the
    materialised synthetic (img, year) sets for the augmentation run."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i][0], self.base[i][1]


def collate_xy(batch):
    """Take only (img, year) from each item and coerce year to a tensor, so real
    (int label) and synthetic (tensor label) sets collate uniformly."""
    xs = torch.stack([b[0] for b in batch])
    ys = torch.as_tensor([int(b[1]) for b in batch], dtype=torch.long)
    return xs, ys


@torch.no_grad()
def materialize(model, loader, n_years, device, translate=True):
    """Build a (img, target_year) TensorDataset. translate=True runs the GAN to a
    random target year; translate=False is the identity control (no GAN)."""
    imgs, ys = [], []
    for img, _year, _cat in loader:
        img = img.to(device)
        yt = torch.randint(0, n_years, (img.size(0),), device=device)
        out = model.translate(img, yt, stochastic=True)[0] if translate else img
        imgs.append(out.cpu()); ys.append(yt.cpu())
    return TensorDataset(torch.cat(imgs), torch.cat(ys))


def train_eval(make_loader, test_loader, n_years, device, epochs, lr, backbone):
    """Train a fresh oracle on `make_loader()` and return acc@0 / acc@1 on real."""
    model = DateOracle(n_years, backbone=backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for batch in make_loader():
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad(set_to_none=True)
            ce(model(x), y).backward(); opt.step()
    model.eval()
    c0 = c1 = tot = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            p = model.predict(x)
            c0 += (p == y).sum().item()
            c1 += (p - y).abs().le(1).sum().item()
            tot += y.numel()
    return c0 / tot, c1 / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="trained YearCycleGAN checkpoint")
    p.add_argument("--objects", nargs="+",
                   default=["Boy", "Girl", "Woman", "Man", "Person", "House", "Car"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--train_limit", type=int, default=15000,
                   help="#source crops used to build each training set (RAM-bound)")
    p.add_argument("--test_limit", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone", choices=["resnet", "convnext"], default="resnet")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_src, avlabels = build(args.objects, args.img_size, evaluate=False, flip=False)
    test_ds, _ = build(args.objects, args.img_size, evaluate=True, flip=False)
    n_years = len(avlabels)
    train_src = subset(train_src, args.train_limit)
    test_ds = subset(test_ds, args.test_limit)
    print(f"TSTR | {n_years} buckets | train {len(train_src)} | test {len(test_ds)}", flush=True)

    src_loader = DataLoader(train_src, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=collate_xy)

    from gan_models import YearCycleGAN
    ck = torch.load(args.ckpt, map_location=device); a = ck.get("args", {})
    model = YearCycleGAN(n_years=n_years, ngf=a.get("ngf", 64), ndf=a.get("ndf", 64),
                         n_blocks=a.get("n_blocks", 9), style_dim=a.get("style_dim", 64),
                         latent_dim=a.get("latent_dim", 64),
                         color_head=bool(a.get("color_head")),
                         color_classifier=bool(a.get("color_classifier"))).to(device)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()

    print("materialising synthetic (year-swapped) and identity-control sets...", flush=True)
    synth = materialize(model, src_loader, n_years, device, translate=True)
    ident = materialize(model, src_loader, n_years, device, translate=False)

    def mk(ds):
        return lambda: DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_xy)
    kw = dict(test_loader=test_loader, n_years=n_years, device=device,
              epochs=args.epochs, lr=args.lr, backbone=args.backbone)

    real_xy = XY(train_src)
    print("training REAL reference (TRTR)...", flush=True)
    real0, real1 = train_eval(mk(real_xy), **kw)
    print("training SYNTH (TSTR)...", flush=True)
    syn0, syn1 = train_eval(mk(synth), **kw)
    print("training IDENTITY control...", flush=True)
    idt0, idt1 = train_eval(mk(ident), **kw)
    print("training AUGMENT (real+synth)...", flush=True)
    aug0, aug1 = train_eval(mk(ConcatDataset([real_xy, synth])), **kw)

    chance = 1.0 / n_years
    print("\n================ TSTR results (test = real held-out) ================")
    print(f"{'train set':<22}{'acc@0':>8}{'acc@1':>8}")
    for name, (a0, a1) in [("real (TRTR ref)", (real0, real1)),
                           ("synth (TSTR)", (syn0, syn1)),
                           ("identity (control)", (idt0, idt1)),
                           ("real+synth (augment)", (aug0, aug1))]:
        print(f"{name:<22}{a0:>8.3f}{a1:>8.3f}")
    print(f"{'chance':<22}{chance:>8.3f}")
    print(f"\nTSTR gap (real-synth) acc@0 = {real0 - syn0:+.3f}  (small => faithful re-dating)")
    print(f"identity control acc@0    = {idt0:.3f}  (should be ~chance {chance:.3f})")


if __name__ == "__main__":
    main()
