"""
train_oracle.py

Train the INDEPENDENT date oracle D* used to score re-dating accuracy in
eval_cyclegan.py. It is a plain K-way (year-bucket) classifier on real crops,
trained with cross-entropy — deliberately NOT the GAN discriminator and NOT the
metric-learning proxy model that seeds the latent means, so the evaluation is not
circular.

Images use the same [-1,1] convention as the GAN; DateOracle re-normalises to
ImageNet stats internally, so train-time and eval-time preprocessing match.

  CUDA_VISIBLE_DEVICES=6 python train_oracle.py \
      --objects Boy Girl Woman Man Person House Car \
      --limit 30000 --test_limit 4000 --epochs 3 --out runs/oracle
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
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
    avlabels = parts[0].available_labels
    return ConcatDataset(parts), avlabels


def subset(ds, limit, seed=0):
    if limit is None or limit >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:limit].tolist()
    return Subset(ds, idx)


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for img, year, _ in loader:
        pred = model.predict(img.to(device))
        correct += (pred == year.to(device)).sum().item()
        total += year.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", nargs="+",
                   default=["Boy", "Girl", "Woman", "Man", "Person", "House", "Car"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=30000)
    p.add_argument("--test_limit", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone", choices=["resnet", "convnext"], default="resnet")
    p.add_argument("--out", type=str, default="runs/oracle")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    train_ds, avlabels = build(args.objects, args.img_size, evaluate=False, flip=True)
    test_ds, _ = build(args.objects, args.img_size, evaluate=True, flip=False)
    n_years = len(avlabels)
    train_ds = subset(train_ds, args.limit)
    test_ds = subset(test_ds, args.test_limit)
    print(f"oracle: {n_years} buckets | train {len(train_ds)} | test {len(test_ds)} "
          f"| backbone {args.backbone}", flush=True)

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, pin_memory=True)
    vl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    model = DateOracle(n_years, backbone=args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for it, (img, year, _) in enumerate(tl):
            img, year = img.to(device), year.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(img), year)
            loss.backward(); opt.step()
            running += loss.item()
            if it % 50 == 0:
                print(f"  e{ep} it{it}/{len(tl)} loss {loss.item():.3f}", flush=True)
        acc = evaluate_acc(model, vl, device)
        # off-by-one tolerant accuracy is the fairer metric for ordinal year buckets
        print(f"epoch {ep}: train_loss {running/len(tl):.3f} | test acc@0 {acc:.3f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "n_years": n_years,
                    "avlabels": avlabels, "backbone": args.backbone,
                    "args": vars(args), "epoch": ep, "test_acc": acc},
                   os.path.join(args.out, "oracle.pth"))
        best = max(best, acc)
    print(f"done. best test acc@0 {best:.3f} → {args.out}/oracle.pth", flush=True)


if __name__ == "__main__":
    main()
