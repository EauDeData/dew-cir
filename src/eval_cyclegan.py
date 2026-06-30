"""
eval_cyclegan.py

End-to-end evaluation of the year-translation model (gan_models.YearCycleGAN) or
any baseline (baselines.translate_to_year). There is no paired ground truth, so we
score three axes plus a downstream check:

  A. Re-dating accuracy  — an INDEPENDENT date oracle D* must read G(x, y_t) as
                           year y_t.  acc@0, acc@1 (±1 bucket), MAE in years,
                           and the full source->target confusion matrix.
  B. Realism             — FID (Inception) and colour-FID per target bucket vs the
                           real images of that bucket; lower is better.
  C. Colour fidelity     — the colour-science metrics in eval_color (palette ΔE00,
                           hue EMD, colourfulness trend across decades).
  D. Content preservation— cycle SSIM + luminance-SSIM(x, G(x,y)), and the cosine
                           similarity of the year-agnostic object embedding.
  E. TSTR (downstream)   — train a date estimator on YEAR-SWAPPED synthetic images
                           and test on REAL images. If the temporal appearance is
                           encoded correctly, accuracy should approach train-on-real.

The oracle D* must be trained separately and must NOT be the GAN discriminator or
the proxy checkpoint used to seed the latent means, or the metric is circular.

Run:
  python eval_cyclegan.py --ckpt runs/yearcgan/ckpt_epoch99.pth \
       --oracle runs/oracle/oracle.pth --objects Car House --limit 2000
  python eval_cyclegan.py --baseline reinhard --objects Car --limit 2000   # baseline
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval_color as C
import baselines as B


# ══════════════════════════════════════════════════════════════════════════════
# Independent date oracle  D*
# ══════════════════════════════════════════════════════════════════════════════

class DateOracle(nn.Module):
    """A plain K-way date classifier on object crops, independent of the GAN.

    Backbone reuses models.SpecialistModel (ConvNeXt) with a K-way head. Train it
    on REAL crops only (see `fit`), or load a checkpoint. Input: images in [-1,1]
    (the GAN convention) — we re-normalise to ImageNet stats internally."""

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, n_years, backbone="resnet"):
        super().__init__()
        from models import SpecialistModel, ResNetSpecialistModel
        self.net = (ResNetSpecialistModel(n_years) if backbone == "resnet"
                    else SpecialistModel(n_years))

    def _prep(self, x):  # [-1,1] -> ImageNet-normalised
        x = (x.clamp(-1, 1) + 1) / 2
        return (x - self.IMAGENET_MEAN.to(x)) / self.IMAGENET_STD.to(x)

    def forward(self, x):
        return self.net(self._prep(x))

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(1)


def train_oracle(model, loader, epochs, device, lr=1e-4, label_index=1, max_steps=None):
    """Generic CE training; `label_index` selects the year label from the batch
    tuple (the real-data loader yields (img, cond_A, img_B, cond_B, cat); the
    synthetic TSTR loader yields (img, year))."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    model.train()
    step = 0
    for ep in range(epochs):
        for batch in loader:
            x = batch[0].to(device)
            y = batch[label_index].to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward(); opt.step()
            step += 1
            if max_steps and step >= max_steps:
                return model
    return model


@torch.no_grad()
def oracle_eval(model, loader, device, label_index=1):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[label_index].to(device)
        correct += (model.predict(x) == y).sum().item(); total += y.numel()
    return correct / max(total, 1)


# ══════════════════════════════════════════════════════════════════════════════
# A. re-dating accuracy
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def redating_metrics(oracle, gen_by_bucket, device, freq=5):
    """gen_by_bucket[k] = tensor (n,3,H,W) of crops translated *to* bucket k."""
    K = len(gen_by_bucket)
    conf = np.zeros((K, K))
    for k, batch in enumerate(gen_by_bucket):
        if batch.numel() == 0:
            continue
        for i in range(0, len(batch), 64):
            pred = oracle.predict(batch[i:i + 64].to(device)).cpu().numpy()
            for p in pred:
                conf[k, p] += 1
    row = conf.sum(1, keepdims=True); row[row == 0] = 1
    P = conf / row
    acc0 = np.trace(conf) / conf.sum()
    offby1 = sum(conf[k, max(0, k - 1):k + 2].sum() for k in range(K)) / conf.sum()
    yhat = conf.argmax(1)
    mae_buckets = np.average(np.abs(np.arange(K) - (conf * np.arange(K)).sum(1) /
                                    conf.sum(1).clip(1)), weights=conf.sum(1))
    return {"acc@0": acc0, "acc@1": offby1, "MAE_years": mae_buckets * freq,
            "confusion": P}


# ══════════════════════════════════════════════════════════════════════════════
# B. realism — FID with a pluggable feature extractor (Inception by default)
# ══════════════════════════════════════════════════════════════════════════════

class InceptionFeatures(nn.Module):
    """2048-d pool3 Inception-V3 features for FID. Falls back gracefully if the
    pretrained weights are unavailable offline (then use colour-FID instead)."""

    def __init__(self):
        super().__init__()
        import torchvision
        try:
            net = torchvision.models.inception_v3(weights="IMAGENET1K_V1",
                                                  aux_logits=True)
        except Exception as e:
            raise RuntimeError(f"Inception weights unavailable ({e}); "
                               "use colour_fid or provide a local extractor.")
        net.fc = nn.Identity(); net.eval()
        self.net = net

    @torch.no_grad()
    def __call__(self, x):  # x in [-1,1]
        x = F.interpolate((x.clamp(-1, 1) + 1) / 2, size=299, mode="bilinear",
                          align_corners=False)
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x)) / \
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x)
        return self.net(x)


def fid(feat_extractor, gen, real, device, bs=64):
    def acts(batch):
        out = []
        for i in range(0, len(batch), bs):
            out.append(feat_extractor(batch[i:i + bs].to(device)).cpu().numpy())
        return np.concatenate(out, 0)
    g, r = acts(gen), acts(real)
    return C._frechet(g.mean(0), np.cov(g, rowvar=False),
                      r.mean(0), np.cov(r, rowvar=False))


# ══════════════════════════════════════════════════════════════════════════════
# C+B colour fidelity per bucket  &  D content preservation
# ══════════════════════════════════════════════════════════════════════════════

def colour_report(gen_by_bucket, real_by_bucket):
    """Per-bucket colour-science metrics + the colourfulness-vs-year trend."""
    rows = {}
    cf_gen, cf_real = [], []
    for k, (g, r) in enumerate(zip(gen_by_bucket, real_by_bucket)):
        if len(g) < 2 or len(r) < 2:          # need >=2 for a covariance
            continue
        gi = C.to_numpy_imgs(g); ri = C.to_numpy_imgs(r)
        rows[k] = {
            "colour_FID": C.colour_fid(gi, ri),
            "palette_dE00": C.palette_delta_e(gi, ri),
            "hue_EMD": C.hue_emd(gi, ri),
        }
        cf_gen.append(np.mean([C.colourfulness(im) for im in gi]))
        cf_real.append(np.mean([C.colourfulness(im) for im in ri]))
    # does generated colourfulness track the real trend across decades?
    trend_corr = float(np.corrcoef(cf_gen, cf_real)[0, 1]) if len(cf_gen) > 1 else float("nan")
    return rows, {"colourfulness_trend_corr": trend_corr,
                  "colourfulness_gen": cf_gen, "colourfulness_real": cf_real}


@torch.no_grad()
def content_metrics(model, encoder, x, y_origin, device, n_years):
    """Cycle SSIM + year-agnostic embedding cosine (content should survive)."""
    x = x.to(device)
    ssims, coss = [], []
    xn = C.to_numpy_imgs(x)
    for k in range(n_years):
        yt = torch.full((len(x),), k, device=device, dtype=torch.long)
        xf = model.translate(x, yt, stochastic=False)[0]
        xr = model.translate(xf, y_origin.to(device), stochastic=False)[0]
        xrn = C.to_numpy_imgs(xr)
        ssims += [C.lum_ssim(a, b) for a, b in zip(xn, xrn)]      # x vs x_rec
        if encoder is not None:
            fa = F.normalize(encoder(x), dim=1)
            fb = F.normalize(encoder(xf), dim=1)
            coss += (fa * fb).sum(1).cpu().tolist()
    return {"cycle_lumSSIM": float(np.mean(ssims)),
            "content_cosine": float(np.mean(coss)) if coss else float("nan")}


# ══════════════════════════════════════════════════════════════════════════════
# E. TSTR — train a date estimator on year-swapped synthetic, test on real
# ══════════════════════════════════════════════════════════════════════════════

class SwappedSynthetic(torch.utils.data.Dataset):
    """Wrap a real dataset; each item is translated to a uniformly random target
    year and returned with that target as the label. Materialise once for a fair,
    fixed synthetic training set."""

    def __init__(self, real_ds, model, n_years, device, label_index=1):
        self.items = []
        model.eval()
        with torch.no_grad():
            for i in range(len(real_ds)):
                x = real_ds[i][0].unsqueeze(0).to(device)
                yt = np.random.randint(n_years)
                xf = model.translate_to_year(x, yt, stochastic=True).squeeze(0).cpu()
                self.items.append((xf, yt))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def run_tstr(model, train_ds, test_loader, n_years, device, epochs=3):
    """Returns (acc_synth_train, acc_real_train_reference)."""
    from torch.utils.data import DataLoader
    synth = SwappedSynthetic(train_ds, model, n_years, device)
    synth_loader = DataLoader(synth, batch_size=32, shuffle=True)

    o_syn = DateOracle(n_years).to(device)
    o_syn = train_oracle(o_syn, synth_loader, epochs, device, label_index=1)
    acc_syn = oracle_eval(o_syn, test_loader, device, label_index=1)   # test on REAL

    o_real = DateOracle(n_years).to(device)
    real_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    o_real = train_oracle(o_real, real_loader, epochs, device, label_index=1)
    acc_real = oracle_eval(o_real, test_loader, device, label_index=1)
    return {"TSTR_acc": acc_syn, "train_on_real_acc": acc_real,
            "TSTR_gap": acc_real - acc_syn}


# ══════════════════════════════════════════════════════════════════════════════
# driver
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gather(translate_batch, test_imgs, n_years, device, bs=64):
    """Return gen_by_bucket: list[K] of stacked translations to each bucket.
    `translate_batch(xb, k)` maps a batch of crops to target bucket k."""
    out = []
    for k in range(n_years):
        chunks = []
        for i in range(0, len(test_imgs), bs):
            xb = torch.stack(test_imgs[i:i + bs]).to(device)
            chunks.append(translate_batch(xb, k).cpu())
        out.append(torch.cat(chunks) if chunks else torch.empty(0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="YearCycleGAN checkpoint")
    p.add_argument("--baseline", choices=["reinhard", "histmatch", "sepia", "grayscale"],
                   default=None)
    p.add_argument("--oracle", type=str, default=None, help="DateOracle checkpoint")
    p.add_argument("--objects", nargs="+", default=["Car"])
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--no_fid", action="store_true", help="skip Inception FID (offline)")
    p.add_argument("--tstr", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train_year_cyclegan import build_dataset
    ds, avlabels = build_dataset(args.objects, args.img_size, args.limit)
    n_years = len(avlabels)
    test_imgs = [ds[i][0] for i in range(len(ds))]

    # real images grouped by their true bucket (image_A's year = condition_B, index 3)
    real = [[] for _ in range(n_years)]
    for i in range(len(ds)):
        item = ds[i]
        real[int(item[3])].append(item[0])
    real = [torch.stack(r) if r else torch.empty(0) for r in real]

    # translator: GAN (batched) or baseline (per-image numpy)
    if args.ckpt:
        from gan_models import YearCycleGAN
        ck = torch.load(args.ckpt, map_location=device)
        a = ck.get("args", {})
        model = YearCycleGAN(
            n_years=n_years, ngf=a.get("ngf", 64), ndf=a.get("ndf", 64),
            n_blocks=a.get("n_blocks", 9), style_dim=a.get("style_dim", 64),
            latent_dim=a.get("latent_dim", 64),
            color_head=bool(a.get("color_head")),
            color_classifier=bool(a.get("color_classifier")),
            style_mode=a.get("style_mode", "variational"),
        ).to(device)
        model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        model.eval()
        def translate(xb, k):
            return model.translate_to_year(xb, k, stochastic=False)
    else:
        stats = None
        if args.baseline in ("reinhard", "histmatch"):
            stats = B.BucketColorStats(n_years).fit(
                [[C.to_numpy_imgs(im.unsqueeze(0))[0] for im in r] if len(r) else []
                 for r in real])
        def translate(xb, k):
            outs = [B.translate_to_year(im, k, args.baseline, stats)
                    for im in C.to_numpy_imgs(xb)]
            t = torch.from_numpy(np.stack(outs)).permute(0, 3, 1, 2).float()
            return t * 2 - 1

    gen = gather(translate, test_imgs, n_years, device)

    tag = args.baseline or os.path.basename(args.ckpt)
    print(f"== {tag} | {args.objects} | {n_years} buckets | {len(ds)} crops ==")

    # [A] re-dating accuracy (independent oracle)
    if args.oracle:
        ock = torch.load(args.oracle, map_location=device)
        oracle = DateOracle(n_years, backbone=ock.get("backbone", "resnet")).to(device)
        oracle.load_state_dict(ock["state_dict"] if "state_dict" in ock else ock)
        oracle.eval()
        rd = redating_metrics(oracle, gen, device)
        print(f"[A] re-dating: acc@0={rd['acc@0']:.3f} acc@1={rd['acc@1']:.3f} "
              f"MAE={rd['MAE_years']:.1f} yr")

    # [B] Inception FID (offline weights may be unavailable)
    if not args.no_fid:
        try:
            inc = InceptionFeatures().to(device)
            fids = [fid(inc, gen[k], real[k], device)
                    for k in range(n_years) if len(real[k]) > 1]
            print(f"[B] mean bucket-FID = {np.mean(fids):.2f}")
        except RuntimeError as e:
            print(f"[B] FID skipped: {e}")

    # [C] colour fidelity
    rows, trend = colour_report(gen, real)
    print(f"[C] colour-FID={np.mean([r['colour_FID'] for r in rows.values()]):.2f} "
          f"palette ΔE00={np.mean([r['palette_dE00'] for r in rows.values()]):.2f} "
          f"hue-EMD={np.mean([r['hue_EMD'] for r in rows.values()]):.3f} "
          f"colourfulness-trend r={trend['colourfulness_trend_corr']:.3f}")

    # [D] content preservation (GAN only: needs the cycle)
    if args.ckpt:
        nc = min(128, len(ds))
        xc = torch.stack([ds[i][0] for i in range(nc)])
        yo = torch.tensor([int(ds[i][3]) for i in range(nc)])
        cm = content_metrics(model, None, xc, yo, device, n_years)
        print(f"[D] content: cycle-lumSSIM={cm['cycle_lumSSIM']:.3f} "
              f"content-cosine={cm['content_cosine']}")


if __name__ == "__main__":
    main()
