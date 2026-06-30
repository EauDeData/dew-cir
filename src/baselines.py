"""
baselines.py

Non-learned re-dating baselines. The point of these is to answer the reviewer's
first question: "is your GAN doing anything a global colour operation can't?"
The year signal is largely photographic colour (palette / white balance /
contrast / sepia-vs-cool / grain), so the natural lower bounds are classical
colour-transfer methods that move a crop's colour statistics toward those of the
*target year bucket*, estimated from real images of that bucket.

Baselines
---------
  * Reinhard colour transfer (Lab mean/std matching)        [Reinhard 2001]
  * per-channel histogram matching to the target bucket
  * grayscale / fixed sepia (era cliché lower bound)

`BucketColorStats` precomputes, for each of the K year buckets, the average Lab
mean/std and a channel histogram over real images, so any of the baselines can
"translate" an arbitrary crop to any target bucket — directly comparable to
`YearCycleGAN.translate_to_year`.

Images are float numpy (H,W,3) RGB in [0,1] (see eval_color.to_numpy_imgs).
"""

import numpy as np
import cv2


# ── target-bucket colour statistics from real data ──────────────────────────────

class BucketColorStats:
    """Per-bucket Lab mean/std and per-channel CDF, estimated from real images."""

    def __init__(self, n_years, hist_bins=256):
        self.n_years = n_years
        self.bins = hist_bins
        self.lab_mean = [None] * n_years
        self.lab_std = [None] * n_years
        self.cdf = [None] * n_years          # per channel CDF in RGB

    def fit(self, imgs_by_year):
        """imgs_by_year: list of length K, each a list of (H,W,3) [0,1] images."""
        for y, imgs in enumerate(imgs_by_year):
            if not imgs:
                continue
            labs = np.concatenate([cv2.cvtColor(im.astype(np.float32),
                                   cv2.COLOR_RGB2Lab).reshape(-1, 3) for im in imgs], 0)
            self.lab_mean[y] = labs.mean(0)
            self.lab_std[y] = labs.std(0) + 1e-6
            rgb = np.concatenate([im.reshape(-1, 3) for im in imgs], 0)
            cdfs = []
            for c in range(3):
                h, _ = np.histogram(rgb[:, c], bins=self.bins, range=(0, 1))
                cdf = np.cumsum(h).astype(np.float64)
                cdfs.append(cdf / cdf[-1])
            self.cdf[y] = cdfs
        return self


# ── Reinhard colour transfer ────────────────────────────────────────────────────

def reinhard_to_bucket(img, stats: BucketColorStats, year):
    """Match the crop's Lab mean/std to the target bucket's (Reinhard 2001)."""
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)
    src_m, src_s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = (lab - src_m) / src_s * stats.lab_std[year] + stats.lab_mean[year]
    rgb = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(rgb, 0, 1)


# ── per-channel histogram matching ──────────────────────────────────────────────

def histmatch_to_bucket(img, stats: BucketColorStats, year):
    out = np.empty_like(img)
    bins = stats.bins
    for c in range(3):
        src = img[..., c].ravel()
        h, _ = np.histogram(src, bins=bins, range=(0, 1))
        src_cdf = np.cumsum(h).astype(np.float64); src_cdf /= src_cdf[-1]
        tgt_cdf = stats.cdf[year][c]
        # map source intensity -> quantile -> target intensity
        src_q = np.interp(src, np.linspace(0, 1, bins), src_cdf)
        mapped = np.interp(src_q, tgt_cdf, np.linspace(0, 1, bins))
        out[..., c] = mapped.reshape(img.shape[:2])
    return np.clip(out, 0, 1)


# ── era clichés (lower bound) ───────────────────────────────────────────────────

def to_grayscale(img):
    g = (img * np.array([0.299, 0.587, 0.114])).sum(2, keepdims=True)
    return np.repeat(g, 3, axis=2)


def to_sepia(img):
    M = np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]])
    return np.clip(img @ M.T, 0, 1)


# ── unified API mirroring the GAN ───────────────────────────────────────────────

BASELINES = {
    "reinhard": reinhard_to_bucket,
    "histmatch": histmatch_to_bucket,
}


def translate_to_year(img, year, method="reinhard", stats=None):
    """Drop-in analogue of YearCycleGAN.translate_to_year for a baseline."""
    if method == "grayscale":
        return to_grayscale(img)
    if method == "sepia":
        return to_sepia(img)
    assert stats is not None, f"method '{method}' needs BucketColorStats"
    return BASELINES[method](img, stats, year)


if __name__ == "__main__":
    import sys
    from PIL import Image
    from eval_color import colourfulness, palette_delta_e
    path = sys.argv[1] if len(sys.argv) > 1 else "paper_gan/figures/fig_assets/x.png"
    rgb = np.asarray(Image.open(path).convert("RGB").resize((128, 128))) / 255.0
    # fake a 2-bucket world: bucket 0 = sepia-ish references, bucket 1 = the colour crop
    stats = BucketColorStats(2).fit([[to_sepia(rgb)], [rgb]])
    out0 = translate_to_year(rgb, 0, "reinhard", stats)   # -> should warm/desaturate
    out1 = translate_to_year(rgb, 1, "histmatch", stats)  # -> should stay colourful
    print(f"src colourfulness        : {colourfulness(rgb):6.2f}")
    print(f"reinhard -> bucket0(sepia): {colourfulness(out0):6.2f} "
          f"(ΔE00 to sepia ref {palette_delta_e([out0], [to_sepia(rgb)]):.2f})")
    print(f"histmatch -> bucket1(col) : {colourfulness(out1):6.2f}")
    print("OK")
