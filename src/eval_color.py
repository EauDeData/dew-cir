"""
eval_color.py

Colour-science evaluation for the year-translation GAN, using the metrics the
colour-transfer / recolouring / colorization communities actually report. The
year signal in this dataset is mostly *photographic colour*: palette, white
balance, contrast/dynamic range, channel correlation (sepia vs. cool), and film
grain. So "did we re-date the image" is, to first order, "does the output's
colour signature match the colour signature of real photos from the target year".

Because the problem is *unpaired* (no object exists in two years), we never
compare a generated image to a pixel-aligned reference. Everything here is either
(a) a per-image colour descriptor whose *distribution* we compare to the real
target bucket, or (b) a perceptual colour difference between matched summaries
(e.g. mean palettes / sorted percentiles).

Implemented (no skimage dependency; uses cv2 + numpy + scipy):
  * sRGB -> CIELAB conversion (D65)
  * CIEDE2000 (ΔE00) perceptual colour difference   [Sharma et al. 2005]
  * colourfulness                                    [Hasler & Süsstrunk 2003]
  * colour signature (palette/contrast/correlation/grain), and the Fréchet
    distance between signature distributions ("colour-FID")
  * hue-histogram Earth Mover's Distance             [Rubner et al. 2000]
  * luminance-SSIM / chroma-ΔE split for content-vs-colour separation

All functions take images as float arrays in [0, 1], shape (H, W, 3), RGB, or a
list/stack of them. Use `to_numpy_imgs` to convert from the GAN's [-1, 1] tensors.
"""

import numpy as np
import cv2
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm


# ── conversion helpers ─────────────────────────────────────────────────────────

def to_numpy_imgs(x):
    """(B,3,H,W) torch tensor in [-1,1]  ->  (B,H,W,3) float numpy in [0,1] RGB."""
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        x = (x.clamp(-1, 1) + 1) / 2
        return x.permute(0, 2, 3, 1).numpy()
    return np.asarray(x)


def rgb_to_lab(img):
    """sRGB [0,1] (H,W,3) -> CIELAB, L in [0,100], a,b ~ [-128,127]."""
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)


# ── CIEDE2000 (ΔE00) ───────────────────────────────────────────────────────────

def ciede2000(lab1, lab2):
    """Perceptual colour difference between two arrays of Lab triplets (...,3).

    Reference implementation of Sharma, Wu & Dalal (2005). Returns ΔE00 per
    sample (broadcasts over leading dims)."""
    lab1 = np.asarray(lab1, float); lab2 = np.asarray(lab2, float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbar = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) > 180, (hsum + 360) / 2, hsum / 2)
    hbar = np.where((C1p * C2p) == 0, hsum, hbar)

    T = (1 - 0.17 * np.cos(np.radians(hbar - 30))
           + 0.24 * np.cos(np.radians(2 * hbar))
           + 0.32 * np.cos(np.radians(3 * hbar + 6))
           - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    Sl = 1 + (0.015 * (Lbar - 50)**2) / np.sqrt(20 + (Lbar - 50)**2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    dtheta = 30 * np.exp(-(((hbar - 275) / 25)**2))
    Rc = 2 * np.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7))
    Rt = -Rc * np.sin(np.radians(2 * dtheta))

    return np.sqrt((dLp / Sl)**2 + (dCp / Sc)**2 + (dHp / Sh)**2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def palette_delta_e(gen_imgs, real_imgs, n_levels=8):
    """ΔE00 between the *tonal palettes* of two (unpaired) image sets.

    We sort each set's pixels by lightness, split into `n_levels` luminance
    bands, average the Lab colour in each band, and compare matched bands with
    ΔE00. This compares 'what colour is the shadow/mid/highlight region' between
    the generated target bucket and the real target bucket, which is exactly what
    a tone/colour transfer should get right. Returns mean ΔE00 over bands."""
    def palette(imgs):
        labs = np.concatenate([rgb_to_lab(im).reshape(-1, 3) for im in imgs], 0)
        order = np.argsort(labs[:, 0])
        labs = labs[order]
        bands = np.array_split(labs, n_levels)
        return np.stack([b.mean(0) for b in bands], 0)         # (n_levels, 3)
    return float(ciede2000(palette(gen_imgs), palette(real_imgs)).mean())


# ── colourfulness (Hasler & Süsstrunk 2003) ────────────────────────────────────

def colourfulness(img):
    """Single scalar 'how colourful' (on 0..~150). Old b/w-ish buckets score low,
    saturated modern photos high; we compare its *distribution* across buckets."""
    R, G, B = (img[..., 0] * 255, img[..., 1] * 255, img[..., 2] * 255)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std = np.hypot(rg.std(), yb.std())
    mean = np.hypot(rg.mean(), yb.mean())
    return float(std + 0.3 * mean)


# ── colour signature + Fréchet distance ("colour-FID") ──────────────────────────

def colour_signature(img):
    """Compact 11-D photographic-colour descriptor of one image.

    [Lab mean (3), Lab std (3) = contrast, chroma mean (1), RG/RB/GB sRGB channel
     correlation (3), grain = high-freq luminance energy (1)]."""
    lab = rgb_to_lab(img).reshape(-1, 3)
    mean, std = lab.mean(0), lab.std(0)
    chroma = np.hypot(lab[:, 1], lab[:, 2]).mean()
    rgb = img.reshape(-1, 3)
    c = rgb - rgb.mean(0)
    corr = np.array([(c[:, 0] * c[:, 1]).mean(),
                     (c[:, 0] * c[:, 2]).mean(),
                     (c[:, 1] * c[:, 2]).mean()])
    lum = img.mean(2)
    grain = float((lum - cv2.blur(lum, (3, 3))).std())
    return np.concatenate([mean, std, [chroma], corr, [grain]]).astype(np.float64)


def _frechet(mu1, cov1, mu2, cov2, eps=1e-6):
    """Fréchet distance between two Gaussians, with a diagonal regulariser so a
    degenerate (rank-deficient) covariance — e.g. a baseline that maps all images
    through one fixed colour matrix — does not blow up sqrtm."""
    diff = mu1 - mu2
    cov1 = cov1 + eps * np.eye(len(cov1))
    cov2 = cov2 + eps * np.eye(len(cov2))
    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        covmean = np.zeros_like(cov1)
    return float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))


def colour_fid(gen_imgs, real_imgs):
    """Fréchet distance between the Gaussians fit to the colour signatures of the
    generated target bucket vs. the real target bucket. A low value means the
    *whole distribution* of photographic-colour attributes matches, not just the
    mean — the colour-space analogue of FID, but interpretable per-attribute."""
    if len(gen_imgs) < 2 or len(real_imgs) < 2:
        return float("nan")
    Sg = np.stack([colour_signature(im) for im in gen_imgs], 0)
    Sr = np.stack([colour_signature(im) for im in real_imgs], 0)
    return _frechet(Sg.mean(0), np.cov(Sg, rowvar=False),
                    Sr.mean(0), np.cov(Sr, rowvar=False))


# ── hue-histogram EMD ───────────────────────────────────────────────────────────

def hue_emd(gen_imgs, real_imgs, bins=36, sat_thresh=0.1):
    """Earth Mover's Distance between hue distributions (ignoring near-gray
    pixels). Captures whether the *palette* (which hues are present, e.g. warm
    sepia vs. neutral) matches the target bucket. Hue is circular, so we report
    the min over the cyclic shift."""
    def hue_hist(imgs):
        hs = []
        for im in imgs:
            hsv = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_RGB2HSV)
            h, s = hsv[..., 0].ravel() / 360.0, hsv[..., 1].ravel()
            hs.append(h[s > sat_thresh])
        h = np.concatenate(hs) if hs else np.zeros(0)
        hist, _ = np.histogram(h, bins=bins, range=(0, 1))
        # uniform floor keeps weights positive (scipy requires it) and makes a
        # near-grayscale set read as a flat hue distribution rather than crashing
        return hist.astype(np.float64) + 1e-6
    hg, hr = hue_hist(gen_imgs), hue_hist(real_imgs)
    pos = np.arange(bins)
    return float(min(wasserstein_distance(pos, pos, np.roll(hg, k), hr)
                     for k in range(bins)))


# ── content vs. colour: luminance structure should survive a colour change ──────

def lum_ssim(img_a, img_b):
    """SSIM on the L channel only — structure preserved (paired: x vs G(x,y))."""
    a = rgb_to_lab(img_a)[..., 0]; b = rgb_to_lab(img_b)[..., 0]
    C1, C2 = (0.01 * 100)**2, (0.03 * 100)**2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a**2
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b**2
    vab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / \
        ((mu_a**2 + mu_b**2 + C1) * (va + vb + C2))
    return float(s.mean())


# ── smoke test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from PIL import Image
    path = sys.argv[1] if len(sys.argv) > 1 else "paper_gan/figures/fig_assets/x.png"
    rgb = np.asarray(Image.open(path).convert("RGB").resize((128, 128))) / 255.0
    sepia = np.clip(rgb @ np.array([[.393, .769, .189],
                                    [.349, .686, .168],
                                    [.272, .534, .131]]).T, 0, 1)
    print(f"colourfulness  colour={colourfulness(rgb):6.2f}  sepia={colourfulness(sepia):6.2f}")
    print(f"palette ΔE00 (colour vs sepia)  : {palette_delta_e([rgb], [sepia]):6.2f}")
    print(f"colour-FID    (colour vs sepia) : {colour_fid([rgb, rgb], [sepia, sepia]):8.3f}")
    print(f"hue EMD       (colour vs sepia) : {hue_emd([rgb], [sepia]):6.3f}")
    print(f"lum-SSIM      (colour vs sepia) : {lum_ssim(rgb, sepia):6.3f}  (should be ~1: structure kept)")
    print("OK")
