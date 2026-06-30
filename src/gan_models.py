"""
gan_models.py

Year-translation GANs. Kept deliberately separate from the clean
representation-learning code in models.py.

Goal
----
Translate an image from its origin year-bucket into a chosen destination
year-bucket, with the structure of CycleGAN (Zhu et al. 2017, Fig. 3): two
mapping directions tied by *cycle consistency*, two adversarial signals.

Why this is StarGAN-shaped, not a literal 2-net CycleGAN
--------------------------------------------------------
Vanilla CycleGAN handles exactly two domains X<->Y with two generators
(G, F) and two discriminators (D_Y, D_X). We have 14 year buckets, so the
faithful N-domain generalization is StarGAN (Choi et al. 2018): ONE generator
conditioned on a target year, ONE discriminator that additionally classifies
the year. The cycle is identical to CycleGAN:

    forward cycle :  x --G(., y_t)--> x_fake --G(., y_o)--> x_rec  ~= x
    identity      :  G(x, y_o) ~= x

This mirrors the metric-learning "year swap" trick used by the dataloader
(image_A + condition_A = image_B): translating to a year and back should return
the original.

Variational year style (answer to "can we use a variational latent to
instantiate the generator to a specific year?")
----------------------------------------------------------------------
Each year is encoded as a small Gaussian q(z|y) = N(mu_y, sigma_y^2). We sample
z, map it to a style code w, and inject w into the generator's residual blocks
via AdaIN (StarGAN-v2 style). A KL term regularizes q(z|y).

This is the probabilistic analogue of the metric-learning year proxies mu^y:
there a year is a single point; here it is a compact distribution. That gives:
  * deterministic targeting of a year  -> use the mean mu_y (stochastic=False),
  * intra-year style variation         -> sample z ~ N(mu_y, sigma_y^2),
  * unseen / in-between years          -> interpolate mu between buckets;
                                          KL keeps the latent continuous.

SOTA CycleGAN practices included
--------------------------------
  * ResNet generator (reflection padding + InstanceNorm), Johnson et al. arch.
  * 70x70 PatchGAN discriminator.
  * LSGAN (least-squares) adversarial loss.
  * Image buffer (Shrivastava et al.) to stabilize D.
  * Identity loss, cycle-consistency loss.
  * Normal(0, 0.02) weight init.

Image convention: generators end in tanh, so images live in [-1, 1]. Feed the
network images normalized with mean=std=0.5 and invert for visualization.
"""

import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# init helpers
# ──────────────────────────────────────────────────────────────────────────────

def init_weights(net, init_gain=0.02):
    """CycleGAN's normal(0, 0.02) initialization."""
    def init_func(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm" in cls and getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    return net


_IN = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)


# ──────────────────────────────────────────────────────────────────────────────
# Colorimetric statistics  (the attributes the rest of the net tends to ignore)
# ──────────────────────────────────────────────────────────────────────────────

def color_moments(x):
    """Differentiable global colour signature of a batch of images in [-1, 1].

    InstanceNorm — used throughout the generator and discriminator — normalizes
    away per-channel mean/variance, i.e. exactly the *palette / illuminant /
    contrast* information. To make the model attend to it we measure it
    explicitly and feed it to a dedicated year classifier (see
    `ColorYearClassifier`).

    Returns (B, 10):
      * mean   (3) — palette / colour cast / illuminant white point
      * std    (3) — per-channel contrast / dynamic range
      * corr   (3) — RG/RB/GB channel correlations (overall colour balance,
                     e.g. sepia ties R-G-B together, cool tones anti-correlate)
      * grain  (1) — high-frequency energy (film grain in older buckets)
    """
    b = x.size(0)
    f = x.reshape(b, 3, -1)
    mean = f.mean(dim=2)                                   # (B, 3)
    std = f.std(dim=2)                                     # (B, 3)
    fc = f - mean.unsqueeze(2)
    rg = (fc[:, 0] * fc[:, 1]).mean(1)
    rb = (fc[:, 0] * fc[:, 2]).mean(1)
    gb = (fc[:, 1] * fc[:, 2]).mean(1)
    corr = torch.stack([rg, rb, gb], dim=1)               # (B, 3)
    grain = (x - F.avg_pool2d(x, 3, 1, 1)).reshape(b, 3, -1).std(dim=2).mean(1, keepdim=True)
    return torch.cat([mean, std, corr, grain], dim=1)     # (B, 10)


COLOR_STAT_DIM = 10


def load_year_proxies(path, epoch=None, normalize="unit"):
    """Load the year centroids learned by the representation-learning proxy loss.

    `path` is either a `year_loss_epoch*.pth` file or the run folder containing
    them (then `epoch` selects one; default = latest available). The checkpoint
    holds `{"proxies": (n_years, proxy_dim)}`.

    `normalize`:
      * "unit" — L2-normalize each centroid (the proxy loss is cosine-based, so
                 only direction is meaningful; keeps the KL prior well-behaved).
      * "std"  — standardize to ~unit global std (matches the random-init scale).
      * "none" — use the raw vectors.
    """
    import glob
    import os

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "year_loss_epoch*.pth"))
        if not files:
            raise FileNotFoundError(f"no year_loss_epoch*.pth in {path}")
        if epoch is not None:
            path = os.path.join(path, f"year_loss_epoch{epoch}.pth")
        else:
            path = max(files, key=lambda f: int(f.rsplit("epoch", 1)[1].split(".")[0]))
    proxies = torch.load(path, map_location="cpu")["proxies"].float()
    if normalize == "unit":
        proxies = F.normalize(proxies, dim=1)
    elif normalize == "std":
        proxies = proxies / proxies.std().clamp_min(1e-6)
    return proxies


class ColorYearClassifier(nn.Module):
    """Predicts the year bucket from the global colour signature alone.

    Because its input is `color_moments` (and nothing else), the only way to
    satisfy it is to get the palette / contrast / colour-balance / grain right.
    Trained to classify real images by their true year; the generator is then
    pushed to make a translated image's colour signature read as the *target*
    year — a colorimetric analogue of the StarGAN domain classifier that the
    structure-biased PatchGAN classifier cannot provide."""

    def __init__(self, n_years, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COLOR_STAT_DIM, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, n_years),
        )

    def forward(self, x):
        return self.net(color_moments(x))


# ──────────────────────────────────────────────────────────────────────────────
# Vanilla CycleGAN blocks  (use for a strict 2-domain X<->Y model)
# ──────────────────────────────────────────────────────────────────────────────

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), _IN(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), _IN(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Unconditional CycleGAN ResNet generator (Johnson et al.). Fully
    convolutional: works at any spatial size divisible by 4."""

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), _IN(ngf), nn.ReLU(True)]
        for i in range(2):                                   # downsample x4
            m = 2 ** i
            layers += [nn.Conv2d(ngf * m, ngf * m * 2, 3, 2, 1), _IN(ngf * m * 2), nn.ReLU(True)]
        for _ in range(n_blocks):                            # transform
            layers += [ResnetBlock(ngf * 4)]
        for i in range(2):                                   # upsample x4
            m = 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, 2, 1, output_padding=1),
                       _IN(ngf * m // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# Year-conditional (AdaIN-modulated) generator
# ──────────────────────────────────────────────────────────────────────────────

class AdaIN(nn.Module):
    """Adaptive Instance Norm: normalize per channel, then scale/shift with
    (gamma, beta) predicted from the style code w. Init near identity."""

    def __init__(self, num_features, style_dim):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, w):
        gamma, beta = self.fc(w).chunk(2, dim=1)             # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * F.instance_norm(x) + beta


class AdaINResnetBlock(nn.Module):
    def __init__(self, dim, style_dim):
        super().__init__()
        self.pad1, self.conv1, self.norm1 = nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), AdaIN(dim, style_dim)
        self.pad2, self.conv2, self.norm2 = nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), AdaIN(dim, style_dim)
        self.act = nn.ReLU(True)

    def forward(self, x, w):
        h = self.act(self.norm1(self.conv1(self.pad1(x)), w))
        h = self.norm2(self.conv2(self.pad2(h)), w)
        return x + h


class ColorGrainHead(nn.Module):
    """Year-conditioned *global* colour/tone transform + film grain, applied to
    the generator's RGB output.

    The generator's AdaIN modulation lives deep in the bottleneck and is then
    re-normalized by the unconditional InstanceNorm layers of the upsampling
    tail, so the style code has no clean way to set a *global* colour cast or
    contrast. This head is the missing lever, acting directly on the output RGB:

        gamma  (3) — per-channel tone curve   -> contrast / fade / sepia roll-off
        matrix (9) — 3x3 colour mixing        -> illuminant / white-balance / palette
        bias   (3) — per-channel offset       -> colour cast
        grain  (1) — additive luminance noise -> film grain for older buckets

    `reset_identity()` zeroes the predictor so the head starts as a no-op (plain
    generator output) and learns the colorimetric transform from there. Call it
    *after* any global weight init (e.g. CycleGAN's normal(0, 0.02))."""

    def __init__(self, style_dim, max_grain=0.1):
        super().__init__()
        self.fc = nn.Linear(style_dim, 9 + 3 + 3 + 1)
        self.max_grain = max_grain
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.fc.bias.data[-1] = -6.0          # softplus(-6) ~ 0 -> no grain at init

    def forward(self, img, w):
        p = self.fc(w)
        b, c, h, wd = img.shape
        mat = p[:, :9].view(b, 3, 3) + torch.eye(3, device=img.device)   # ~identity
        bias = p[:, 9:12].view(b, 3, 1, 1)
        gamma = p[:, 12:15].exp().view(b, 3, 1, 1)                        # ~1 at init
        grain = F.softplus(p[:, 15:16]).view(b, 1, 1, 1) * self.max_grain

        x = ((img + 1) * 0.5).clamp(1e-4, 1.0)                            # [-1,1] -> [0,1]
        x = x.pow(gamma)                                                  # tone curve
        x = torch.einsum("boc,bchw->bohw", mat, x) + bias                # colour matrix + cast
        x = x + grain * torch.randn(b, 1, h, wd, device=img.device)      # film grain
        return (x.clamp(0.0, 1.0) * 2 - 1)                               # back to [-1,1]


class StyleResnetGenerator(nn.Module):
    """CycleGAN ResNet generator whose residual blocks are modulated by a year
    style code w (AdaIN). One generator maps an image to ANY target year.

    With `color_head=True` an explicit year-conditioned colour/tone/grain
    transform is applied to the output (see `ColorGrainHead`); with the default
    `color_head=False` the generator is exactly the original."""

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9, style_dim=64, color_head=False):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), _IN(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1), _IN(ngf * 2), nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), _IN(ngf * 4), nn.ReLU(True),
        )
        self.blocks = nn.ModuleList([AdaINResnetBlock(ngf * 4, style_dim) for _ in range(n_blocks)])
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1), _IN(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1), _IN(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh(),
        )
        self.color = ColorGrainHead(style_dim) if color_head else None

    def forward(self, x, w):
        h = self.head(x)
        for blk in self.blocks:
            h = blk(h, w)
        out = self.tail(h)
        return self.color(out, w) if self.color is not None else out


class VariationalYearStyle(nn.Module):
    """Per-year Gaussian latent q(z|y) = N(mu_y, sigma_y^2) -> style code w.

    The KL term keeps each year's latent compact and the overall space
    continuous, so years are interpolable. Year separation is preserved by the
    adversarial year-classifier in the discriminator (it forces z to carry year
    identity), not by KL — so a small lambda_kl will not collapse the years.
    """

    def __init__(self, n_years, latent_dim=64, style_dim=64, init_mu=None,
                 style_mode="variational"):
        super().__init__()
        # style_mode selects the year-conditioning mechanism so the SOTA baselines
        # share this exact backbone and isolate only the conditioning:
        #   "variational"   : per-year Gaussian q(z|y)=N(mu_y,sigma_y^2)   (ours)
        #   "deterministic" : per-year learned point -> AdaIN              (StarGAN-style)
        #   "freelatent"    : global noise z~N(0,I) + per-year shift       (StarGAN-v2-style)
        self.style_mode = style_mode
        self.latent_dim = latent_dim
        self.mu = nn.Embedding(n_years, latent_dim)
        self.logvar = nn.Embedding(n_years, latent_dim)
        if init_mu is not None:
            # Seed the year means with the centroids learned by the
            # representation-learning proxy loss, instead of random init, so the
            # generator starts from the *true* year geometry (ordering/spacing).
            assert init_mu.shape == (n_years, latent_dim), \
                f"init_mu {tuple(init_mu.shape)} != ({n_years}, {latent_dim})"
            self.mu.weight.data.copy_(init_mu)
        else:
            nn.init.normal_(self.mu.weight, 0.0, 0.1)
        nn.init.constant_(self.logvar.weight, -2.0)          # start with small variance
        if style_mode == "freelatent":
            self.year_embed = nn.Embedding(n_years, style_dim)   # domain shift
            nn.init.normal_(self.year_embed.weight, 0.0, 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, style_dim), nn.ReLU(True),
            nn.Linear(style_dim, style_dim), nn.ReLU(True),
        )

    def sample_latent(self, years, stochastic=True):
        mu, logvar = self.mu(years), self.logvar(years)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if stochastic else mu
        return z, mu, logvar

    @staticmethod
    def kl(mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()

    def w_from_latent(self, z):
        return self.mlp(z)

    def forward(self, years, stochastic=True):
        if self.style_mode == "deterministic":           # StarGAN-style domain embedding
            z = self.mu(years)
            zero = torch.zeros_like(z)
            return self.mlp(z), zero, zero                # kl(0,0)=0 -> no KL
        if self.style_mode == "freelatent":              # StarGAN-v2-style mapping net
            z = (torch.randn(years.size(0), self.latent_dim, device=years.device)
                 if stochastic else
                 torch.zeros(years.size(0), self.latent_dim, device=years.device))
            w = self.mlp(z) + self.year_embed(years)
            zero = torch.zeros(years.size(0), self.latent_dim, device=years.device)
            return w, zero, zero
        z, mu, logvar = self.sample_latent(years, stochastic)   # variational (ours)
        return self.mlp(z), mu, logvar

    @torch.no_grad()
    def interpolate_w(self, year_a, year_b, t):
        """Style code for a year *between* buckets a and b (t in [0,1])."""
        z = (1 - t) * self.mu.weight[year_a] + t * self.mu.weight[year_b]
        return self.mlp(z.unsqueeze(0))


# ──────────────────────────────────────────────────────────────────────────────
# PatchGAN discriminator (+ optional year classifier)
# ──────────────────────────────────────────────────────────────────────────────

class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN. If n_years is given it also outputs year logits (StarGAN
    domain classifier) — the signal that instantiates the target year."""

    def __init__(self, in_ch=3, ndf=64, n_layers=3, n_years=None):
        super().__init__()
        seq = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1), _IN(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1), _IN(ndf * mult), nn.LeakyReLU(0.2, True)]
        self.features = nn.Sequential(*seq)
        self.patch = nn.Conv2d(ndf * mult, 1, 4, 1, 1)                       # realism patches
        self.cls = nn.Conv2d(ndf * mult, n_years, 1) if n_years else None    # year logits

    def forward(self, x):
        f = self.features(x)
        src = self.patch(f)
        year = self.cls(f).mean(dim=(2, 3)) if self.cls is not None else None
        return src, year


# ──────────────────────────────────────────────────────────────────────────────
# Losses / utilities
# ──────────────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """LSGAN by default (CycleGAN standard)."""

    def __init__(self, mode="lsgan"):
        super().__init__()
        self.loss = nn.MSELoss() if mode == "lsgan" else nn.BCEWithLogitsLoss()

    def __call__(self, pred, target_is_real):
        target = (torch.ones_like if target_is_real else torch.zeros_like)(pred)
        return self.loss(pred, target)


class ImagePool:
    """Buffer of previously generated images (Shrivastava et al.) to reduce
    oscillation when updating the discriminator."""

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                i = random.randint(0, self.pool_size - 1)
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator: year-conditional CycleGAN with variational year style
# ──────────────────────────────────────────────────────────────────────────────

class YearCycleGAN(nn.Module):
    """Single-generator, year-conditional CycleGAN (StarGAN-shaped) with a
    variational year-style latent.

    Call `d_losses` / `g_losses` per step (separate optimizers for
    `discriminator_parameters()` and `generator_parameters()`).
    """

    def __init__(self, n_years, ngf=64, ndf=64, n_blocks=9,
                 style_dim=64, latent_dim=64,
                 lambda_cyc=10.0, lambda_idt=5.0, lambda_cls=1.0, lambda_kl=0.01,
                 pool_size=50, color_head=False, color_classifier=False,
                 lambda_color=1.0, init_centroids=None, style_mode="variational",
                 oracle_cls=None):
        super().__init__()
        # ── modular colorimetric add-ons (both default off → original model) ──
        #   color_head       : year-conditioned colour/tone/grain lever on G's output
        #   color_classifier : year signal that sees only global colour statistics
        self.color_head = color_head
        self.color_classifier = color_classifier
        self.lambda_color = lambda_color

        self.G = StyleResnetGenerator(ngf=ngf, n_blocks=n_blocks, style_dim=style_dim,
                                      color_head=color_head)
        self.style = VariationalYearStyle(n_years, latent_dim, style_dim,
                                          init_mu=init_centroids, style_mode=style_mode)
        self.D = NLayerDiscriminator(ndf=ndf, n_years=n_years)
        self.color_cls = ColorYearClassifier(n_years) if color_classifier else None
        # warm-started oracle as the discriminator's year-classifier branch:
        # a pretrained date classifier that keeps training alongside D. When set,
        # it provides the year signal in place of D's own classification head
        # (D's real/fake adversarial head is unchanged).
        self.oracle_cls = oracle_cls

        self.gan, self.l1, self.ce = GANLoss("lsgan"), nn.L1Loss(), nn.CrossEntropyLoss()
        self.lambda_cyc, self.lambda_idt = lambda_cyc, lambda_idt
        self.lambda_cls, self.lambda_kl = lambda_cls, lambda_kl
        self.pool = ImagePool(pool_size)

        init_weights(self.G)
        init_weights(self.D)
        if self.color_cls is not None:
            init_weights(self.color_cls)
        if self.G.color is not None:
            self.G.color.reset_identity()    # undo normal(0,0.02) → start as a no-op

    # parameter groups for the two optimizers
    def generator_parameters(self):
        return list(self.G.parameters()) + list(self.style.parameters())

    def discriminator_parameters(self):
        # the colour year-classifier is trained on reals alongside D
        params = list(self.D.parameters())
        if self.color_cls is not None:
            params += list(self.color_cls.parameters())
        if self.oracle_cls is not None:                      # warm-started, keeps training
            params += list(self.oracle_cls.parameters())
        return params

    # core conditional mapping
    def translate(self, x, years, stochastic=True):
        w, mu, logvar = self.style(years, stochastic)
        return self.G(x, w), mu, logvar

    @torch.no_grad()
    def translate_to_year(self, x, year_idx, stochastic=False):
        """Inference helper: send a batch to a single target year (defaults to
        the deterministic mean style)."""
        years = torch.full((x.size(0),), year_idx, dtype=torch.long, device=x.device)
        return self.translate(x, years, stochastic)[0]

    # ── discriminator step ────────────────────────────────────────────────────
    def d_losses(self, x_real, y_origin, y_target):
        with torch.no_grad():
            x_fake, _, _ = self.translate(x_real, y_target)
        x_fake = self.pool.query(x_fake)

        src_real, cls_real = self.D(x_real)
        src_fake, _ = self.D(x_fake)

        adv = self.gan(src_real, True) + self.gan(src_fake, False)
        # year signal: the warm-started oracle if attached, else D's own head
        cls_logits = self.oracle_cls(x_real) if self.oracle_cls is not None else cls_real
        cls = self.ce(cls_logits, y_origin)                  # classify REAL images by true year
        total = adv + self.lambda_cls * cls
        logs = {"d_adv": adv.item(), "d_cls": cls.item()}

        if self.color_cls is not None:                       # colour classifier on reals
            ccls = self.ce(self.color_cls(x_real), y_origin)
            total = total + self.lambda_color * ccls
            logs["d_ccls"] = ccls.item()
        return total, logs

    # ── generator step ────────────────────────────────────────────────────────
    def g_losses(self, x_real, y_origin, y_target):
        x_fake, mu_t, logvar_t = self.translate(x_real, y_target)   # x -> target year
        src_fake, cls_fake = self.D(x_fake)

        adv = self.gan(src_fake, True)
        cls_logits = self.oracle_cls(x_fake) if self.oracle_cls is not None else cls_fake
        cls = self.ce(cls_logits, y_target)                  # must land in the target year

        x_rec, _, _ = self.translate(x_fake, y_origin)       # cycle back
        cyc = self.l1(x_rec, x_real)

        x_idt, mu_o, logvar_o = self.translate(x_real, y_origin)   # identity
        idt = self.l1(x_idt, x_real)

        kl = self.style.kl(mu_t, logvar_t) + self.style.kl(mu_o, logvar_o)

        total = (adv
                 + self.lambda_cls * cls
                 + self.lambda_cyc * cyc
                 + self.lambda_idt * idt
                 + self.lambda_kl * kl)
        logs = {"g_adv": adv.item(), "g_cls": cls.item(), "cyc": cyc.item(),
                "idt": idt.item(), "kl": kl.item()}

        if self.color_cls is not None:        # push fake's colour signature to target year
            ccls = self.ce(self.color_cls(x_fake), y_target)
            total = total + self.lambda_color * ccls
            logs["g_ccls"] = ccls.item()
        return total, logs


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_YEARS, B, S = 14, 2, 64

    model = YearCycleGAN(n_years=N_YEARS, ngf=32, n_blocks=2, pool_size=8).to(device)
    x = torch.randn(B, 3, S, S, device=device)
    y_o = torch.randint(0, N_YEARS, (B,), device=device)
    y_t = torch.randint(0, N_YEARS, (B,), device=device)

    n_g = sum(p.numel() for p in model.generator_parameters())
    n_d = sum(p.numel() for p in model.discriminator_parameters())
    print(f"Generator+style params: {n_g/1e6:.2f}M | Discriminator params: {n_d/1e6:.2f}M")

    d_loss, d_logs = model.d_losses(x, y_o, y_t)
    d_loss.backward()
    print("D step OK :", {k: round(v, 4) for k, v in d_logs.items()})

    g_loss, g_logs = model.g_losses(x, y_o, y_t)
    g_loss.backward()
    print("G step OK :", {k: round(v, 4) for k, v in g_logs.items()})

    out = model.translate_to_year(x, year_idx=7)
    print("translate_to_year(7):", tuple(out.shape), f"range[{out.min():.2f},{out.max():.2f}]")

    w_mid = model.style.interpolate_w(0, 13, t=0.5)          # halfway between first & last bucket
    print("interpolated style code:", tuple(w_mid.shape))
    print("ALL OK")
