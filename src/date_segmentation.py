"""
detect_segment_year_hf_sam2.py
--------------------------------

Upload an image via browser →
  1. OWLv2 detects objects from --objects list  (boxes only – used for crop inference)
  2. SAM2 segments each detected box into a precise pixel mask  (used for visualisation only)
  3. CIR model infers category + year via PROXY CLOSENESS (same logic as qualitative.py)
       • cat_idx  = argmin_c  dist(emb,        cat_proxy[c])
       • year_idx = argmin_y  dist(emb − cat_proxy[cat_idx],  year_proxy[y])
  4. Results returned:
       • Left  – masks coloured by inferred CATEGORY  (+ legend)
       • Right – masks coloured by inferred YEAR       (+ legend)
       • Below – interactive canvas: hover a region to see category + year tooltip

Inference boxes:
  The OWLv2 bounding box is cropped and fed to the CNN.
  SAM2 is only used to paint a tight mask in the overlay – it does NOT affect inference.

Dependencies:
    pip install -U transformers accelerate flask pillow matplotlib torch torchvision pytorch-metric-learning

Recommended SAM2 checkpoint:
    facebook/sam2.1-hiera-small
"""

import argparse
import base64
import io
import os
import re
import time
import json
from annoy import AnnoyIndex
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, render_template_string, request
from pytorch_metric_learning import losses
from torchvision import transforms as T

from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    Sam2Model,
    Sam2Processor,
)

# ── local imports ─────────────────────────────────────────────────────────────
from core_datautils import df as data_complete
from models import ConditionedToYear, SpecialistModel
from train_experts_dataloader import SpecialistDataloaderWithClass

# ── constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM  = 1024
OWL_THRESHOLD  = 0.20
OVERLAY_ALPHA  = 0.60
OUTLINE_WIDTH  = 2
SAM2_MODEL_ID  = "facebook/sam2.1-hiera-small"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Standard ImageNet pre-processing
_PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

IMAGENET_TRANSFORMS_VAL = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── colour palettes ───────────────────────────────────────────────────────────
_CAT_PALETTE = [
    (255,  80,  80), ( 80, 160, 255), ( 80, 220, 120),
    (255, 190,  50), (200, 100, 255), (255, 130,  50),
    ( 50, 220, 210), (255,  80, 180), (160, 255,  80),
    (100, 100, 255), (255, 220, 100), ( 80, 255, 200),
]

def _year_to_rgb(year_idx: int, n_years: int) -> tuple:
    t   = year_idx / max(n_years - 1, 1)
    rgb = cm.plasma(t)[:3]
    return tuple(int(c * 255) for c in rgb)



def build_annoy_index(
    objects: list,
    ckpt_folder: str,
    epoch: int,
    device,
    batch_size: int = 32,
    num_workers: int = 4,
):
    annoy_path = os.path.join(ckpt_folder, "qualitative.ann")
    meta_path  = os.path.join(ckpt_folder, "qualitative_meta.json")

    if os.path.exists(annoy_path) and os.path.exists(meta_path):
        print("✔  Annoy index already exists – loading.")
        ann = AnnoyIndex(EMBEDDING_DIM, "euclidean")
        ann.load(annoy_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return  True, (ann, meta)
    return False, (None, None)

def _run_ann(ann, meta, query_vec: torch.Tensor, k = 5):
    qv_np = F.normalize(query_vec, p=2, dim=0).numpy().tolist()
    indices, distances = ann.get_nns_by_vector(qv_np, k, include_distances=True)
    results = []
    for idx, dist in zip(indices, distances):
        m = meta.get(str(idx), {})
        results.append({
            "b64":      m.get("b64", ""),
            "category": m.get("category", "?"),
            "year_idx": m.get("year_idx", -1),
            "dist":     float(dist),
        })
    return results, F.normalize(query_vec, p=2, dim=0)

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def pil_from_b64(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str.split(",", 1)[-1])
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def latest_epoch(ckpt_folder: str) -> int:
    epochs = [
        int(m.group(1))
        for f in os.listdir(ckpt_folder)
        if (m := re.match(r"model_epoch(\d+)\.pth", f))
    ]
    if not epochs:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_folder}")
    return max(epochs)


# ─────────────────────────────────────────────────────────────────────────────
# CIR model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cir_model(ckpt_folder: str, objects: list, epoch: int, device):
    dataset0 = SpecialistDataloaderWithClass(
        data_complete, objects[0], transforms=IMAGENET_TRANSFORMS_VAL,
    )
    avlabels = dataset0.available_labels
    n_years  = len(avlabels)

    date_sensitive = SpecialistModel(n_years).to(device)
    model          = ConditionedToYear(date_sensitive, output_dim=n_years).to(device)
    model.load_state_dict(
        torch.load(f"{ckpt_folder}/model_epoch{epoch}.pth", map_location=device)
    )
    model.eval()

    contrastive_loss_fn = losses.ProxyNCALoss(len(objects), EMBEDDING_DIM, softmax_scale=1)
    contrastive_loss_fn.load_state_dict(
        torch.load(f"{ckpt_folder}/contrastive_loss_epoch{epoch}.pth", map_location=device)
    )

    year_loss_fn = losses.ProxyNCALoss(n_years, EMBEDDING_DIM, softmax_scale=1)
    year_loss_fn.load_state_dict(
        torch.load(f"{ckpt_folder}/year_loss_epoch{epoch}.pth", map_location=device)
    )

    return model, contrastive_loss_fn, year_loss_fn, avlabels


# ─────────────────────────────────────────────────────────────────────────────
# OWLv2 – object detection
# ─────────────────────────────────────────────────────────────────────────────

def load_owlv2(device):
    print("Loading OWLv2 …")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model     = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    ).to(device)
    model.eval()
    print("  OWLv2 ready.")
    return processor, model


def run_owlv2(pil_img, objects, processor, owl_model, device, threshold=OWL_THRESHOLD):
    texts  = [[f"a photo of a {obj.lower()}" for obj in objects]]
    inputs = processor(text=texts, images=pil_img, return_tensors="pt").to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = owl_model(**inputs)
        else:
            outputs = owl_model(**inputs)

    target_sizes = torch.tensor([[pil_img.height, pil_img.width]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes,
    )[0]

    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label":    objects[label_id.item()],
            "label_id": label_id.item(),
            "score":    float(score.item()),
            "box_xyxy": [float(v) for v in box.tolist()],
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 – segmentation
# ─────────────────────────────────────────────────────────────────────────────

def load_sam2(device):
    print(f"Loading SAM2: {SAM2_MODEL_ID} …")
    processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    model     = Sam2Model.from_pretrained(
        SAM2_MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("  SAM2 ready.")
    return processor, model


def run_sam2_on_box(
    pil_img: Image.Image,
    box_xyxy: list,
    sam_processor,
    sam_model,
    device,
) -> np.ndarray:
    """
    Run SAM2 with a single bounding-box prompt.
    Returns a boolean mask of shape (H, W).
    Falls back to the filled rectangle if SAM2 raises an exception.
    """
    try:
        input_boxes = [[box_xyxy]]  # 3 levels: [image, box_list, coords]
        inputs = sam_processor(
            images=pil_img,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = sam_model(**inputs)
            else:
                outputs = sam_model(**inputs)

        # Skip post_process_masks; upsample raw logits manually
        pred = outputs.pred_masks  # (1, 1, n_candidates, H', W')
        pred = pred[0, 0]          # (n_candidates, H', W')

        best = int(torch.stack([m.sigmoid().sum() for m in pred]).argmax().item())
        logits = pred[best].unsqueeze(0).unsqueeze(0).float()  # (1,1,H',W')

        logits_up = torch.nn.functional.interpolate(
            logits,
            size=(pil_img.height, pil_img.width),
            mode="bilinear",
            align_corners=False,
        )
        mask = (logits_up[0, 0].sigmoid() > 0.5).cpu().numpy()
        return mask.astype(bool)

    except Exception as exc:
        print(f"  [SAM2 fallback – box mask] {exc}")
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        mask = np.zeros((pil_img.height, pil_img.width), dtype=bool)
        mask[max(y1,0):min(y2,pil_img.height), max(x1,0):min(x2,pil_img.width)] = True
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# CIR inference
# ─────────────────────────────────────────────────────────────────────────────

def embed_crop(crop_pil: Image.Image, model, device) -> torch.Tensor:
    tensor = _PREPROCESS(crop_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb, _ = model(tensor, None)
    return emb.squeeze(0).cpu()


def infer_proxy(
    emb: torch.Tensor,
    cat_proxies: torch.Tensor,
    year_proxies: torch.Tensor,
) -> tuple[int, int]:
    dists_cat = torch.cdist(emb.unsqueeze(0), cat_proxies).squeeze(0)
    cat_idx   = int(dists_cat.argmin().item())

    residual   = emb - cat_proxies[cat_idx]
    dists_year = torch.cdist(residual.unsqueeze(0), year_proxies).squeeze(0)
    year_idx   = int(dists_year.argmin().item())

    return cat_idx, year_idx


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation – clean masks only, no boxes or text
# ─────────────────────────────────────────────────────────────────────────────

def build_overlay(
    base_pil: Image.Image,
    detections: list,
    mode: str,
    n_years: int,
    objects: list,
    alpha: float = OVERLAY_ALPHA,
) -> Image.Image:
    base   = np.array(base_pil.convert("RGB"), dtype=np.float32)
    result = base.copy()

    for det in detections:
        mask    = det["mask"]
        cat_idx = det["cat_idx"]
        yr_idx  = det["year_idx"]

        colour = np.array(
            _CAT_PALETTE[cat_idx % len(_CAT_PALETTE)] if mode == "category"
            else _year_to_rgb(yr_idx, n_years),
            dtype=np.float32,
        )
        result[mask] = (1.0 - alpha) * result[mask] + alpha * colour

    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Mask encoding for interactive canvas
# ─────────────────────────────────────────────────────────────────────────────

def encode_mask_rle(mask: np.ndarray) -> dict:
    """
    Encode a boolean (H, W) mask as a run-length encoded dict for JSON transfer.
    Returns { "shape": [H, W], "runs": [start, length, start, length, ...] }
    where indices are into the flattened row-major array.
    """
    flat = mask.flatten().astype(np.uint8)
    runs = []
    i = 0
    while i < len(flat):
        if flat[i] == 1:
            start = i
            while i < len(flat) and flat[i] == 1:
                i += 1
            runs.extend([int(start), int(i - start)])
        else:
            i += 1
    return {"shape": list(mask.shape), "runs": runs}


# ─────────────────────────────────────────────────────────────────────────────
# HTML frontend
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIR · Segment &amp; Date</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:   #0a0a0f;
    --card: #13131a;
    --line: #1e1e2e;
    --acc:  #c8f542;
    --acc2: #42f5c8;
    --acc3: #f542c8;
    --txt:  #e2e2f0;
    --dim:  #6b6b8a;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--txt);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header {
    width: 100%;
    padding: 2rem 3rem 1.2rem;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
  }
  header h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.7rem; color: var(--acc); letter-spacing: -.02em; }
  header .sub { color: var(--dim); font-size: .78rem; }

  .controls {
    width: 100%; max-width: 900px;
    margin: 2.5rem auto 0;
    padding: 0 1.5rem;
    display: flex; gap: .75rem; flex-wrap: wrap; align-items: flex-end;
  }
  .upload-zone {
    flex: 1 1 240px; min-height: 140px;
    background: var(--card);
    border: 1px dashed var(--line);
    border-radius: 10px;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .5rem;
    cursor: pointer; transition: border-color .15s; position: relative;
  }
  .upload-zone:hover { border-color: var(--acc); }
  .upload-zone.loaded { border-style: solid; border-color: var(--acc2); }
  .upload-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-zone .preview { max-height: 110px; max-width: 100%; border-radius: 4px; object-fit: contain; pointer-events: none; display: none; }
  .upload-zone .uz-hint { font-size: .7rem; color: var(--dim); pointer-events: none; }
  .upload-zone .uz-tag  { font-size: .65rem; color: var(--acc2); pointer-events: none; }

  button {
    background: var(--acc); color: #0a0a0f;
    border: none; cursor: pointer;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .85rem;
    padding: .6rem 1.4rem; border-radius: 6px;
    transition: opacity .15s, transform .1s;
    white-space: nowrap; align-self: flex-end;
  }
  button:hover { opacity: .85; transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled { opacity: .35; cursor: not-allowed; transform: none; }

  #status {
    width: 100%; max-width: 900px;
    margin: 1rem auto 0; padding: 0 1.5rem;
    font-size: .74rem; color: var(--dim); min-height: 1.2em;
  }
  #status.err { color: #ff6b6b; }

  /* ── static overlays ── */
  .results {
    width: 100%; max-width: 1100px;
    margin: 2rem auto 0; padding: 0 1.5rem;
    display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;
  }
  .panel { display: flex; flex-direction: column; gap: .6rem; }
  .panel-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .8rem;
    color: var(--acc3); letter-spacing: .1em; text-transform: uppercase;
  }
  .panel img { width: 100%; border-radius: 8px; border: 1px solid var(--line); display: block; }

  /* ── legend ── */
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem .45rem;
    margin-top: .25rem;
  }
  .legend-chip {
    display: flex;
    align-items: center;
    gap: .3rem;
    font-size: .63rem;
    color: var(--txt);
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: .18rem .4rem;
  }
  .legend-swatch {
    width: 9px; height: 9px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  /* ── interactive canvas section ── */
  .interactive-section {
    width: 100%; max-width: 1100px;
    margin: 2.5rem auto 3rem; padding: 0 1.5rem;
  }
  .interactive-header {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .8rem;
    color: var(--acc2); letter-spacing: .1em; text-transform: uppercase;
    margin-bottom: .75rem;
    display: flex; align-items: center; gap: .6rem;
  }
  .interactive-header .badge {
    font-family: 'DM Mono', monospace; font-weight: 400; font-size: .62rem;
    color: var(--dim); background: var(--card); border: 1px solid var(--line);
    border-radius: 4px; padding: .15rem .4rem; letter-spacing: 0;
    text-transform: none;
  }
  .canvas-wrap {
    position: relative;
    display: inline-block;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--line);
    cursor: crosshair;
  }
  .canvas-wrap canvas {
    display: block;
    width: 100%;
    height: auto;
  }
  /* tooltip */
  #tooltip {
    position: fixed;
    pointer-events: none;
    display: none;
    background: #0a0a0fee;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: .45rem .7rem;
    font-size: .72rem;
    color: var(--txt);
    z-index: 9999;
    backdrop-filter: blur(4px);
    min-width: 120px;
  }
  #tooltip .tt-label { font-family: 'Syne', sans-serif; font-weight: 700; font-size: .82rem; margin-bottom: .25rem; }
  #tooltip .tt-row   { display: flex; align-items: center; gap: .4rem; color: var(--dim); }
  #tooltip .tt-val   { color: var(--txt); }
  #tooltip .tt-swatch { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }

  /* mode toggle */
  .mode-toggle {
    display: flex; gap: .4rem; margin-bottom: .75rem;
  }
  .mode-btn {
    background: var(--card); color: var(--dim);
    border: 1px solid var(--line); cursor: pointer;
    font-family: 'DM Mono', monospace; font-weight: 400; font-size: .72rem;
    padding: .3rem .75rem; border-radius: 5px;
    transition: all .15s;
  }
  .mode-btn.active {
    background: var(--acc2); color: #0a0a0f;
    border-color: var(--acc2); font-weight: 500;
  }

  .spinner {
    display: inline-block; width: 15px; height: 15px;
    border: 2px solid #0a0a0f55; border-top-color: #0a0a0f;
    border-radius: 50%; animation: spin .6s linear infinite;
    margin-left: .4rem; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .detections {
    width: 100%; max-width: 900px;
    margin: .5rem auto 0; padding: 0 1.5rem;
    display: flex; flex-wrap: wrap; gap: .4rem;
  }
  .det-chip {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 5px; padding: .25rem .6rem;
    font-size: .68rem; color: var(--txt);
  }
  .det-chip .cat { color: var(--acc); }
  .det-chip .yr  { color: var(--acc2); }
</style>
</head>
<body>

<header>
  <h1>CIR · Segment &amp; Date</h1>
  <span class="sub">OWLv2 detection → SAM2 mask → proxy-based category &amp; year inference</span>
</header>

<div class="controls">
  <div class="upload-zone" id="dropzone">
    <input type="file" accept="image/*" id="fileInput">
    <img class="preview" id="preview" alt="preview">
    <span class="uz-hint" id="hint">Drop image or click to upload</span>
    <span class="uz-tag">INPUT IMAGE</span>
  </div>
  <button id="runBtn" onclick="run()" disabled>
    ANALYSE <span class="spinner" id="spin" style="display:none"></span>
  </button>
</div>

<div id="status">Upload an image to begin.</div>

<div class="detections" id="chips"></div>

<!-- ── static overlays ── -->
<div class="results" id="results" style="display:none">
  <div class="panel">
    <span class="panel-title">Category overlay</span>
    <img id="catImg" alt="category overlay">
    <div class="legend" id="legendCat"></div>
  </div>
  <div class="panel">
    <span class="panel-title">Year overlay</span>
    <img id="yearImg" alt="year overlay">
    <div class="legend" id="legendYear"></div>
  </div>
</div>

<!-- ── interactive hover canvas ── -->
<div class="interactive-section" id="interactiveSection" style="display:none">
  <div class="interactive-header">
    Interactive explorer
    <span class="badge">hover to inspect regions</span>
  </div>
  <div class="mode-toggle">
    <button class="mode-btn active" id="btnCat"  onclick="setMode('category')">Category colours</button>
    <button class="mode-btn"        id="btnYear" onclick="setMode('year')">Year colours</button>
  </div>
  <div class="canvas-wrap" id="canvasWrap">
    <canvas id="hoverCanvas"></canvas>
  </div>
</div>

<!-- tooltip -->
<div id="tooltip">
  <div class="tt-label" id="ttLabel"></div>
  <div class="tt-row">
    <span class="tt-swatch" id="ttCatSwatch"></span>
    <span style="color:var(--dim)">cat&nbsp;</span>
    <span class="tt-val" id="ttCat"></span>
  </div>
  <div class="tt-row" style="margin-top:.15rem">
    <span class="tt-swatch" id="ttYearSwatch"></span>
    <span style="color:var(--dim)">year&nbsp;</span>
    <span class="tt-val" id="ttYear"></span>
  </div>
</div>

<script>
/* ── state ─────────────────────────────────────────────────────────── */
let imageB64    = null;
let analysisData = null;   // full JSON response from /analyse
let canvasMode  = 'category';

/* decoded masks: array of { mask: Uint8Array, shape:[H,W], cat_idx, year_idx, label, cat_color, year_color } */
let decodedMasks = [];

/* per-pixel lookup: Int32Array of length H*W, value = detection index (-1 = background) */
let pixelIndex  = null;

/* ── file handling ──────────────────────────────────────────────────── */
const fileInput = document.getElementById('fileInput');
const dropzone  = document.getElementById('dropzone');
const preview   = document.getElementById('preview');
const hint      = document.getElementById('hint');
const runBtn    = document.getElementById('runBtn');

fileInput.addEventListener('change', e => loadFile(e.target.files[0]));
dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.style.borderColor = 'var(--acc)'; });
dropzone.addEventListener('dragleave', ()  => { dropzone.style.borderColor = ''; });
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.style.borderColor = '';
  loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    imageB64 = ev.target.result;
    preview.src = imageB64;
    preview.style.display = 'block';
    hint.style.display = 'none';
    dropzone.classList.add('loaded');
    runBtn.disabled = false;
    setStatus('Ready – click ANALYSE.');
  };
  reader.readAsDataURL(file);
}

/* ── analyse ────────────────────────────────────────────────────────── */
async function run() {
  if (!imageB64) return;
  setLoading(true);
  document.getElementById('results').style.display = 'none';
  document.getElementById('interactiveSection').style.display = 'none';
  document.getElementById('chips').innerHTML = '';
  setStatus('Running OWLv2 + SAM2 + CIR …');

  try {
    const resp = await fetch('/analyse', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image_b64: imageB64 }),
    });
    const data = await resp.json();
    if (data.error) { setStatus('❌ ' + data.error, true); return; }

    analysisData = data;

    /* static overlays */
    document.getElementById('catImg').src  = 'data:image/png;base64,' + data.category_img_b64;
    document.getElementById('yearImg').src = 'data:image/png;base64,' + data.year_img_b64;
    document.getElementById('results').style.display = 'grid';

    /* legends */
    buildLegend('legendCat',  data.legend_cat);
    buildLegend('legendYear', data.legend_year);

    /* detection chips */
    const chips = document.getElementById('chips');
    (data.detections || []).forEach(d => {
      const chip = document.createElement('span');
      chip.className = 'det-chip';
      chip.innerHTML =
        `<span class="cat">${d.label}</span> ` +
        `score <span class="yr">${d.score.toFixed(2)}</span> · ` +
        `yr <span class="yr">${d.year_idx}</span>`;
      chips.appendChild(chip);
    });

    /* interactive canvas */
    initInteractive(data);
    document.getElementById('interactiveSection').style.display = 'block';

    setStatus(`Done in ${data.elapsed_ms} ms · ${data.detections.length} object(s) detected.`);
  } catch (err) {
    setStatus('❌ ' + err, true);
  } finally {
    setLoading(false);
  }
}

/* ── legend builder ─────────────────────────────────────────────────── */
function buildLegend(containerId, items) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  items.forEach(({ label, color }) => {
    const chip = document.createElement('span');
    chip.className = 'legend-chip';
    chip.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${label}`;
    el.appendChild(chip);
  });
}

/* ── interactive canvas ─────────────────────────────────────────────── */
function decodeRle(rle) {
  const [H, W] = rle.shape;
  const flat = new Uint8Array(H * W);
  const runs = rle.runs;
  for (let i = 0; i < runs.length; i += 2) {
    const start = runs[i], len = runs[i + 1];
    flat.fill(1, start, start + len);
  }
  return { mask: flat, H, W };
}

function parseColor(cssRgb) {
  /* "rgb(r,g,b)" → [r,g,b] */
  const m = cssRgb.match(/(\d+),\s*(\d+),\s*(\d+)/);
  return m ? [+m[1], +m[2], +m[3]] : [128, 128, 128];
}

function initInteractive(data) {
  const canvas = document.getElementById('hoverCanvas');
  decodedMasks = [];

  /* decode all RLE masks */
  data.mask_data.forEach((md, i) => {
    const { mask, H, W } = decodeRle(md.rle);
    decodedMasks.push({
      mask,
      H, W,
      cat_idx:    md.cat_idx,
      year_idx:   md.year_idx,
      label:      md.label,
      cat_color:  parseColor(data.legend_cat[md.cat_idx].color),
      year_color: parseColor(data.legend_year[md.year_idx] ? data.legend_year[md.year_idx].color : 'rgb(128,128,128)'),
    });
  });

  if (!decodedMasks.length) return;

  const H = decodedMasks[0].H;
  const W = decodedMasks[0].W;
  canvas.width  = W;
  canvas.height = H;

  /* build per-pixel index: last mask wins (painter's algo) */
  pixelIndex = new Int32Array(H * W).fill(-1);
  decodedMasks.forEach((dm, i) => {
    for (let p = 0; p < dm.mask.length; p++) {
      if (dm.mask[p]) pixelIndex[p] = i;
    }
  });

  drawCanvas('category');

  /* mouse move handler */
  const wrap = document.getElementById('canvasWrap');
  wrap.addEventListener('mousemove', onCanvasMove);
  wrap.addEventListener('mouseleave', () => {
    document.getElementById('tooltip').style.display = 'none';
    drawCanvas(canvasMode);   // remove highlight
  });
}

function drawCanvas(mode, highlightIdx = -1) {
  const canvas = document.getElementById('hoverCanvas');
  const ctx    = canvas.getContext('2d');
  if (!decodedMasks.length) return;

  const H = canvas.height, W = canvas.width;
  const imgData = ctx.createImageData(W, H);
  const buf     = imgData.data;

  /* draw source image first (encoded as PNG in the response) */
  /* We'll composite masks over a grey base; the base image is re-drawn via offscreen */
  /* Use the already-displayed static overlay as reference? No – redraw from scratch.   */
  /* Strategy: draw base (src image pixels) + alpha-blend masks on top.                */
  /* The src image is available as imageB64 – decode via offscreen canvas.              */

  const src = window.__srcPixels;   // set once on first draw
  if (!src) {
    /* async: decode image then redraw */
    const img = new window.Image();
    img.onload = () => {
      const off = document.createElement('canvas');
      off.width = W; off.height = H;
      const octx = off.getContext('2d');
      octx.drawImage(img, 0, 0, W, H);
      window.__srcPixels = octx.getImageData(0, 0, W, H).data;
      drawCanvas(mode, highlightIdx);
    };
    img.src = imageB64;
    return;
  }

  const ALPHA = 0.55;
  const HI_ALPHA = 0.80;

  for (let p = 0; p < H * W; p++) {
    const b = p * 4;
    buf[b]   = src[b];
    buf[b+1] = src[b+1];
    buf[b+2] = src[b+2];
    buf[b+3] = 255;

    const di = pixelIndex[p];
    if (di < 0) continue;

    const dm = decodedMasks[di];
    const col = mode === 'category' ? dm.cat_color : dm.year_color;
    const a   = (di === highlightIdx) ? HI_ALPHA : ALPHA;

    buf[b]   = Math.round((1 - a) * buf[b]   + a * col[0]);
    buf[b+1] = Math.round((1 - a) * buf[b+1] + a * col[1]);
    buf[b+2] = Math.round((1 - a) * buf[b+2] + a * col[2]);
  }

  ctx.putImageData(imgData, 0, 0);

  /* outline highlight */
  if (highlightIdx >= 0) {
    const dm  = decodedMasks[highlightIdx];
    const col = mode === 'category' ? dm.cat_color : dm.year_color;
    /* draw 1px bright border around bounding box of the mask */
    let minX = canvas.width, maxX = 0, minY = canvas.height, maxY = 0;
    for (let p = 0; p < dm.mask.length; p++) {
      if (!dm.mask[p]) continue;
      const x = p % canvas.width, y = Math.floor(p / canvas.width);
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    ctx.strokeStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.lineWidth   = 2;
    ctx.shadowColor = '#000';
    ctx.shadowBlur  = 4;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
    ctx.shadowBlur  = 0;
  }
}

function onCanvasMove(e) {
  const canvas = document.getElementById('hoverCanvas');
  if (!pixelIndex || !canvas) return;

  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const cx     = Math.floor((e.clientX - rect.left)  * scaleX);
  const cy     = Math.floor((e.clientY - rect.top)   * scaleY);

  if (cx < 0 || cy < 0 || cx >= canvas.width || cy >= canvas.height) return;

  const p  = cy * canvas.width + cx;
  const di = pixelIndex[p];

  const tt = document.getElementById('tooltip');

  if (di < 0) {
    tt.style.display = 'none';
    drawCanvas(canvasMode);
    return;
  }

  const dm = decodedMasks[di];

  /* redraw with highlight */
  drawCanvas(canvasMode, di);

  /* position tooltip */
  tt.style.display = 'block';
  const ttW = tt.offsetWidth  || 140;
  const ttH = tt.offsetHeight || 70;
  let tx = e.clientX + 16;
  let ty = e.clientY - 8;
  if (tx + ttW > window.innerWidth  - 8) tx = e.clientX - ttW - 16;
  if (ty + ttH > window.innerHeight - 8) ty = e.clientY - ttH - 8;
  tt.style.left = tx + 'px';
  tt.style.top  = ty + 'px';

  const catCol  = `rgb(${dm.cat_color.join(',')})`;
  const yearCol = `rgb(${dm.year_color.join(',')})`;

  document.getElementById('ttLabel').textContent          = dm.label;
  document.getElementById('ttCat').textContent            = dm.label;
  document.getElementById('ttCatSwatch').style.background = catCol;
  document.getElementById('ttYear').textContent            = 'yr ' + dm.year_idx;
  document.getElementById('ttYearSwatch').style.background = yearCol;
}

function setMode(mode) {
  canvasMode = mode;
  document.getElementById('btnCat').classList.toggle('active',  mode === 'category');
  document.getElementById('btnYear').classList.toggle('active', mode === 'year');
  drawCanvas(mode);
}

/* ── utils ──────────────────────────────────────────────────────────── */
function setLoading(on) {
  document.getElementById('spin').style.display = on ? 'inline-block' : 'none';
  runBtn.disabled = on;
}
function setStatus(msg, isErr = false) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className   = isErr ? 'err' : '';
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Flask application
# ─────────────────────────────────────────────────────────────────────────────

def create_app(
    objects,
    model,
    cat_proxies,
    year_proxies,
    owl_processor, owl_model,
    sam_processor, sam_model,
    device,
):
    app     = Flask(__name__)
    n_years = year_proxies.shape[0]

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/analyse", methods=["POST"])
    def analyse():
        t0   = time.time()
        body = request.get_json()
        if not body or "image_b64" not in body:
            return jsonify({"error": "Missing image_b64"}), 400

        pil_img = pil_from_b64(body["image_b64"])

        # ── 1. OWLv2: detect objects ──────────────────────────────────────────
        detections = run_owlv2(pil_img, objects, owl_processor, owl_model, device)

        # ── 2. Per-detection: SAM2 mask + CIR proxy inference ─────────────────
        enriched = []
        for det in detections:
            box = det["box_xyxy"]

            mask = run_sam2_on_box(pil_img, box, sam_processor, sam_model, device)

            x1 = max(int(box[0]), 0)
            y1 = max(int(box[1]), 0)
            x2 = min(int(box[2]), pil_img.width)
            y2 = min(int(box[3]), pil_img.height)

            if (x2 - x1) < 8 or (y2 - y1) < 8:
                cat_idx, year_idx = 0, 0
            else:
                crop = pil_img.crop((x1, y1, x2, y2))
                emb  = embed_crop(crop, model, device)
                cat_idx, year_idx = infer_proxy(emb, cat_proxies, year_proxies)

            enriched.append({
                **det,
                "mask":     mask,
                "cat_idx":  cat_idx,
                "year_idx": year_idx,
            })

        # ── 3. Build overlays ─────────────────────────────────────────────────
        cat_overlay  = build_overlay(pil_img, enriched, "category", n_years, objects)
        year_overlay = build_overlay(pil_img, enriched, "year",     n_years, objects)

        # ── 4. Legend data ────────────────────────────────────────────────────
        legend_cat = [
            {
                "label": objects[i],
                "color": "rgb({},{},{})".format(*_CAT_PALETTE[i % len(_CAT_PALETTE)])
            }
            for i in range(len(objects))
        ]
        legend_year = [
            {
                "label": f"yr {i}",
                "color": "rgb({},{},{})".format(*_year_to_rgb(i, n_years))
            }
            for i in range(n_years)
        ]

        # ── 5. Mask data for interactive canvas ───────────────────────────────
        mask_data = [
            {
                "rle":      encode_mask_rle(d["mask"]),
                "cat_idx":  d["cat_idx"],
                "year_idx": d["year_idx"],
                "label":    d["label"],
            }
            for d in enriched
        ]

        # ── 6. Detection summary ──────────────────────────────────────────────
        det_summary = [
            {
                "label":    d["label"],
                "score":    round(d["score"], 3),
                "cat_idx":  d["cat_idx"],
                "year_idx": d["year_idx"],
            }
            for d in enriched
        ]

        return jsonify({
            "category_img_b64": pil_to_b64(cat_overlay),
            "year_img_b64":     pil_to_b64(year_overlay),
            "detections":       det_summary,
            "elapsed_ms":       int((time.time() - t0) * 1000),
            "legend_cat":       legend_cat,
            "legend_year":      legend_year,
            "mask_data":        mask_data,
        })

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CIR segment-and-date demo (SAM2 masks + proxy inference)"
    )
    p.add_argument("--objects",     nargs="+", required=True,
                   help="Object classes to detect, e.g. --objects Car Bus Truck")
    p.add_argument("--ckpt_folder", required=True,
                   help="Folder containing model_epochN.pth / *_loss_epochN.pth")
    p.add_argument("--epoch",       type=int, default=None,
                   help="Checkpoint epoch (default: latest in ckpt_folder)")
    p.add_argument("--port",        type=int, default=5001)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    epoch = args.epoch if args.epoch is not None else latest_epoch(args.ckpt_folder)
    print(f"Epoch  : {epoch}")

    print("Loading CIR model …")
    model, contrastive_loss_fn, year_loss_fn, avlabels = load_cir_model(
        args.ckpt_folder, args.objects, epoch, device
    )


    
    is_there_db, (ann, meta) = build_annoy_index(args.objects, args.ckpt_folder, epoch, device)

    cat_proxies  = contrastive_loss_fn.proxies.detach().cpu()
    year_proxies = year_loss_fn.proxies.detach().cpu()
    print(f"  Category proxies : {cat_proxies.shape}")
    print(f"  Year proxies     : {year_proxies.shape}")

    owl_processor, owl_model = load_owlv2(device)
    sam_processor, sam_model = load_sam2(device)

    app = create_app(
        objects       = args.objects,
        model         = model,
        cat_proxies   = cat_proxies,
        year_proxies  = year_proxies,
        owl_processor = owl_processor,
        owl_model     = owl_model,
        sam_processor = sam_processor,
        sam_model     = sam_model,
        device        = device,
    )

    print(f"\n🚀  http://0.0.0.0:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()