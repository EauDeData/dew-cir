"""
qualitative.py
--------------
Build / reuse an Annoy vector-DB from a CIR checkpoint, then serve a small
Flask app that queries it with  CATEGORY_proxy + YEAR_proxy  as the search
vector.

Endpoints
---------
  /                    – original proxy-based search
  /query_by_example/   – image upload → CNN embedding + year_proxy
  /query_two_images/   – image_A (category) + image_B (year reference) search

Usage:
    python qualitative.py \
        --objects Dog Cat \
        --ckpt_folder /data/.../cir_date/<run_id>/ \
        --epoch 10 \
        [--port 5000]

The script auto-detects the latest epoch if --epoch is not given.
"""

import argparse
import base64
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from annoy import AnnoyIndex
from flask import Flask, jsonify, render_template_string, request
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── local imports (same src/ package as main_cir.py) ────────────────────────
# I have also df_test
from core_datautils import df as data_complete
from models import ConditionedToYear, SpecialistModel
from train_experts_dataloader import SpecialistDataloaderWithClass
from torchvision import transforms

# Standard ImageNet normalization
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Validation / test transforms (deterministic)
IMAGENET_TRANSFORMS_VAL = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE,
])

# Training transforms (with augmentations)
IMAGENET_TRANSFORMS_TRAIN = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE,
])
# ── constants ────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1024
N_TREES = 50          # more → better recall, slower build
THUMB_SIZE = (64, 64)

# ── shared image pre-processing (same as IMAGENET_TRANSFORMS_VAL but for PIL) ──
import torchvision.transforms as T
_PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def img_tensor_to_b64(tensor: torch.Tensor) -> str:
    """Convert a CHW float tensor (ImageNet-normalised) to a 64×64 base-64 PNG."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1)
    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    pil = pil.resize(THUMB_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_from_b64(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str.split(",", 1)[-1])   # strip data-URI prefix
    return Image.open(io.BytesIO(data)).convert("RGB")


def embed_pil(pil_img: Image.Image, model, device, normalize = False) -> torch.Tensor:
    """
    Run a PIL image through the CNN and return a normalised 1024-d embedding.
    Passes condition=None (year unknown).
    """
    tensor = _PREPROCESS(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb_agnostic, _ = model(tensor, None)
    
    # For objects we dont normalize
    if normalize:
        return F.normalize(emb_agnostic.squeeze(0), p=2, dim=0).cpu()
    return emb_agnostic.squeeze(0).cpu()


def latest_epoch(ckpt_folder: str) -> int:
    import re
    epochs = [
        int(m.group(1))
        for f in os.listdir(ckpt_folder)
        if (m := re.match(r"model_epoch(\d+)\.pth", f))
    ]
    if not epochs:
        raise FileNotFoundError(f"No model checkpoints found in {ckpt_folder}")
    return max(epochs)


# ─────────────────────────────────────────────────────────────────────────────
# load model + losses from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_everything(ckpt_folder: str, objects: list, epoch: int, device):
    dataset0 = SpecialistDataloaderWithClass(
        data_complete, objects[0], transforms=IMAGENET_TRANSFORMS_VAL
    )
    avlabels = dataset0.available_labels
    n_years  = len(avlabels)

    date_sensitive = SpecialistModel(n_years).to(device)
    model = ConditionedToYear(date_sensitive, output_dim=n_years).to(device)
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
# build / load Annoy index
# ─────────────────────────────────────────────────────────────────────────────

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
        return ann, meta

    print("Building Annoy index from scratch…")
    model, contrastive_loss_fn, year_loss_fn, avlabels = load_everything(
        ckpt_folder, objects, epoch, device
    )

    obj2label = {obj: i for i, obj in enumerate(objects)}
    dataset = SpecialistDataloaderWithClass(
        data_complete, objects[0], transforms=IMAGENET_TRANSFORMS_VAL
    )
    dataset.available_labels = avlabels
    for obj in objects[1:]:
        extra = SpecialistDataloaderWithClass(
            data_complete, obj, transforms=IMAGENET_TRANSFORMS_VAL
        )
        dataset = dataset + extra
        dataset.available_labels = avlabels

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=False)

    ann  = AnnoyIndex(EMBEDDING_DIM, "euclidean")
    meta = {}
    global_idx = 0

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(loader, desc="Ingesting"):
            image_A     = image_A.to(device)
            condition_A = condition_A.to(device)

            emb_agnostic, _ = model(image_A, condition_A)
            emb_agnostic = F.normalize(emb_agnostic, p=2, dim=1)
            emb_np = emb_agnostic.cpu().numpy()

            for i in range(len(emb_np)):
                ann.add_item(global_idx, emb_np[i].tolist())
                b64 = img_tensor_to_b64(image_A[i])
                meta[str(global_idx)] = {
                    "b64":      b64,
                    "category": category[i] if isinstance(category[i], str) else category[i],
                    "year_idx": int(condition_B[i].item()),
                }
                global_idx += 1

            with open(meta_path, "w") as f:
                json.dump(meta, f)
            print(f"  saved meta ({global_idx} items so far)")
            if len(meta) > 10000:
                break

    ann.build(N_TREES)
    ann.save(annoy_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"✔  Annoy index built: {global_idx} items → {annoy_path}")
    return ann, meta


# ─────────────────────────────────────────────────────────────────────────────
# HTML templates
# ─────────────────────────────────────────────────────────────────────────────

# ── shared CSS / font variables injected into every page ────────────────────
_SHARED_HEAD = """
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
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
  header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    color: var(--acc);
    letter-spacing: -.02em;
  }
  header span.sub { color: var(--dim); font-size: .8rem; }
  nav {
    margin-left: auto;
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
  }
  nav a {
    color: var(--dim);
    font-size: .72rem;
    text-decoration: none;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: .3rem .65rem;
    transition: color .15s, border-color .15s;
  }
  nav a:hover, nav a.active { color: var(--acc); border-color: var(--acc); }

  .controls {
    width: 100%;
    max-width: 860px;
    margin: 2.5rem auto 0;
    padding: 0 1.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: .75rem;
    align-items: end;
  }
  label { display: flex; flex-direction: column; gap: .4rem; font-size: .72rem; color: var(--dim); flex: 1 1 140px; }
  select, input[type=number] {
    background: var(--card);
    border: 1px solid var(--line);
    color: var(--txt);
    font-family: 'DM Mono', monospace;
    font-size: .9rem;
    padding: .55rem .75rem;
    border-radius: 6px;
    outline: none;
    transition: border .15s;
    width: 100%;
  }
  select:focus, input:focus { border-color: var(--acc); }
  button {
    background: var(--acc);
    color: #0a0a0f;
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: .85rem;
    padding: .6rem 1.3rem;
    border-radius: 6px;
    cursor: pointer;
    transition: opacity .15s, transform .1s;
    white-space: nowrap;
    align-self: flex-end;
  }
  button:hover { opacity: .85; transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled { opacity: .4; cursor: not-allowed; transform: none; }

  .upload-zone {
    background: var(--card);
    border: 1px dashed var(--line);
    border-radius: 10px;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: .5rem;
    cursor: pointer;
    transition: border-color .15s;
    flex: 1 1 180px;
    min-height: 120px;
    justify-content: center;
    position: relative;
  }
  .upload-zone:hover { border-color: var(--acc); }
  .upload-zone.has-img { border-style: solid; border-color: var(--acc2); }
  .upload-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .upload-zone img {
    max-height: 90px;
    max-width: 100%;
    border-radius: 4px;
    object-fit: contain;
    pointer-events: none;
  }
  .upload-zone .uz-label { font-size: .7rem; color: var(--dim); pointer-events: none; }
  .upload-zone .uz-tag   { font-size: .65rem; color: var(--acc2); pointer-events: none; }

  #status {
    width: 100%;
    max-width: 860px;
    margin: 1rem auto 0;
    padding: 0 1.5rem;
    font-size: .75rem;
    color: var(--dim);
    min-height: 1.2em;
  }
  .query-vec {
    width: 100%;
    max-width: 860px;
    margin: .3rem auto 0;
    padding: 0 1.5rem;
    font-size: .72rem;
    color: var(--acc2);
  }
  #results {
    width: 100%;
    max-width: 980px;
    margin: 2rem auto 3rem;
    padding: 0 1.5rem;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 1rem;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color .15s, transform .15s;
  }
  .card:hover { border-color: var(--acc); transform: translateY(-3px); }
  .card img { width: 100%; aspect-ratio: 1; object-fit: cover; image-rendering: pixelated; }
  .card-info { padding: .5rem .6rem .6rem; font-size: .68rem; color: var(--dim); display: flex; flex-direction: column; gap: .2rem; }
  .card-info .cat  { color: var(--acc);  font-weight: 500; }
  .card-info .dist { color: var(--acc2); }
  .spinner {
    display: none; width: 18px; height: 18px;
    border: 2px solid #0a0a0f55;
    border-top-color: #0a0a0f;
    border-radius: 50%;
    animation: spin .6s linear infinite;
    margin-left: .4rem;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .section-label {
    width: 100%;
    max-width: 860px;
    margin: 1.5rem auto 0;
    padding: 0 1.5rem;
    font-size: .7rem;
    color: var(--acc3);
    letter-spacing: .08em;
    text-transform: uppercase;
  }
</style>
"""

# ── page 1: proxy-based (original) ──────────────────────────────────────────
HTML_PROXY = """<!DOCTYPE html><html lang="en"><head>
<title>CIR Explorer · Proxy Search</title>
""" + _SHARED_HEAD + """
</head><body>
<header>
  <h1>CIR Explorer</h1>
  <span class="sub">proxy-guided nearest-neighbour search</span>
  <nav>
    <a href="/" class="active">Proxy</a>
    <a href="/query_by_example/">By Example</a>
    <a href="/query_two_images/">Two Images</a>
  </nav>
</header>

<div class="controls">
  <label>CATEGORY
    <select id="sel-cat">
      {% for obj in objects %}<option value="{{ loop.index0 }}">{{ obj }}</option>{% endfor %}
    </select>
  </label>
  <label>YEAR INDEX
    <input type="number" id="inp-year" value="0" min="0" max="{{ max_year }}" step="1">
  </label>
  <label>TOP-K
    <input type="number" id="inp-k" value="12" min="1" max="50" step="1">
  </label>
  <button onclick="query()" id="btn">SEARCH <span class="spinner" id="spin"></span></button>
</div>

<div id="status">Ready.</div>
<div class="query-vec" id="qvec"></div>
<div id="results"></div>

<script>
async function query() {
  const cat_idx  = parseInt(document.getElementById('sel-cat').value);
  const year_idx = parseInt(document.getElementById('inp-year').value);
  const k        = parseInt(document.getElementById('inp-k').value);
  setLoading(true);
  try {
    const data = await post('/search', { cat_idx, year_idx, k });
    if (data.error) { setStatus('❌ ' + data.error); return; }
    setStatus(`${data.results.length} results · query = cat_proxy[${cat_idx}] + year_proxy[${year_idx}]`);
    document.getElementById('qvec').textContent = 'query vec norm: ' + data.query_norm.toFixed(4);
    renderCards(data.results);
  } finally { setLoading(false); }
}
async function post(url, body) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  return r.json();
}
function setLoading(on) {
  document.getElementById('spin').style.display = on ? 'inline-block' : 'none';
  document.getElementById('btn').disabled = on;
}
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function renderCards(results) {
  const c = document.getElementById('results');
  c.innerHTML = '';
  results.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'card';
    d.innerHTML = `<img src="data:image/png;base64,${r.b64}" alt="result ${i}">
      <div class="card-info"><span class="cat">${r.category}</span><span>year idx: ${r.year_idx}</span><span class="dist">dist: ${r.dist.toFixed(4)}</span></div>`;
    c.appendChild(d);
  });
}
</script>
</body></html>
"""

# ── page 2: query by example (image → CNN → + year proxy) ───────────────────
HTML_BY_EXAMPLE = """<!DOCTYPE html><html lang="en"><head>
<title>CIR Explorer · By Example</title>
""" + _SHARED_HEAD + """
</head><body>
<header>
  <h1>CIR Explorer</h1>
  <span class="sub">image embedding + year proxy search</span>
  <nav>
    <a href="/">Proxy</a>
    <a href="/query_by_example/" class="active">By Example</a>
    <a href="/query_two_images/">Two Images</a>
  </nav>
</header>

<div class="section-label">Query Image (category vector via CNN)</div>
<div class="controls">
  <div class="upload-zone" id="zone-query">
    <input type="file" accept="image/*" onchange="previewAndStore(event,'zone-query','queryB64')">
    <span class="uz-label">Drop or click to upload</span>
    <span class="uz-tag">QUERY IMAGE</span>
  </div>

  <label>YEAR INDEX (proxy)
    <input type="number" id="inp-year" value="0" min="0" max="{{ max_year }}" step="1">
  </label>
  <label>TOP-K
    <input type="number" id="inp-k" value="12" min="1" max="50" step="1">
  </label>
  <button onclick="query()" id="btn">SEARCH <span class="spinner" id="spin"></span></button>
</div>

<div id="status">Upload an image to start.</div>
<div class="query-vec" id="qvec"></div>
<div id="results"></div>

<script>
let queryB64 = null;

function previewAndStore(event, zoneId, varName) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    if (varName === 'queryB64') queryB64 = e.target.result;
    const zone = document.getElementById(zoneId);
    zone.classList.add('has-img');
    let img = zone.querySelector('img');
    if (!img) { img = document.createElement('img'); zone.prepend(img); }
    img.src = e.target.result;
    zone.querySelectorAll('.uz-label').forEach(el => el.style.display = 'none');
  };
  reader.readAsDataURL(file);
}

async function query() {
  if (!queryB64) { setStatus('⚠ Please upload a query image first.'); return; }
  const year_idx = parseInt(document.getElementById('inp-year').value);
  const k        = parseInt(document.getElementById('inp-k').value);
  setLoading(true);
  try {
    const data = await post('/search_by_example', { image_b64: queryB64, year_idx, k });
    if (data.error) { setStatus('❌ ' + data.error); return; }
    setStatus(`${data.results.length} results · img_emb + year_proxy[${year_idx}]`);
    document.getElementById('qvec').textContent =
      `img emb norm: ${data.img_norm.toFixed(4)}  |  query norm: ${data.query_norm.toFixed(4)}`;
    renderCards(data.results);
  } finally { setLoading(false); }
}

async function post(url, body) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  return r.json();
}
function setLoading(on) {
  document.getElementById('spin').style.display = on ? 'inline-block' : 'none';
  document.getElementById('btn').disabled = on;
}
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function renderCards(results) {
  const c = document.getElementById('results');
  c.innerHTML = '';
  results.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'card';
    d.innerHTML = `<img src="data:image/png;base64,${r.b64}" alt="result ${i}">
      <div class="card-info"><span class="cat">${r.category}</span><span>year idx: ${r.year_idx}</span><span class="dist">dist: ${r.dist.toFixed(4)}</span></div>`;
    c.appendChild(d);
  });
}
</script>
</body></html>
"""

# ── page 3: two-image query ──────────────────────────────────────────────────
HTML_TWO_IMAGES = """<!DOCTYPE html><html lang="en"><head>
<title>CIR Explorer · Two Images</title>
""" + _SHARED_HEAD + """
<style>
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; width: 100%; max-width: 860px; margin: 1.5rem auto 0; padding: 0 1.5rem; }
  .col-block { display: flex; flex-direction: column; gap: .75rem; }
  .col-title { font-size: .7rem; color: var(--acc3); letter-spacing: .08em; text-transform: uppercase; }
  .col-desc  { font-size: .68rem; color: var(--dim); line-height: 1.5; }
  .vec-info  { font-size: .68rem; color: var(--acc2); min-height: 1em; }
</style>
</head><body>
<header>
  <h1>CIR Explorer</h1>
  <span class="sub">two-image compositional search</span>
  <nav>
    <a href="/">Proxy</a>
    <a href="/query_by_example/">By Example</a>
    <a href="/query_two_images/" class="active">Two Images</a>
  </nav>
</header>

<div class="two-col">
  <div class="col-block">
    <span class="col-title">Image A — Category</span>
    <span class="col-desc">Defines <em>what</em> to search for.<br>Embedded by the CNN → category vector.</span>
    <div class="upload-zone" id="zone-a">
      <input type="file" accept="image/*" onchange="previewImg(event,'zone-a','imgA')">
      <span class="uz-label">Drop or click · CATEGORY IMAGE</span>
    </div>
    <div class="vec-info" id="vec-a"></div>
  </div>

  <div class="col-block">
    <span class="col-title">Image B — Year Reference</span>
    <span class="col-desc">Defines <em>when</em>.<br>Embedded → minus closest category proxy → year residual.</span>
    <div class="upload-zone" id="zone-b">
      <input type="file" accept="image/*" onchange="previewImg(event,'zone-b','imgB')">
      <span class="uz-label">Drop or click · YEAR IMAGE</span>
    </div>
    <div class="vec-info" id="vec-b"></div>
  </div>
</div>

<div class="controls" style="margin-top:1.2rem;">
  <label>TOP-K
    <input type="number" id="inp-k" value="12" min="1" max="50" step="1">
  </label>
  <button onclick="query()" id="btn">SEARCH <span class="spinner" id="spin"></span></button>
</div>

<div id="status">Upload both images to start.</div>
<div class="query-vec" id="qvec"></div>
<div id="results"></div>

<script>
let imgA = null, imgB = null;

function previewImg(event, zoneId, varName) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    if (varName === 'imgA') imgA = e.target.result;
    if (varName === 'imgB') imgB = e.target.result;
    const zone = document.getElementById(zoneId);
    zone.classList.add('has-img');
    let img = zone.querySelector('img');
    if (!img) { img = document.createElement('img'); zone.prepend(img); }
    img.src = e.target.result;
    zone.querySelectorAll('.uz-label').forEach(el => el.style.display = 'none');
    // clear old info
    document.getElementById(zoneId === 'zone-a' ? 'vec-a' : 'vec-b').textContent = '';
  };
  reader.readAsDataURL(file);
}

async function query() {
  if (!imgA || !imgB) { setStatus('⚠ Please upload both images first.'); return; }
  const k = parseInt(document.getElementById('inp-k').value);
  setLoading(true);
  try {
    const data = await post('/search_two_images', { image_a_b64: imgA, image_b_b64: imgB, k });
    if (data.error) { setStatus('❌ ' + data.error); return; }
    document.getElementById('vec-a').textContent =
      `cat emb norm: ${data.cat_norm.toFixed(4)}`;
    document.getElementById('vec-b').textContent =
      `year ref norm: ${data.year_norm.toFixed(4)}  |  closest proxy: ${data.closest_proxy_idx}`;
    setStatus(`${data.results.length} results · img_A_emb + (img_B_emb − closest_cat_proxy)`);
    document.getElementById('qvec').textContent =
      `query norm: ${data.query_norm.toFixed(4)}`;
    renderCards(data.results);
  } finally { setLoading(false); }
}

async function post(url, body) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  return r.json();
}
function setLoading(on) {
  document.getElementById('spin').style.display = on ? 'inline-block' : 'none';
  document.getElementById('btn').disabled = on;
}
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function renderCards(results) {
  const c = document.getElementById('results');
  c.innerHTML = '';
  results.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'card';
    d.innerHTML = `<img src="data:image/png;base64,${r.b64}" alt="result ${i}">
      <div class="card-info"><span class="cat">${r.category}</span><span>year idx: ${r.year_idx}</span><span class="dist">dist: ${r.dist.toFixed(4)}</span></div>`;
    c.appendChild(d);
  });
}
</script>
</body></html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

def create_flask_app(ann: AnnoyIndex, meta: dict, objects: list,
                     model, contrastive_loss_fn, year_loss_fn, device):
    """Return a configured Flask app."""
    app = Flask(__name__)

    cat_proxies  = contrastive_loss_fn.proxies.detach().cpu()   # (n_obj, 1024)
    year_proxies = year_loss_fn.proxies.detach().cpu()          # (n_years, 1024)
    max_year_idx = year_proxies.shape[0] - 1

    # ── helpers ──────────────────────────────────────────────────────────────

    def _run_ann(query_vec: torch.Tensor, k: int):
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

    # ── route: / (proxy search – original) ───────────────────────────────────

    @app.route("/")
    def index():
        return render_template_string(HTML_PROXY, objects=objects, max_year=max_year_idx)

    @app.route("/search", methods=["POST"])
    def search():
        body     = request.get_json()
        cat_idx  = int(body.get("cat_idx",  0))
        year_idx = int(body.get("year_idx", 0))
        k        = int(body.get("k",        12))

        if cat_idx >= cat_proxies.shape[0]:
            return jsonify({"error": f"cat_idx {cat_idx} out of range"}), 400
        if year_idx > max_year_idx:
            return jsonify({"error": f"year_idx {year_idx} out of range (max {max_year_idx})"}), 400

        query_vec = cat_proxies[cat_idx] + year_proxies[year_idx]
        results, qv_normed = _run_ann(query_vec, k)
        return jsonify({"results": results, "query_norm": float(qv_normed.norm().item())})

    # ── route: /query_by_example/ ─────────────────────────────────────────────

    @app.route("/query_by_example/")
    def page_by_example():
        return render_template_string(HTML_BY_EXAMPLE, objects=objects, max_year=max_year_idx)

    @app.route("/search_by_example", methods=["POST"])
    def search_by_example():
        """
        Body: { image_b64: "<data-URI or raw b64>", year_idx: int, k: int }

        query_vec = CNN(image) + year_proxy[year_idx]
        """
        body = request.get_json()
        year_idx = int(body.get("year_idx", 0))
        k        = int(body.get("k", 12))

        if year_idx > max_year_idx:
            return jsonify({"error": f"year_idx {year_idx} out of range (max {max_year_idx})"}), 400

        try:
            pil_img = pil_from_b64(body["image_b64"])
        except Exception as e:
            return jsonify({"error": f"Could not decode image: {e}"}), 400

        # CNN embedding (condition=None → year-agnostic)
        img_emb = embed_pil(pil_img, model, device, normalize = False)          # (1024,) normalised

        query_vec = img_emb + year_proxies[year_idx]
        results, qv_normed = _run_ann(query_vec, k)

        return jsonify({
            "results":    results,
            "img_norm":   float(img_emb.norm().item()),
            "query_norm": float(qv_normed.norm().item()),
        })

    # ── route: /query_two_images/ ─────────────────────────────────────────────

    @app.route("/query_two_images/")
    def page_two_images():
        return render_template_string(HTML_TWO_IMAGES, objects=objects, max_year=max_year_idx)

    @app.route("/search_two_images", methods=["POST"])
    def search_two_images():
        """
        Body: { image_a_b64: "...", image_b_b64: "...", k: int }

        category_vec  = CNN(image_A)
        year_img_emb  = CNN(image_B)
        closest_proxy = argmin_i  dist(year_img_emb, cat_proxy_i)
        year_vec      = year_img_emb − cat_proxy[closest_proxy]
        query_vec     = category_vec + year_vec
        """
        body = request.get_json()
        k    = int(body.get("k", 12))

        try:
            pil_a = pil_from_b64(body["image_a_b64"])
            pil_b = pil_from_b64(body["image_b_b64"])
        except Exception as e:
            return jsonify({"error": f"Could not decode image(s): {e}"}), 400

        # Embed both images (condition=None for year-agnostic embedding)
        cat_emb      = embed_pil(pil_a, model, device, normalize = False)   # (1024,) normalised
        year_img_emb = embed_pil(pil_b, model, device, normalize = False)   # (1024,) normalised

        # Find the category proxy closest to image_B
        # cat_proxies: (n_obj, 1024)
        dists_to_proxies = torch.cdist(
            year_img_emb.unsqueeze(0),          # (1, 1024)
            cat_proxies,                         # (n_obj, 1024)
        ).squeeze(0)                             # (n_obj,)
        closest_proxy_idx = int(dists_to_proxies.argmin().item())

        # Year residual = image_B_emb − closest_category_proxy
        year_vec = year_img_emb - cat_proxies[closest_proxy_idx]

        # Final query: category from A, year "style" from B
        query_vec = F.normalize(cat_emb + year_vec, p=2, dim=0)
        results, qv_normed = _run_ann(query_vec, k)

        return jsonify({
            "results":           results,
            "cat_norm":          float(cat_emb.norm().item()),
            "year_norm":         float(year_vec.norm().item()),
            "closest_proxy_idx": closest_proxy_idx,
            "query_norm":        float(qv_normed.norm().item()),
        })

    return app


# ─────────────────────────────────────────────────────────────────────────────
# entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qualitative CIR explorer")
    p.add_argument("--objects",     nargs="+", required=True)
    p.add_argument("--ckpt_folder", required=True,
                   help="Folder with model_epochN.pth / contrastive_loss_epochN.pth / …")
    p.add_argument("--epoch",       type=int, default=None,
                   help="Which epoch to load (defaults to latest found).")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--port",        type=int, default=5000)
    p.add_argument("--rebuild",     action="store_true",
                   help="Force rebuild of the Annoy index even if it already exists.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epoch = args.epoch if args.epoch is not None else latest_epoch(args.ckpt_folder)
    print(f"Using checkpoint epoch {epoch} from {args.ckpt_folder}")

    if args.rebuild:
        for fname in ("qualitative.ann", "qualitative_meta.json"):
            p = os.path.join(args.ckpt_folder, fname)
            if os.path.exists(p):
                os.remove(p)
                print(f"  removed {p}")

    ann, meta = build_annoy_index(
        objects=args.objects,
        ckpt_folder=args.ckpt_folder,
        epoch=epoch,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, contrastive_loss_fn, year_loss_fn, _ = load_everything(
        args.ckpt_folder, args.objects, epoch, device
    )

    flask_app = create_flask_app(
        ann, meta, args.objects,
        model, contrastive_loss_fn, year_loss_fn, device
    )
    print(f"\n🚀  Serving on http://0.0.0.0:{args.port}")
    print(f"    /                   → proxy search")
    print(f"    /query_by_example/  → image + year proxy")
    print(f"    /query_two_images/  → two-image compositional search\n")
    flask_app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()