"""
CUDA_VISIBLE_DEVICES=6 python clip_baseline_1.py     --objects Window Man Building House Train Shirt Boy Tower Door "Street light"               Trousers Tree "Vehicle registration plate" Dress Woman Skirt               "Human head" Poster Tie Car Girl Belt Coat Jeans Tire Billboard               Castle Shorts Wheel "Parking meter" Hat Glasses Bus Watch Footwear               "Human face" Suit Taxi Ladder "Human ear" "Traffic sign" Jacket               Person Stairs "Traffic light" Bench Sock Van Curtain Boat     --ckpt_folder /data/113-2/users/amolina/cir_date/clip_baseline/ --freq 5 --batch_size 4096 --num_workers 8 

evaluation_clip.py

CLIP baseline for Composed Image Retrieval (CIR) evaluation.
Mirrors the three evaluation tracks in evaluate_cir.py, replacing the
learned CIR model with an OpenCLIP ViT-B/32 encoder + text prompts.

───────────────────────────────────────────────────────────────────────
Evaluation tracks  (matching evaluate_cir.py §Evaluation Protocol)
───────────────────────────────────────────────────────────────────────

All retrieval metrics are computed at rank K.

1. LABEL + LABEL  (query = μ^c + μ^y)
   CLIP equivalent: query = T("A photo of {c}") + T("A photo in {year_y}")
   - Object Precision@K : fraction of top-K whose category == c
   - Date   Precision@K : fraction of top-K whose year_idx  == y
   Averaged over all (c, y) pairs present in the test set.

2. IMAGE + LABEL  (query = f(x_c) + μ^y)
   CLIP equivalent: query = V(x_c) + T("A photo in {year_y}")
   - Object Precision@K : fraction of top-K whose category == c(x_c)
   - Date   Precision@K : fraction of top-K whose year_idx  == y
   Averaged over all (x_c, y) pairs.

3. IMAGE + IMAGE  (query = f(x_c) + r_y,  r_y = f(x_y) − μ^ĉ)
   CLIP equivalent:
     ĉ         = argmax_c  cos(V(x_y), T("A photo of {c}"))
     r_y       = V(x_y) − T("A photo of {ĉ}")
     query_vec = V(x_c) + r_y
   Reference image x_y is sampled uniformly from the train index.
   - Object Precision@K : fraction of top-K whose category == c(x_c)
   - Date   Precision@K : fraction of top-K whose year_idx  == year(x_y)
   Averaged over all query images.

Usage
─────
CUDA_VISIBLE_DEVICES=0 python evaluation_clip.py \
    --objects Window Man Building House Train ... \
    --ckpt_folder /data/foo/clip_cir/ \
    [--rebuild] [--top_k 10] [--min_date 1930] [--max_date 1999] [--freq 5]
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex
from tqdm import tqdm
from torch.utils.data import DataLoader

import open_clip

# ── local imports ─────────────────────────────────────────────────────────────
from core_datautils import df as data_complete
from core_datautils import df_test as data_test
from train_experts_dataloader import SpecialistDataloaderWithClass

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIM     = 512
N_TREES           = 50
TOP_K             = 10
OPENCLIP_MODEL    = "ViT-B-32"
OPENCLIP_PRETRAIN = "laion2b_s34b_b79k"

MIN_DATE = 1930
MAX_DATE = 1999
FREQ     = 5


# ─────────────────────────────────────────────────────────────────────────────
# Text-prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def year_idx_to_year(idx: int, year_labels: list) -> int:
    return year_labels[idx]


def year_prompt(year: int) -> str:
    return f"A photo in {year}"


def category_prompt(category: str) -> str:
    return f"A photo of {category}"


# ─────────────────────────────────────────────────────────────────────────────
# CLIP loader + encode helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAIN, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    return model, preprocess, tokenizer


def encode_text(clip_model, texts: list, device, tokenizer) -> torch.Tensor:
    """Returns L2-normalised (N, D) CPU tensor."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
    return F.normalize(feats.float(), p=2, dim=1).cpu()


def encode_images_batch(clip_model, images: torch.Tensor, device) -> torch.Tensor:
    """images: (B, 3, H, W) CLIP-preprocessed.  Returns L2-normalised (B, D) CPU tensor."""
    with torch.no_grad():
        feats = clip_model.encode_image(images.to(device))
    return F.normalize(feats.float(), p=2, dim=1).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_avlabels(min_date=MIN_DATE, max_date=MAX_DATE, freq=FREQ) -> list:
    return list(range(min_date, max_date + 1, freq))


def build_loader(df_split, objects: list, avlabels: list,
                 preprocess, batch_size: int, num_workers: int,
                 evaluate: bool = False) -> DataLoader:
    dataset = SpecialistDataloaderWithClass(
        df_split, objects[0], transforms=preprocess, evaluate=evaluate
    )
    dataset.available_labels = avlabels
    for obj in objects[1:]:
        extra = SpecialistDataloaderWithClass(
            df_split, obj, transforms=preprocess, evaluate=evaluate
        )
        dataset = dataset + extra
        dataset.available_labels = avlabels
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# Annoy index  (CLIP vision embeddings of the TRAIN split)
# ─────────────────────────────────────────────────────────────────────────────

def build_annoy_index(
    objects: list, ckpt_folder: str, avlabels: list,
    clip_model, preprocess, device,
    batch_size: int = 64, num_workers: int = 4,
    rebuild: bool = False,
):
    """
    Build (or load) an Annoy index of CLIP vision embeddings over the TRAIN split.
    Meta entries: { category, year_idx }
    Embeddings stored L2-normalised (same convention as evaluate_cir.py).
    """
    annoy_path = os.path.join(ckpt_folder, "clip_cir_eval.ann")
    meta_path  = os.path.join(ckpt_folder, "clip_cir_eval_meta.json")

    os.makedirs(ckpt_folder, exist_ok=True)

    if not rebuild and os.path.exists(annoy_path) and os.path.exists(meta_path):
        print("✔  CLIP Annoy index already exists – loading.")
        ann = AnnoyIndex(EMBEDDING_DIM, "euclidean")
        ann.load(annoy_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return ann, meta

    print("Building CLIP Annoy index from TRAIN split…")
    loader = build_loader(data_complete, objects, avlabels, preprocess,
                          batch_size, num_workers, evaluate=False)

    ann        = AnnoyIndex(EMBEDDING_DIM, "euclidean")
    meta       = {}
    global_idx = 0

    for image_A, condition_A, image_B, condition_B, category in tqdm(loader, desc="Indexing train"):
        embs = encode_images_batch(clip_model, image_A, device)   # (B, 512) normalised
        for i in range(len(embs)):
            ann.add_item(global_idx, embs[i].tolist())
            cat = category[i] if isinstance(category[i], str) else category[i]
            meta[str(global_idx)] = {
                "category": cat,
                "year_idx": int(condition_B[i].item()),
            }
            global_idx += 1

    ann.build(N_TREES)
    ann.save(annoy_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"✔  CLIP Annoy index built: {global_idx} train items → {annoy_path}")
    return ann, meta


# ─────────────────────────────────────────────────────────────────────────────
# Low-level query helpers  (same API as evaluate_cir.py)
# ─────────────────────────────────────────────────────────────────────────────

def ann_query(ann: AnnoyIndex, query_vec: torch.Tensor, k: int):
    """L2-normalise query_vec, return k nearest neighbours."""
    qv = F.normalize(query_vec, p=2, dim=0).numpy().tolist()
    indices, distances = ann.get_nns_by_vector(qv, k, include_distances=True)
    return indices, distances


def get_neighbours(ann: AnnoyIndex, meta: dict,
                   query_vec: torch.Tensor, k: int) -> list:
    indices, _ = ann_query(ann, query_vec, k)
    return [meta[str(idx)] for idx in indices]


def precision_at_k(neighbours: list, key: str, target_value) -> float:
    """Fraction of retrieved neighbours whose meta[key] == target_value."""
    return float(np.mean([int(n[key] == target_value) for n in neighbours]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL + LABEL   query = T(category c) + T(year y)
# ─────────────────────────────────────────────────────────────────────────────

def eval_label_label(
    ann: AnnoyIndex, meta: dict,
    cat_text_embs: torch.Tensor,    # (n_obj, D)  "A photo of {c}"
    year_text_embs: torch.Tensor,   # (n_yr,  D)  "A photo in {year}"
    objects: list,
    n_years: int,
    top_k: int,
):
    """
    Mirrors evaluate_cir.py::eval_label_label exactly, substituting
    learned proxies with CLIP text embeddings:
        μ^c  →  T("A photo of {c}")
        μ^y  →  T("A photo in {year_y}")
    """
    n_obj = len(objects)

    obj_prec_sum  = np.zeros((n_obj, n_years))
    date_prec_sum = np.zeros((n_obj, n_years))

    for c_idx, obj in enumerate(tqdm(objects, desc="Label+Label")):
        mu_c = cat_text_embs[c_idx]                       # (D,)
        for y in range(n_years):
            mu_y      = year_text_embs[y]                 # (D,)
            query_vec = mu_c + mu_y                       # (D,)  — un-normalised sum

            neighbours = get_neighbours(ann, meta, query_vec, top_k)

            obj_prec_sum[c_idx, y]  = precision_at_k(neighbours, "category", obj)
            date_prec_sum[c_idx, y] = precision_at_k(neighbours, "year_idx", y)

    obj_prec_mean  = float(obj_prec_sum.mean())
    date_prec_mean = float(date_prec_sum.mean())

    results = {
        "label_label_object_precision_at_k": obj_prec_mean,
        "label_label_date_precision_at_k":   date_prec_mean,
        "label_label_obj_prec_matrix":  {objects[c]: obj_prec_sum[c].tolist()  for c in range(n_obj)},
        "label_label_date_prec_matrix": {objects[c]: date_prec_sum[c].tolist() for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE + LABEL   query = V(x_c) + T(year y)
# ─────────────────────────────────────────────────────────────────────────────

def eval_image_label(
    ann: AnnoyIndex, meta: dict,
    year_text_embs: torch.Tensor,   # (n_yr, D)
    test_loader: DataLoader,
    clip_model, device,
    objects: list,
    n_years: int,
    top_k: int,
):
    """
    Mirrors evaluate_cir.py::eval_image_label, replacing:
        f(x_c)  →  V(x_c)   (CLIP vision embedding, un-normalised before addition)
        μ^y     →  T("A photo in {year_y}")

    Note: CLIP vision embeddings are already unit-norm after encode_images_batch,
    which corresponds to the normalised CIR embeddings.  We therefore add the
    (also unit-norm) text embedding directly, matching the residual-addition
    scheme in evaluate_cir.py.
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_prec_sum  = np.zeros((n_obj, n_years))
    date_prec_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Image+Label"):
        # L2-normalised CLIP vision embeddings
        embs      = encode_images_batch(clip_model, image_A, device)  # (B, D)
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(embs)):
            emb      = embs[i]                            # (D,) normalised
            true_cat = true_cats[i]
            cat_idx  = obj2idx[true_cat]

            for y in range(n_years):
                query_vec  = emb + year_text_embs[y]     # (D,)
                neighbours = get_neighbours(ann, meta, query_vec, top_k)

                obj_prec_sum[cat_idx, y]  += precision_at_k(neighbours, "category", true_cat)
                date_prec_sum[cat_idx, y] += precision_at_k(neighbours, "year_idx", y)
                count_matrix[cat_idx, y]  += 1

    safe     = np.where(count_matrix > 0, count_matrix, 1)
    obj_mat  = obj_prec_sum  / safe
    date_mat = date_prec_sum / safe

    results = {
        "image_label_object_precision_at_k": float(obj_mat.mean()),
        "image_label_date_precision_at_k":   float(date_mat.mean()),
        "image_label_obj_prec_by_category":  {objects[c]: float(obj_mat[c].mean())  for c in range(n_obj)},
        "image_label_date_prec_by_year":     {y: float(date_mat[:, y].mean())       for y in range(n_years)},
        "image_label_obj_prec_matrix":       {objects[c]: obj_mat[c].tolist()       for c in range(n_obj)},
        "image_label_date_prec_matrix":      {objects[c]: date_mat[c].tolist()      for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE + IMAGE   query = V(x_c) + r_y,  r_y = V(x_y) − T("A photo of {ĉ}")
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_train_ref(ann: AnnoyIndex, meta: dict, n_items: int):
    """
    Sample a random train embedding from the Annoy index.
    The stored vector is already L2-normalised (no norm field needed for CLIP).
    Returns (ref_emb, ref_category, ref_year_idx).
    """
    idx     = random.randrange(n_items)
    m       = meta[str(idx)]
    ref_emb = torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)  # unit-norm
    return ref_emb, m["category"], m["year_idx"]


def eval_image_image(
    ann: AnnoyIndex, meta: dict,
    cat_text_embs: torch.Tensor,    # (n_obj, D)  "A photo of {c}"
    n_train_items: int,
    test_loader: DataLoader,
    clip_model, device,
    objects: list,
    n_years: int,
    top_k: int,
):
    """
    Mirrors evaluate_cir.py::eval_image_image, replacing learned proxies with
    CLIP text embeddings:

        ĉ         = argmax_c  cos(V(x_y),  T("A photo of {c}"))
        r_y       = V(x_y)   − T("A photo of {ĉ}")
        query_vec = V(x_c)   + r_y

    Because both V(·) and T(·) are unit-norm, this residual carries the
    temporal "style" of x_y while stripping its category component —
    the same conceptual operation as in evaluate_cir.py.
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_prec_sum  = np.zeros((n_obj, n_years))
    date_prec_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    all_obj_prec  = []
    all_date_prec = []

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Image+Image"):
        embs      = encode_images_batch(clip_model, image_A, device)  # (B, D) normalised
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(embs)):
            cat_emb  = embs[i]                            # (D,) normalised
            true_cat = true_cats[i]
            cat_idx  = obj2idx[true_cat]

            # ── temporal reference ────────────────────────────────────────
            ref_emb, ref_cat, ref_year = sample_random_train_ref(
                ann, meta, n_train_items
            )

            # infer category of reference image: argmax cosine sim vs text proxies
            # ref_emb is unit-norm; cat_text_embs are unit-norm → dot = cosine
            sims_cat        = (ref_emb.unsqueeze(0) @ cat_text_embs.T).squeeze(0)  # (n_obj,)
            closest_cat_idx = int(sims_cat.argmax())

            # extract temporal residual (strip category text proxy from ref embedding)
            r_y = ref_emb - cat_text_embs[closest_cat_idx]   # (D,)

            # ── compose and retrieve ──────────────────────────────────────
            query_vec  = cat_emb + r_y                        # (D,)
            neighbours = get_neighbours(ann, meta, query_vec, top_k)

            op = precision_at_k(neighbours, "category", true_cat)
            dp = precision_at_k(neighbours, "year_idx", ref_year)

            all_obj_prec.append(op)
            all_date_prec.append(dp)

            obj_prec_sum[cat_idx, ref_year]  += op
            date_prec_sum[cat_idx, ref_year] += dp
            count_matrix[cat_idx, ref_year]  += 1

    safe     = np.where(count_matrix > 0, count_matrix, 1)
    obj_mat  = obj_prec_sum  / safe
    date_mat = date_prec_sum / safe

    results = {
        "image_image_object_precision_at_k": float(np.mean(all_obj_prec)),
        "image_image_date_precision_at_k":   float(np.mean(all_date_prec)),
        "image_image_obj_prec_by_category":  {objects[c]: float(obj_mat[c].mean())  for c in range(n_obj)},
        "image_image_date_prec_by_ref_year": {y: float(date_mat[:, y].mean())       for y in range(n_years)},
        "image_image_obj_prec_matrix":       {objects[c]: obj_mat[c].tolist()       for c in range(n_obj)},
        "image_image_date_prec_matrix":      {objects[c]: date_mat[c].tolist()      for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers  (same API as evaluate_cir.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_scalars(d: dict):
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<55s}: {v:.4f}")


def print_matrix(matrix: dict, col_labels: list, title: str, fmt: str = ".3f"):
    col_w = 8
    head  = " " * 24 + "".join(f"{str(l):>{col_w}}" for l in col_labels)
    print(f"\n  {title}")
    print("  " + head)
    for row_label, row in matrix.items():
        row_str = "".join(f"{v:{col_w}{fmt}}" for v in row)
        print(f"  {str(row_label):<24}{row_str}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLIP baseline – CIR offline evaluation")
    p.add_argument("--objects",     nargs="+", required=True)
    p.add_argument("--ckpt_folder", required=True,
                   help="Directory for caching the Annoy index and results.")
    p.add_argument("--top_k",       type=int, default=TOP_K)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--rebuild",     action="store_true",
                   help="Force rebuild of the CLIP Annoy index.")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--min_date",    type=int, default=MIN_DATE)
    p.add_argument("--max_date",    type=int, default=MAX_DATE)
    p.add_argument("--freq",        type=int, default=FREQ)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"CLIP model : {OPENCLIP_MODEL} / {OPENCLIP_PRETRAIN}")
    print(f"TOP_K      : {args.top_k}")

    # ── year / category label lists ───────────────────────────────────────────
    avlabels    = build_avlabels(args.min_date, args.max_date, args.freq)
    year_labels = avlabels                           # actual year values
    n_years     = len(year_labels)
    objects     = args.objects
    print(f"Year labels: {year_labels[0]} … {year_labels[-1]}  ({n_years} bins)")
    print(f"Objects    : {objects}")

    # ── load CLIP ─────────────────────────────────────────────────────────────
    clip_model, preprocess, tokenizer = load_clip(device)

    # ── pre-compute all text embeddings (= proxies μ^c, μ^y) ─────────────────
    print("\nEncoding category text proxies …")
    cat_prompts    = [category_prompt(o) for o in objects]
    cat_text_embs  = encode_text(clip_model, cat_prompts, device, tokenizer)   # (n_obj, D)

    print("Encoding year text proxies …")
    year_prompts   = [year_prompt(y) for y in year_labels]
    year_text_embs = encode_text(clip_model, year_prompts, device, tokenizer)  # (n_yr, D)

    print(f"  Category proxies : {cat_text_embs.shape}")
    print(f"  Year proxies     : {year_text_embs.shape}")

    # ── build / load Annoy index ──────────────────────────────────────────────
    ann, meta = build_annoy_index(
        objects=objects, ckpt_folder=args.ckpt_folder, avlabels=avlabels,
        clip_model=clip_model, preprocess=preprocess, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        rebuild=args.rebuild,
    )
    n_train_items = len(meta)
    print(f"  Annoy index size : {n_train_items} train items")

    # ── build test dataloader ─────────────────────────────────────────────────
    test_loader = build_loader(
        data_test, objects, avlabels, preprocess,
        args.batch_size, args.num_workers, evaluate=True,
    )
    print(f"  Test dataset size: {len(test_loader.dataset)} items")

    year_indices = list(range(n_years))   # column labels for matrices

    # ── 1. LABEL + LABEL ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1 / 3  LABEL + LABEL   (query = T(c) + T(year_y))")
    print("=" * 60)
    ll_results = eval_label_label(
        ann, meta, cat_text_embs, year_text_embs,
        objects, n_years, args.top_k,
    )
    print_scalars(ll_results)
    print_matrix(ll_results["label_label_obj_prec_matrix"],  year_indices,
                 "Object Precision@K matrix [category × year]")
    print_matrix(ll_results["label_label_date_prec_matrix"], year_indices,
                 "Date   Precision@K matrix [category × year]")

    # ── 2. IMAGE + LABEL ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2 / 3  IMAGE + LABEL   (query = V(x_c) + T(year_y))")
    print("=" * 60)
    il_results = eval_image_label(
        ann, meta, year_text_embs, test_loader, clip_model, device,
        objects, n_years, args.top_k,
    )
    print_scalars(il_results)
    print_matrix(il_results["image_label_obj_prec_matrix"],  year_indices,
                 "Object Precision@K matrix [category × target year]")
    print_matrix(il_results["image_label_date_prec_matrix"], year_indices,
                 "Date   Precision@K matrix [category × target year]")

    # ── 3. IMAGE + IMAGE ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3 / 3  IMAGE + IMAGE   (query = V(x_c) + r_y)")
    print("=" * 60)
    ii_results = eval_image_image(
        ann, meta, cat_text_embs, n_train_items, test_loader, clip_model, device,
        objects, n_years, args.top_k,
    )
    print_scalars(ii_results)
    print_matrix(ii_results["image_image_obj_prec_matrix"],  year_indices,
                 "Object Precision@K matrix [query category × ref year]")
    print_matrix(ii_results["image_image_date_prec_matrix"], year_indices,
                 "Date   Precision@K matrix [query category × ref year]")

    # ── scalar summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALAR SUMMARY")
    print("=" * 60)
    all_results = {**ll_results, **il_results, **ii_results}
    print_scalars(all_results)

    # ── save ──────────────────────────────────────────────────────────────────
    summary_path = os.path.join(
        args.ckpt_folder, f"clip_cir_eval_topk{args.top_k}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔  Full results saved to {summary_path}")


if __name__ == "__main__":
    main()