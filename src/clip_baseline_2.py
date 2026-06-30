"""
CUDA_VISIBLE_DEVICES=0 python clip_baseline_combined.py \
    --objects Window Man Building House Train Shirt Boy Tower Door "Street light" \
              Trousers Tree "Vehicle registration plate" Dress Woman Skirt \
              "Human head" Poster Tie Car Girl Belt Coat Jeans Tire Billboard \
              Castle Shorts Wheel "Parking meter" Hat Glasses Bus Watch Footwear \
              "Human face" Suit Taxi Ladder "Human ear" "Traffic sign" Jacket \
              Person Stairs "Traffic light" Bench Sock Van Curtain Boat \
    --ckpt_folder /data/foo/clip_combined/ \
    --freq 5 --batch_size 4096 --num_workers 8

clip_baseline_combined.py

CLIP baseline for Composed Image Retrieval (CIR) evaluation.

Combines:
  - The three evaluation TRACKS from clip_baseline_1.py (Label+Label,
    Image+Label, Image+Image) and their METRICS (Object Precision@K,
    Date Precision@K).
  - The RERANKING strategy from clip_baseline_2.py: for each track, in
    addition to the original direct-query retrieval, a reranked variant
    first fetches a larger candidate pool then re-ranks it with the
    complementary modality signal.

───────────────────────────────────────────────────────────────────────
Evaluation tracks
───────────────────────────────────────────────────────────────────────

1. LABEL + LABEL   (query = T(c) + T(year_y))
   Direct   : retrieve top-K by combined text embedding.
   Reranked : retrieve pool by T(c), re-rank by cosine sim to T(year_y).

2. IMAGE + LABEL   (query = V(x_c) + T(year_y))
   Direct   : retrieve top-K by combined image+text embedding.
   Reranked : retrieve pool by V(x_c), re-rank by cosine sim to T(year_y).

3. IMAGE + IMAGE   (query = V(x_c) + r_y,  r_y = V(x_y) − T("A photo of {ĉ}"))
   Direct   : retrieve top-K by composed query vector.
   Reranked : retrieve pool by V(x_c), re-rank by cosine sim to V(x_y).

All metrics (Object Precision@K, Date Precision@K) are identical to
clip_baseline_1.py and are averaged over all (category, year) pairs.
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
RERANK_POOL       = 100          # candidate pool size before re-ranking
OPENCLIP_MODEL    = "ViT-B-32"
OPENCLIP_PRETRAIN = "laion2b_s34b_b79k"

MIN_DATE = 1930
MAX_DATE = 1999
FREQ     = 5


# ─────────────────────────────────────────────────────────────────────────────
# Text-prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """images: (B, 3, H, W) CLIP-preprocessed. Returns L2-normalised (B, D) CPU tensor."""
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
    Meta entries: { category, year_idx }.
    Embeddings stored L2-normalised.
    """
    annoy_path = os.path.join(ckpt_folder, "clip_cir_eval.ann")
    meta_path  = os.path.join(ckpt_folder, "clip_cir_eval_meta.json")

    os.makedirs(ckpt_folder, exist_ok=True)

    # if not rebuild and os.path.exists(annoy_path) and os.path.exists(meta_path):
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
        embs = encode_images_batch(clip_model, image_A, device)   # (B, 512)
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
# Low-level ANN query helpers
# ─────────────────────────────────────────────────────────────────────────────

def ann_query(ann: AnnoyIndex, query_vec: torch.Tensor, k: int):
    """L2-normalise query_vec, return (indices, distances) of k nearest neighbours."""
    qv = F.normalize(query_vec, p=2, dim=0).numpy().tolist()
    return ann.get_nns_by_vector(qv, k, include_distances=True)


def get_neighbours(ann: AnnoyIndex, meta: dict,
                   query_vec: torch.Tensor, k: int) -> list:
    """Return list of k meta dicts for nearest neighbours."""
    indices, _ = ann_query(ann, query_vec, k)
    return [meta[str(idx)] for idx in indices]


def get_pool(ann: AnnoyIndex, meta: dict,
             query_vec: torch.Tensor, pool_size: int):
    """
    Retrieve a candidate pool by ANN similarity.
    Returns (list[meta_dict], pool_vecs_tensor (pool_size, D)).
    """
    indices, _ = ann_query(ann, query_vec, pool_size)
    metas = [meta[str(idx)] for idx in indices]
    vecs  = torch.stack([
        torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)
        for idx in indices
    ])  # (pool, D) – already L2-normalised
    vecs = F.normalize(vecs, p=2, dim=1)
    return metas, vecs


def rerank(pool_metas: list, pool_vecs: torch.Tensor,
           rerank_emb: torch.Tensor, top_k: int) -> list:
    """
    Re-rank a candidate pool by cosine similarity to `rerank_emb`.
    Returns the top-K meta dicts.
    """
    rerank_emb = F.normalize(rerank_emb, p=2, dim=0)
    scores     = (pool_vecs @ rerank_emb).numpy()   # (pool,)
    top_k_pos  = np.argsort(scores)[::-1][:top_k]
    return [pool_metas[p] for p in top_k_pos]


def precision_at_k(neighbours: list, key: str, target_value) -> float:
    """Fraction of retrieved neighbours whose meta[key] == target_value."""
    return float(np.mean([int(n[key] == target_value) for n in neighbours]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL + LABEL
# ─────────────────────────────────────────────────────────────────────────────

def eval_label_label(
    ann: AnnoyIndex, meta: dict,
    cat_text_embs: torch.Tensor,    # (n_obj, D)
    year_text_embs: torch.Tensor,   # (n_yr,  D)
    objects: list,
    n_years: int,
    top_k: int,
    rerank_pool: int,
):
    """
    Direct   : query = T(c) + T(year_y) → retrieve top-K directly.
    Reranked : retrieve `rerank_pool` by T(c), re-rank by cosine sim to T(year_y).

    Metrics  : Object Precision@K, Date Precision@K  (same as baseline_1).
    """
    n_obj = len(objects)

    # Accumulators – one cell per (category, year) pair
    direct_obj_sum   = np.zeros((n_obj, n_years))
    direct_date_sum  = np.zeros((n_obj, n_years))
    rerank_obj_sum   = np.zeros((n_obj, n_years))
    rerank_date_sum  = np.zeros((n_obj, n_years))

    for c_idx, obj in enumerate(tqdm(objects, desc="Label+Label")):
        mu_c = cat_text_embs[c_idx]                          # (D,)

        for y in range(n_years):
            mu_y = year_text_embs[y]                          # (D,)

            # ── Direct: sum then retrieve ────────────────────────────────
            query_vec  = mu_c + mu_y
            neighbours = get_neighbours(ann, meta, query_vec, top_k)
            direct_obj_sum[c_idx, y]  = precision_at_k(neighbours, "category", obj)
            direct_date_sum[c_idx, y] = precision_at_k(neighbours, "year_idx",  y)

            # ── Reranked: retrieve by T(c), re-rank by T(year_y) ────────
            pool_metas, pool_vecs = get_pool(ann, meta, mu_c, rerank_pool)
            neighbours            = rerank(pool_metas, pool_vecs, mu_y, top_k)
            rerank_obj_sum[c_idx, y]  = precision_at_k(neighbours, "category", obj)
            rerank_date_sum[c_idx, y] = precision_at_k(neighbours, "year_idx",  y)

    results = {
        # scalar summaries
        "label_label_direct_object_precision_at_k":  float(direct_obj_sum.mean()),
        "label_label_direct_date_precision_at_k":    float(direct_date_sum.mean()),
        "label_label_rerank_object_precision_at_k":  float(rerank_obj_sum.mean()),
        "label_label_rerank_date_precision_at_k":    float(rerank_date_sum.mean()),
        # per-category / per-year breakdown
        "label_label_direct_obj_prec_matrix":  {objects[c]: direct_obj_sum[c].tolist()  for c in range(n_obj)},
        "label_label_direct_date_prec_matrix": {objects[c]: direct_date_sum[c].tolist() for c in range(n_obj)},
        "label_label_rerank_obj_prec_matrix":  {objects[c]: rerank_obj_sum[c].tolist()  for c in range(n_obj)},
        "label_label_rerank_date_prec_matrix": {objects[c]: rerank_date_sum[c].tolist() for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE + LABEL
# ─────────────────────────────────────────────────────────────────────────────

def eval_image_label(
    ann: AnnoyIndex, meta: dict,
    year_text_embs: torch.Tensor,   # (n_yr, D)
    test_loader: DataLoader,
    clip_model, device,
    objects: list,
    n_years: int,
    top_k: int,
    rerank_pool: int,
):
    """
    Direct   : query = V(x_c) + T(year_y) → retrieve top-K directly.
    Reranked : retrieve `rerank_pool` by V(x_c), re-rank by cosine sim to T(year_y).

    Metrics  : Object Precision@K, Date Precision@K.
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    direct_obj_sum   = np.zeros((n_obj, n_years))
    direct_date_sum  = np.zeros((n_obj, n_years))
    rerank_obj_sum   = np.zeros((n_obj, n_years))
    rerank_date_sum  = np.zeros((n_obj, n_years))
    count_matrix     = np.zeros((n_obj, n_years), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Image+Label"):
        embs      = encode_images_batch(clip_model, image_A, device)  # (B, D)
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(embs)):
            emb      = embs[i]
            true_cat = true_cats[i]
            cat_idx  = obj2idx[true_cat]

            # Fetch the candidate pool once per image (shared across years)
            pool_metas, pool_vecs = get_pool(ann, meta, emb, rerank_pool)

            for y in range(n_years):
                mu_y = year_text_embs[y]

                # ── Direct ───────────────────────────────────────────────
                query_vec  = emb + mu_y
                neighbours = get_neighbours(ann, meta, query_vec, top_k)
                direct_obj_sum[cat_idx, y]  += precision_at_k(neighbours, "category", true_cat)
                direct_date_sum[cat_idx, y] += precision_at_k(neighbours, "year_idx",  y)

                # ── Reranked ─────────────────────────────────────────────
                neighbours = rerank(pool_metas, pool_vecs, mu_y, top_k)
                rerank_obj_sum[cat_idx, y]  += precision_at_k(neighbours, "category", true_cat)
                rerank_date_sum[cat_idx, y] += precision_at_k(neighbours, "year_idx",  y)

                count_matrix[cat_idx, y] += 1

    safe = np.where(count_matrix > 0, count_matrix, 1)
    dm_obj   = direct_obj_sum  / safe
    dm_date  = direct_date_sum / safe
    rk_obj   = rerank_obj_sum  / safe
    rk_date  = rerank_date_sum / safe

    results = {
        "image_label_direct_object_precision_at_k":  float(dm_obj.mean()),
        "image_label_direct_date_precision_at_k":    float(dm_date.mean()),
        "image_label_rerank_object_precision_at_k":  float(rk_obj.mean()),
        "image_label_rerank_date_precision_at_k":    float(rk_date.mean()),
        "image_label_direct_obj_prec_by_category":   {objects[c]: float(dm_obj[c].mean())  for c in range(n_obj)},
        "image_label_direct_date_prec_by_year":      {y: float(dm_date[:, y].mean())       for y in range(n_years)},
        "image_label_rerank_obj_prec_by_category":   {objects[c]: float(rk_obj[c].mean())  for c in range(n_obj)},
        "image_label_rerank_date_prec_by_year":      {y: float(rk_date[:, y].mean())       for y in range(n_years)},
        "image_label_direct_obj_prec_matrix":        {objects[c]: dm_obj[c].tolist()        for c in range(n_obj)},
        "image_label_direct_date_prec_matrix":       {objects[c]: dm_date[c].tolist()       for c in range(n_obj)},
        "image_label_rerank_obj_prec_matrix":        {objects[c]: rk_obj[c].tolist()        for c in range(n_obj)},
        "image_label_rerank_date_prec_matrix":       {objects[c]: rk_date[c].tolist()       for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE + IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_train_ref(ann: AnnoyIndex, meta: dict, n_items: int):
    """
    Sample a random train embedding from the Annoy index.
    Returns (ref_emb: Tensor D, ref_category: str, ref_year_idx: int).
    """
    idx     = random.randrange(n_items)
    m       = meta[str(idx)]
    ref_emb = torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)  # unit-norm
    return ref_emb, m["category"], m["year_idx"]


def eval_image_image(
    ann: AnnoyIndex, meta: dict,
    cat_text_embs: torch.Tensor,    # (n_obj, D)
    n_train_items: int,
    test_loader: DataLoader,
    clip_model, device,
    objects: list,
    n_years: int,
    top_k: int,
    rerank_pool: int,
):
    """
    For each query image x_c and randomly sampled reference image x_y:

        ĉ   = argmax_c  cos(V(x_y), T("A photo of {c}"))
        r_y = V(x_y) − T("A photo of {ĉ}")

    Direct   : query = V(x_c) + r_y → retrieve top-K directly.
    Reranked : retrieve `rerank_pool` by V(x_c), re-rank by cosine sim to
               V(x_y) (the reference image embedding itself).

    Metrics  : Object Precision@K (category of x_c),
               Date Precision@K   (year_idx of x_y).
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    direct_obj_prec  = []
    direct_date_prec = []
    rerank_obj_prec  = []
    rerank_date_prec = []

    direct_obj_sum   = np.zeros((n_obj, n_years))
    direct_date_sum  = np.zeros((n_obj, n_years))
    rerank_obj_sum   = np.zeros((n_obj, n_years))
    rerank_date_sum  = np.zeros((n_obj, n_years))
    count_matrix     = np.zeros((n_obj, n_years), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Image+Image"):
        embs      = encode_images_batch(clip_model, image_A, device)  # (B, D)
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(embs)):
            cat_emb  = embs[i]
            true_cat = true_cats[i]
            cat_idx  = obj2idx[true_cat]

            # ── sample temporal reference from train index ────────────────
            ref_emb, ref_cat, ref_year = sample_random_train_ref(
                ann, meta, n_train_items
            )
            ref_emb = F.normalize(ref_emb, p=2, dim=0)

            # infer category of reference image via CLIP text proxies
            sims_cat        = (ref_emb.unsqueeze(0) @ cat_text_embs.T).squeeze(0)  # (n_obj,)
            closest_cat_idx = int(sims_cat.argmax())

            # temporal residual: strip inferred category from reference embedding
            r_y = ref_emb - cat_text_embs[closest_cat_idx]               # (D,)

            # ── Direct: compose then retrieve ────────────────────────────
            query_vec  = cat_emb + r_y
            neighbours = get_neighbours(ann, meta, query_vec, top_k)

            op = precision_at_k(neighbours, "category", true_cat)
            dp = precision_at_k(neighbours, "year_idx",  ref_year)
            direct_obj_prec.append(op)
            direct_date_prec.append(dp)
            direct_obj_sum[cat_idx, ref_year]  += op
            direct_date_sum[cat_idx, ref_year] += dp

            # ── Reranked: pool by V(x_c), re-rank by V(x_y) ─────────────
            pool_metas, pool_vecs = get_pool(ann, meta, cat_emb, rerank_pool)
            neighbours            = rerank(pool_metas, pool_vecs, ref_emb, top_k)

            op = precision_at_k(neighbours, "category", true_cat)
            dp = precision_at_k(neighbours, "year_idx",  ref_year)
            rerank_obj_prec.append(op)
            rerank_date_prec.append(dp)
            rerank_obj_sum[cat_idx, ref_year]  += op
            rerank_date_sum[cat_idx, ref_year] += dp

            count_matrix[cat_idx, ref_year] += 1

    safe = np.where(count_matrix > 0, count_matrix, 1)
    dm_obj  = direct_obj_sum  / safe
    dm_date = direct_date_sum / safe
    rk_obj  = rerank_obj_sum  / safe
    rk_date = rerank_date_sum / safe

    results = {
        "image_image_direct_object_precision_at_k":   float(np.mean(direct_obj_prec)),
        "image_image_direct_date_precision_at_k":     float(np.mean(direct_date_prec)),
        "image_image_rerank_object_precision_at_k":   float(np.mean(rerank_obj_prec)),
        "image_image_rerank_date_precision_at_k":     float(np.mean(rerank_date_prec)),
        "image_image_direct_obj_prec_by_category":    {objects[c]: float(dm_obj[c].mean())  for c in range(n_obj)},
        "image_image_direct_date_prec_by_ref_year":   {y: float(dm_date[:, y].mean())       for y in range(n_years)},
        "image_image_rerank_obj_prec_by_category":    {objects[c]: float(rk_obj[c].mean())  for c in range(n_obj)},
        "image_image_rerank_date_prec_by_ref_year":   {y: float(rk_date[:, y].mean())       for y in range(n_years)},
        "image_image_direct_obj_prec_matrix":         {objects[c]: dm_obj[c].tolist()        for c in range(n_obj)},
        "image_image_direct_date_prec_matrix":        {objects[c]: dm_date[c].tolist()       for c in range(n_obj)},
        "image_image_rerank_obj_prec_matrix":         {objects[c]: rk_obj[c].tolist()        for c in range(n_obj)},
        "image_image_rerank_date_prec_matrix":        {objects[c]: rk_date[c].tolist()       for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_scalars(d: dict):
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<65s}: {v:.4f}")


def print_matrix(matrix: dict, col_labels: list, title: str, fmt: str = ".3f"):
    col_w = 8
    head  = " " * 24 + "".join(f"{str(l):>{col_w}}" for l in col_labels)
    print(f"\n  {title}")
    print("  " + head)
    for row_label, row in matrix.items():
        row_str = "".join(f"{v:{col_w}{fmt}}" for v in row)
        print(f"  {str(row_label):<24}{row_str}")


def print_section(results: dict, track: str, year_indices: list, objects: list):
    """Print scalars + key matrices for a given track."""
    print_scalars(results)

    by_cat_keys = [k for k in results if k.endswith("_obj_prec_by_category")]
    for bck in by_cat_keys:
        label = bck.replace("_obj_prec_by_category", "").replace("_", " ")
        print(f"\n  Object Precision@K by category  [{label}]:")
        for cat, v in results[bck].items():
            print(f"    {cat:<22s}: {v:.4f}")

    by_yr_keys = [k for k in results if "_date_prec_by_" in k]
    for byk in by_yr_keys:
        label = byk.replace("_date_prec_by_year", "").replace("_date_prec_by_ref_year", "").replace("_", " ")
        print(f"\n  Date Precision@K by year  [{label}]:")
        for y, v in results[byk].items():
            print(f"    year_idx {y:<4}: {v:.4f}")

    for suffix, title_tmpl in [
        ("_direct_obj_prec_matrix",  "Object Prec@K  [Direct]   [category × year]"),
        ("_rerank_obj_prec_matrix",  "Object Prec@K  [Reranked] [category × year]"),
        ("_direct_date_prec_matrix", "Date   Prec@K  [Direct]   [category × year]"),
        ("_rerank_date_prec_matrix", "Date   Prec@K  [Reranked] [category × year]"),
    ]:
        key = track + suffix
        if key in results:
            print_matrix(results[key], year_indices, title_tmpl)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CLIP combined baseline – CIR offline evaluation"
    )
    p.add_argument("--objects",      nargs="+", required=True)
    p.add_argument("--ckpt_folder",  required=True,
                   help="Directory for caching the Annoy index and results.")
    p.add_argument("--top_k",        type=int, default=TOP_K)
    p.add_argument("--rerank_pool",  type=int, default=RERANK_POOL,
                   help="Candidate pool size before re-ranking.")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--rebuild",      action="store_true",
                   help="Force rebuild of the CLIP Annoy index.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--min_date",     type=int, default=MIN_DATE)
    p.add_argument("--max_date",     type=int, default=MAX_DATE)
    p.add_argument("--freq",         type=int, default=FREQ)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"CLIP model  : {OPENCLIP_MODEL} / {OPENCLIP_PRETRAIN}")
    print(f"TOP_K       : {args.top_k}")
    print(f"Rerank pool : {args.rerank_pool}")

    # ── year / category label lists ───────────────────────────────────────────
    avlabels    = build_avlabels(args.min_date, args.max_date, args.freq)
    year_labels = avlabels
    n_years     = len(year_labels)
    objects     = args.objects
    year_indices = list(range(n_years))

    print(f"Year labels : {year_labels[0]} … {year_labels[-1]}  ({n_years} bins)")
    print(f"Objects     : {objects}")

    # ── load CLIP ─────────────────────────────────────────────────────────────
    clip_model, preprocess, tokenizer = load_clip(device)

    # ── pre-compute text embeddings (= proxies μ^c, μ^y) ─────────────────────
    print("\nEncoding category text proxies …")
    cat_prompts    = [category_prompt(o) for o in objects]
    cat_text_embs  = encode_text(clip_model, cat_prompts,  device, tokenizer)  # (n_obj, D)

    print("Encoding year text proxies …")
    year_prompts   = [year_prompt(y) for y in year_labels]
    year_text_embs = encode_text(clip_model, year_prompts, device, tokenizer)  # (n_yr,  D)

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

    # ── test dataloader ───────────────────────────────────────────────────────
    test_loader = build_loader(
        data_test, objects, avlabels, preprocess,
        args.batch_size, args.num_workers, evaluate=True,
    )
    print(f"  Test dataset size: {len(test_loader.dataset)} items\n")

    # ── 1. LABEL + LABEL ──────────────────────────────────────────────────────
    print("=" * 70)
    print("1 / 3  LABEL + LABEL   (query = T(c) + T(year_y))")
    print("=" * 70)
    ll_results = eval_label_label(
        ann, meta, cat_text_embs, year_text_embs,
        objects, n_years, args.top_k, args.rerank_pool,
    )
    print_section(ll_results, "label_label", year_indices, objects)

    # ── 2. IMAGE + LABEL ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2 / 3  IMAGE + LABEL   (query = V(x_c) + T(year_y))")
    print("=" * 70)
    il_results = eval_image_label(
        ann, meta, year_text_embs, test_loader, clip_model, device,
        objects, n_years, args.top_k, args.rerank_pool,
    )
    print_section(il_results, "image_label", year_indices, objects)

    # ── 3. IMAGE + IMAGE ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3 / 3  IMAGE + IMAGE   (query = V(x_c) + r_y)")
    print("=" * 70)
    ii_results = eval_image_image(
        ann, meta, cat_text_embs, n_train_items, test_loader, clip_model, device,
        objects, n_years, args.top_k, args.rerank_pool,
    )
    print_section(ii_results, "image_image", year_indices, objects)

    # ── scalar summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SCALAR SUMMARY")
    print("=" * 70)
    all_results = {**ll_results, **il_results, **ii_results}
    print_scalars(all_results)

    # ── save ──────────────────────────────────────────────────────────────────
    summary_path = os.path.join(
        args.ckpt_folder,
        f"clip_combined_eval_topk{args.top_k}_pool{args.rerank_pool}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔  Full results saved to {summary_path}")


if __name__ == "__main__":
    main()