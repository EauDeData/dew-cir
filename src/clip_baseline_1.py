"""
CUDA_VISIBLE_DEVICES=6 python clip_baseline_1.py     --objects Window Man Building House Train Shirt Boy Tower Door "Street light"               Trousers Tree "Vehicle registration plate" Dress Woman Skirt               "Human head" Poster Tie Car Girl Belt Coat Jeans Tire Billboard               Castle Shorts Wheel "Parking meter" Hat Glasses Bus Watch Footwear               "Human face" Suit Taxi Ladder "Human ear" "Traffic sign" Jacket               Person Stairs "Traffic light" Bench Sock Van Curtain Boat     --ckpt_folder /data/113-2/users/amolina/cir_date/clip_baseline/ --freq 5 --batch_size 4096 --num_workers 8 



eval_clip_baseline_1.py

CLIP baseline for Composed Image Retrieval (CIR) evaluation.
Mirrors the four evaluation tracks in evaluate_cir.py but replaces the
learned CIR model with an OpenCLIP ViT-L/14 (laion2b_s32b_b82k) encoder
+ text prompts.

Tracks
──────
1. DATE ESTIMATION
   • Proxy-based : score image against "A photo in YYYY" for each year → argmax
   • KNN-based   : TOP_K nearest image neighbours by vision embedding → mean year

2. OBJECT ESTIMATION
   • Proxy-based : score image against "A photo of {category}" for each category → argmax
   • KNN-based   : TOP_K nearest image neighbours → majority-vote category

3. IMAGE + YEAR-PROXY TRANSLATION  (query by example → target year)
   • Retrieve top-100 candidates by image similarity, then re-rank with
     CLIP text score for "A photo in YYYY" to get final TOP_K.
   Metrics: object accuracy, year consistency.

4. TWO-IMAGE TRANSLATION
   • Retrieve top-100 candidates by image similarity to the query image,
     then re-rank by cosine similarity to the reference image embedding
     to get final TOP_K.
   Metrics: object accuracy, year consistency.

Year index ↔ year text
──────────────────────
Year labels are generated as:
    available_labels = list(range(min_date, max_date + 1, freq))
    i.e.  [1930, 1935, 1940, …, 1995]  (default min=1930, max=1999, freq=5)
So  year_idx  i  →  actual year  1930 + 5 * i.

Usage
──────
CUDA_VISIBLE_DEVICES=0 python eval_clip_baseline_1.py \
    --objects Window Man Building House ... \
    --ckpt_folder /data/foo/bar/ \
    [--rebuild] [--top_k 10] [--rerank_pool 100]
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# OpenCLIP  –  pip install open_clip_torch
import open_clip

# ── local imports ─────────────────────────────────────────────────────────────
from core_datautils import df as data_complete   # train split → index
from core_datautils import df_test as data_test  # test  split → queries
from train_experts_dataloader import SpecialistDataloaderWithClass

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIM     = 512          # OpenCLIP ViT-L/14 embedding size
N_TREES           = 50
TOP_K             = 10
RERANK_POOL       = 100          # candidate pool size before re-ranking (tracks 3 & 4)
OPENCLIP_MODEL    = 'ViT-B-32'
OPENCLIP_PRETRAIN = "laion2b_s34b_b79k"

# Year label parameters – must match those used during training
MIN_DATE = 1930
MAX_DATE = 1999
FREQ     = 5

# _NORMALIZE = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                                    std=(0.26862954, 0.26130258, 0.27577711))


# ─────────────────────────────────────────────────────────────────────────────
# Year index helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_year_labels(min_date: int = MIN_DATE,
                      max_date: int = MAX_DATE,
                      freq: int = FREQ) -> list:
    """Return the list of actual years corresponding to year indices."""
    return list(range(min_date, max_date + 1, freq))


def year_idx_to_year(idx: int, year_labels: list) -> int:
    return year_labels[idx]


def year_to_prompt(year: int) -> str:
    return f"A photo in {year}"


def category_to_prompt(category: str) -> str:
    return f"A photo of {category}"


def composed_prompt(category: str, year: int) -> str:
    return f"A photo of {category} in {year}"


# ─────────────────────────────────────────────────────────────────────────────
# CLIP loader
# ─────────────────────────────────────────────────────────────────────────────

def load_clip(device) -> tuple:
    """
    Return (model, preprocess_val, tokenizer) for the chosen OpenCLIP variant.
    open_clip.create_model_and_transforms returns
        (model, preprocess_train, preprocess_val)
    We only need the val preprocess for evaluation.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAIN, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    return model, preprocess, tokenizer


def encode_text(clip_model, texts: list, device, tokenizer) -> torch.Tensor:
    """
    Encode a list of text strings with OpenCLIP.
    Returns L2-normalised (N, D) tensor on CPU.
    """
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
    return F.normalize(feats.float(), p=2, dim=1).cpu()


def encode_images_batch(clip_model, images: torch.Tensor, device) -> torch.Tensor:
    """
    images : (B, 3, H, W) already pre-processed for OpenCLIP (float32, normalised).
    Returns L2-normalised (B, D) tensor on CPU.
    """
    with torch.no_grad():
        feats = clip_model.encode_image(images.to(device))
    return F.normalize(feats.float(), p=2, dim=1).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_clip_transforms(preprocess) -> transforms.Compose:
    """
    Wrap the CLIP preprocess so it can be used with SpecialistDataloaderWithClass
    (which expects a torchvision-compatible transform).
    CLIP's preprocess already handles resize + centre-crop + normalisation.
    """
    return preprocess


def build_avlabels(objects: list, min_date=MIN_DATE, max_date=MAX_DATE, freq=FREQ):
    """
    Reconstruct available_labels so we can reuse SpecialistDataloaderWithClass
    without loading the CIR model.
    """
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
# Annoy index  (vision embeddings of the TRAIN split)
# ─────────────────────────────────────────────────────────────────────────────

def build_clip_annoy_index(
    objects: list, ckpt_folder: str, avlabels: list,
    clip_model, preprocess, tokenizer, device,
    batch_size: int = 64, num_workers: int = 32,
    rebuild: bool = False,
):
    """
    Build (or load) an Annoy index of CLIP vision embeddings over the TRAIN split.
    Stored as  <ckpt_folder>/clip_eval.ann  and  clip_eval_meta.json.
    """
    annoy_path = os.path.join(ckpt_folder, "clip_eval.ann")
    meta_path  = os.path.join(ckpt_folder, "clip_eval_meta.json")

    os.makedirs(ckpt_folder, exist_ok = True)
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

    ann  = AnnoyIndex(EMBEDDING_DIM, "euclidean")
    meta = {}
    global_idx = 0

    for image_A, condition_A, image_B, condition_B, category in tqdm(loader, desc="Indexing train"):
        embs = encode_images_batch(clip_model, image_A, device)  # (B, 512)
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
# ANN query helpers
# ─────────────────────────────────────────────────────────────────────────────

def ann_query_vec(ann: AnnoyIndex, query_vec: torch.Tensor, k: int):
    """Query Annoy with an L2-normalised vector. Returns (indices, distances)."""
    qv = F.normalize(query_vec, p=2, dim=0).numpy().tolist()
    return ann.get_nns_by_vector(qv, k, include_distances=True)


def get_neighbours(ann: AnnoyIndex, meta: dict,
                   query_vec: torch.Tensor, k: int) -> list:
    indices, _ = ann_query_vec(ann, query_vec, k)
    return [meta[str(idx)] for idx in indices]


def get_neighbours_with_vecs(ann: AnnoyIndex, meta: dict,
                              query_vec: torch.Tensor, k: int):
    """Return (list_of_meta, list_of_embedding_tensors) for k neighbours."""
    indices, _ = ann_query_vec(ann, query_vec, k)
    metas = [meta[str(idx)] for idx in indices]
    vecs  = [torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)
             for idx in indices]
    return metas, vecs


# ─────────────────────────────────────────────────────────────────────────────
# Track 1 – DATE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_date_estimation(
    ann: AnnoyIndex, meta: dict,
    year_text_embs: torch.Tensor,   # (n_yr, 512)  "A photo in YYYY"
    year_labels: list,              # [1930, 1935, …]
    test_loader: DataLoader,
    clip_model, device, top_k: int,
):
    """
    Proxy-based  : argmax CLIP similarity between image and "A photo in YYYY"
    KNN-based    : mean year_idx of TOP_K vision neighbours → round to int
    """
    n_years = len(year_labels)

    proxy_correct, proxy_abs_err = [], []
    knn_correct,   knn_abs_err   = [], []
    print('years available', n_years, year_labels)
    proxy_confusion = np.zeros((n_years, n_years), dtype=np.int64)
    knn_confusion   = np.zeros((n_years, n_years), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Date estimation"):
        img_embs   = encode_images_batch(clip_model, image_A, device)   # (B, 512)
        true_years = condition_B.numpy()                                  # year_idx

        for i in range(len(img_embs)):
            emb       = img_embs[i]          # (512,)
            true_year = int(true_years[i])

            # ── proxy-based: cosine sim vs each year text embedding ──────
            sims           = (emb.unsqueeze(0) @ year_text_embs.T).squeeze(0)  # (n_yr,)
            pred_proxy     = int(sims.argmax())

            proxy_correct.append(int(pred_proxy == true_year))
            proxy_abs_err.append(abs(pred_proxy - true_year))
            proxy_confusion[true_year, pred_proxy] += 1

            # ── knn-based ────────────────────────────────────────────────
            neighbours    = get_neighbours(ann, meta, emb, top_k)
            mean_year     = np.mean([n["year_idx"] for n in neighbours])
            pred_knn      = int(round(mean_year))
            # clamp to valid range
            pred_knn      = max(0, min(pred_knn, n_years - 1))

            knn_correct.append(int(pred_knn == true_year))
            knn_abs_err.append(abs(pred_knn - true_year))
            knn_confusion[true_year, pred_knn] += 1

    results = {
        "proxy_year_accuracy": float(np.mean(proxy_correct)),
        "proxy_year_mae":      float(np.mean(proxy_abs_err)),
        "knn_year_accuracy":   float(np.mean(knn_correct)),
        "knn_year_mae":        float(np.mean(knn_abs_err)),
        "proxy_year_confusion": proxy_confusion.tolist(),
        "knn_year_confusion":   knn_confusion.tolist(),
    }
    print(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Track 2 – OBJECT ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_object_estimation(
    ann: AnnoyIndex, meta: dict,
    cat_text_embs: torch.Tensor,    # (n_obj, 512)  "A photo of {cat}"
    objects: list,
    test_loader: DataLoader,
    clip_model, device, top_k: int,
):
    """
    Proxy-based : argmax CLIP similarity between image and "A photo of {cat}"
    KNN-based   : majority vote of TOP_K neighbours
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    proxy_correct, knn_correct = [], []
    proxy_confusion = np.zeros((n_obj, n_obj), dtype=np.int64)
    knn_confusion   = np.zeros((n_obj, n_obj), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Object estimation"):
        img_embs  = encode_images_batch(clip_model, image_A, device)
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(img_embs)):
            emb      = img_embs[i]
            true_cat = true_cats[i]
            true_idx = obj2idx[true_cat]

            # ── proxy-based ──────────────────────────────────────────────
            sims     = (emb.unsqueeze(0) @ cat_text_embs.T).squeeze(0)
            pred_idx = int(sims.argmax())

            proxy_correct.append(int(objects[pred_idx] == true_cat))
            proxy_confusion[true_idx, pred_idx] += 1

            # ── knn-based ────────────────────────────────────────────────
            neighbours = get_neighbours(ann, meta, emb, top_k)
            cats       = [n["category"] for n in neighbours]
            pred_knn   = Counter(cats).most_common(1)[0][0]
            pred_knn_idx = obj2idx.get(pred_knn, 0)

            knn_correct.append(int(pred_knn == true_cat))
            knn_confusion[true_idx, pred_knn_idx] += 1

    results = {
        "proxy_object_accuracy":  float(np.mean(proxy_correct)),
        "knn_object_accuracy":    float(np.mean(knn_correct)),
        "proxy_object_confusion": proxy_confusion.tolist(),
        "knn_object_confusion":   knn_confusion.tolist(),
    }
    print(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Track 3 – IMAGE + YEAR-PROXY TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_image_year_proxy_translation(
    ann: AnnoyIndex, meta: dict,
    year_text_embs: torch.Tensor,   # (n_yr, 512)
    year_labels: list,
    objects: list,
    test_loader: DataLoader,
    clip_model, device, top_k: int,
    rerank_pool: int = RERANK_POOL,
):
    """
    For every (query image, target year y):
        1. Retrieve `rerank_pool` candidates by vision similarity (Annoy).
        2. Re-rank those candidates by cosine similarity to the CLIP text
           embedding of  "A photo in YYYY"  and keep the TOP_K.
    Metrics: object accuracy, year consistency.
    """
    n_years = len(year_labels)
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_accs, year_consts = [], []

    obj_acc_sum   = np.zeros((n_obj, n_years))
    year_cons_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Image+YearProxy translation"):
        img_embs  = encode_images_batch(clip_model, image_A, device)
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(img_embs)):
            emb      = img_embs[i]
            true_cat = true_cats[i]
            cat_idx  = obj2idx[true_cat]

            # ── step 1: retrieve candidate pool by vision similarity ─────
            pool_indices, _ = ann_query_vec(ann, emb, rerank_pool)
            pool_metas  = [meta[str(idx)] for idx in pool_indices]
            # retrieve stored vision embeddings for each candidate
            pool_vecs   = torch.stack([
                torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)
                for idx in pool_indices
            ])  # (pool, 512) – already L2-normalised (stored after normalisation)
            pool_vecs = F.normalize(pool_vecs, p=2, dim=1)

            for y in range(n_years):
                # ── step 2: re-rank by  "A photo in YYYY"  ───────────────
                year_emb  = year_text_embs[y]                          # (512,)
                scores    = (pool_vecs @ year_emb).numpy()             # (pool,)
                top_k_pos = np.argsort(scores)[::-1][:top_k]
                neighbours = [pool_metas[p] for p in top_k_pos]

                obj_acc    = float(np.mean([int(n["category"] == true_cat) for n in neighbours]))
                year_const = float(np.mean([int(n["year_idx"]  == y)       for n in neighbours]))

                obj_accs.append(obj_acc)
                year_consts.append(year_const)

                obj_acc_sum[cat_idx, y]   += obj_acc
                year_cons_sum[cat_idx, y] += year_const
                count_matrix[cat_idx, y]  += 1

    safe_count      = np.where(count_matrix > 0, count_matrix, 1)
    error_matrix    = obj_acc_sum   / safe_count
    year_cons_mat   = year_cons_sum / safe_count

    results = {
        "translation_object_accuracy":     float(np.mean(obj_accs)),
        "translation_year_consistency":    float(np.mean(year_consts)),
        "translation_obj_acc_by_category": {objects[c]: float(error_matrix[c].mean())  for c in range(n_obj)},
        "translation_year_cons_by_year":   {y: float(year_cons_mat[:, y].mean())       for y in range(n_years)},
        "translation_obj_acc_matrix":      {objects[c]: error_matrix[c].tolist()       for c in range(n_obj)},
        "translation_year_cons_matrix":    {objects[c]: year_cons_mat[c].tolist()      for c in range(n_obj)},
    }
    print(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Track 4 – TWO-IMAGE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_two_image_translation(
    ann: AnnoyIndex, meta: dict,
    n_train_items: int,
    objects: list,
    year_labels: list,
    test_loader: DataLoader,
    clip_model, device, top_k: int,
    rerank_pool: int = RERANK_POOL,
):
    """
    For every query image q and a randomly sampled reference image r:
        1. Retrieve `rerank_pool` candidates by vision similarity to q (Annoy).
        2. Re-rank those candidates by cosine similarity to r's vision embedding
           and keep the TOP_K.
    Metrics: object accuracy (category of q), year consistency (year of r).
    """
    n_years = len(year_labels)
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_accs, year_consts = [], []
    year_mae_two_images   = []

    obj_acc_sum   = np.zeros((n_obj, n_years))
    year_cons_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    for image_A, condition_A, image_B, condition_B, category in tqdm(
            test_loader, desc="Two-image translation"):
        img_embs  = encode_images_batch(clip_model, image_A, device)
        true_cats = list(category) if not isinstance(category, list) else category

        for i in range(len(img_embs)):
            emb      = img_embs[i]
            true_cat = true_cats[i]
            cat_idx  = obj2idx[true_cat]

            # ── sample a random reference from the train Annoy index ─────
            ref_idx  = random.randrange(n_train_items)
            ref_emb  = torch.tensor(ann.get_item_vector(ref_idx), dtype=torch.float32)
            ref_meta = meta[str(ref_idx)]

            ref_year = ref_meta["year_idx"]

            ref_emb  = F.normalize(ref_emb, p=2, dim=0)

            # ── step 1: candidate pool by vision similarity to query ──────
            pool_indices, _ = ann_query_vec(ann, emb, rerank_pool)
            pool_metas = [meta[str(idx)] for idx in pool_indices]
            pool_vecs  = torch.stack([
                torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)
                for idx in pool_indices
            ])
            pool_vecs = F.normalize(pool_vecs, p=2, dim=1)

            # ── step 2: re-rank by similarity to reference image ──────────
            scores    = (pool_vecs @ ref_emb).numpy()          # (pool,)
            top_k_pos = np.argsort(scores)[::-1][:top_k]
            neighbours = [pool_metas[p] for p in top_k_pos]

            obj_acc    = float(np.mean([int(n["category"] == true_cat) for n in neighbours]))
            year_const = float(np.mean([int(n["year_idx"]  == ref_year) for n in neighbours]))

            mean_year     = np.mean([n["year_idx"] for n in neighbours])
            pred_year_knn = int(round(mean_year))
            year_mae_two_images.append(abs(pred_year_knn - ref_year))

            obj_accs.append(obj_acc)
            year_consts.append(year_const)

            obj_acc_sum[cat_idx, ref_year]   += obj_acc
            year_cons_sum[cat_idx, ref_year] += year_const
            count_matrix[cat_idx, ref_year]  += 1

    safe_count      = np.where(count_matrix > 0, count_matrix, 1)
    obj_acc_mat     = obj_acc_sum   / safe_count
    year_cons_mat   = year_cons_sum / safe_count

    results = {
        "two_image_object_accuracy":        float(np.mean(obj_accs)),
        "two_image_year_consistency":       float(np.mean(year_consts)),
        "two_images_year_mae":              float(np.mean(year_mae_two_images)),
        "two_image_obj_acc_by_category":    {objects[c]: float(obj_acc_mat[c].mean())   for c in range(n_obj)},
        "two_image_year_cons_by_ref_year":  {y: float(year_cons_mat[:, y].mean())       for y in range(n_years)},
        "two_image_obj_acc_matrix":         {objects[c]: obj_acc_mat[c].tolist()        for c in range(n_obj)},
        "two_image_year_cons_matrix":       {objects[c]: year_cons_mat[c].tolist()      for c in range(n_obj)},
    }
    print(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers  (identical API to evaluate_cir.py)
# ─────────────────────────────────────────────────────────────────────────────

def _print_scalars(d: dict):
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<50s}: {v:.4f}")


def _print_confusion(matrix: list, labels: list, title: str):
    n     = len(matrix)
    col_w = max(len(str(l)) for l in labels) + 2
    head  = " " * (col_w + 2) + "".join(f"{str(l):>{col_w}}" for l in labels)
    print(f"\n  {title}")
    print("  " + head)
    for i, row in enumerate(matrix):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {str(labels[i]):<{col_w}}  {row_str}")


def _print_matrix(matrix: dict, col_labels: list, title: str, fmt: str = ".3f"):
    col_w = 8
    head  = " " * 22 + "".join(f"{str(l):>{col_w}}" for l in col_labels)
    print(f"\n  {title}")
    print("  " + head)
    for row_label, row in matrix.items():
        row_str = "".join(f"{v:{col_w}{fmt}}" for v in row)
        print(f"  {str(row_label):<22}{row_str}")


def _print_section(results: dict, year_labels: list, obj_labels: list, section: str):
    _print_scalars(results)

    if section == "date":
        for key, title in [
            ("proxy_year_confusion", "Proxy year confusion (row=true, col=pred)"),
            ("knn_year_confusion",   "KNN   year confusion (row=true, col=pred)"),
        ]:
            if key in results:
                _print_confusion(results[key], list(range(len(year_labels))), title)

    elif section == "object":
        for key, title in [
            ("proxy_object_confusion", "Proxy object confusion (row=true, col=pred)"),
            ("knn_object_confusion",   "KNN   object confusion (row=true, col=pred)"),
        ]:
            if key in results:
                _print_confusion(results[key], obj_labels, title)

    elif section in ("translation", "two_image"):
        prefix    = "translation" if section == "translation" else "two_image"
        ref_label = "target year" if section == "translation" else "ref year"

        by_cat_key = f"{prefix}_obj_acc_by_category"
        if by_cat_key in results:
            print("\n  Object accuracy by query category:")
            for cat, v in results[by_cat_key].items():
                print(f"    {cat:<20s}: {v:.4f}")

        by_yr_key = (f"{prefix}_year_cons_by_year"
                     if section == "translation"
                     else f"{prefix}_year_cons_by_ref_year")
        if by_yr_key in results:
            print(f"\n  Year consistency by {ref_label}:")
            for y, v in results[by_yr_key].items():
                print(f"    year_idx {y:<4}: {v:.4f}  (= {year_labels[int(y)]})")

        obj_mat_key  = f"{prefix}_obj_acc_matrix"
        year_mat_key = f"{prefix}_year_cons_matrix"
        if obj_mat_key in results:
            _print_matrix(results[obj_mat_key],  list(range(len(year_labels))),
                          f"Object accuracy matrix [category × {ref_label}]")
        if year_mat_key in results:
            _print_matrix(results[year_mat_key], list(range(len(year_labels))),
                          f"Year consistency matrix [category × {ref_label}]")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLIP baseline – CIR offline evaluation")
    p.add_argument("--objects",      nargs="+", required=True)
    p.add_argument("--ckpt_folder",  required=True,
                   help="Directory used for caching the Annoy index and results.")
    p.add_argument("--top_k",        type=int, default=TOP_K)
    p.add_argument("--rerank_pool",  type=int, default=RERANK_POOL,
                   help="Candidate pool size before re-ranking (tracks 3 & 4).")
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
    print(f"Device     : {device}")
    print(f"CLIP model : {OPENCLIP_MODEL} / {OPENCLIP_PRETRAIN}")
    print(f"TOP_K      : {args.top_k}")
    print(f"Rerank pool: {args.rerank_pool}")

    # ── build year / category label lists ────────────────────────────────────
    year_labels = build_year_labels(args.min_date, args.max_date, args.freq)
    n_years     = len(year_labels)
    objects     = args.objects
    print(f"Year labels: {year_labels[0]} … {year_labels[-1]}  ({n_years} bins)")
    print(f"Objects    : {objects}")

    # ── load CLIP ─────────────────────────────────────────────────────────────
    clip_model, preprocess, tokenizer = load_clip(device)

    # ── pre-compute all text embeddings ──────────────────────────────────────
    print("\nEncoding year text prompts …")
    year_prompts     = [year_to_prompt(y) for y in year_labels]
    year_text_embs   = encode_text(clip_model, year_prompts, device, tokenizer)        # (n_yr, D)

    print("Encoding category text prompts …")
    cat_prompts      = [category_to_prompt(o) for o in objects]
    cat_text_embs    = encode_text(clip_model, cat_prompts, device, tokenizer)         # (n_obj, D)

    # ── build / load CLIP Annoy index ─────────────────────────────────────────
    avlabels = build_avlabels(objects, args.min_date, args.max_date, args.freq)
    ann, meta = build_clip_annoy_index(
        objects, args.ckpt_folder, avlabels,
        clip_model, preprocess, tokenizer, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        rebuild=args.rebuild,
    )
    n_train_items = len(meta)
    print(f"\n  Annoy index size: {n_train_items} train items")

    # ── build test dataloader ─────────────────────────────────────────────────
    test_loader = build_loader(data_test, objects, avlabels, preprocess,
                               args.batch_size, args.num_workers, evaluate=True)
    print(f"  Test dataset size: {len(test_loader.dataset)} items")

    # ── Track 1: Date estimation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1 / 4  DATE ESTIMATION")
    print("=" * 60)
    date_results = eval_date_estimation(
        ann, meta, year_text_embs, year_labels,
        test_loader, clip_model, device, args.top_k,
    )
    _print_section(date_results, year_labels, objects, "date")

    # ── Track 2: Object estimation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2 / 4  OBJECT ESTIMATION")
    print("=" * 60)
    obj_results = eval_object_estimation(
        ann, meta, cat_text_embs, objects,
        test_loader, clip_model, device, args.top_k,
    )
    _print_section(obj_results, year_labels, objects, "object")

    # ── Track 3: Image + year-proxy translation ───────────────────────────────
    print("\n" + "=" * 60)
    print("3 / 4  IMAGE + YEAR-PROXY TRANSLATION")
    print("=" * 60)
    trans_results = eval_image_year_proxy_translation(
        ann, meta, year_text_embs, year_labels, objects,
        test_loader, clip_model, device, args.top_k, args.rerank_pool,
    )
    _print_section(trans_results, year_labels, objects, "translation")

    # ── Track 4: Two-image translation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("4 / 4  TWO-IMAGE TRANSLATION")
    print("=" * 60)
    print(year_labels)
    two_img_results = eval_two_image_translation(
        ann, meta, n_train_items, objects, year_labels,
        test_loader, clip_model, device, args.top_k, args.rerank_pool,
    )
    _print_section(two_img_results, year_labels, objects, "two_image")

    # ── save results ──────────────────────────────────────────────────────────
    all_results  = {**date_results, **obj_results, **trans_results, **two_img_results}
    summary_path = os.path.join(
        args.ckpt_folder,
        f"clip_eval_topk{args.top_k}_pool{args.rerank_pool}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔  Full results saved to {summary_path}")

    # ── scalar summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALAR SUMMARY")
    print("=" * 60)
    _print_scalars(all_results)


if __name__ == "__main__":
    main()