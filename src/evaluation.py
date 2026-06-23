"""
evaluate_cir.py

Offline evaluation of a CIR (Composed Image Retrieval) checkpoint.

The Annoy index / meta JSON are built from the TRAIN split and stored as
  <ckpt_folder>/qualitative_eval_<epoch>.ann
  <ckpt_folder>/qualitative_eval_meta_<epoch>.json

Queries come from the TEST split.

─────────────────────────────────────────
Metrics computed  (paper §Evaluation Protocol)
─────────────────────────────────────────

All retrieval metrics are computed at rank K.

1. LABEL + LABEL  (query = μ^c + μ^y)
   - Object Precision@K : fraction of top-K whose category == c
   - Date   Precision@K : fraction of top-K whose year_idx  == y
   Averaged over all (c, y) pairs present in the test set.

2. IMAGE + LABEL  (query = f(x_c) + μ^y)
   - Object Precision@K : fraction of top-K whose category == c(x_c)
   - Date   Precision@K : fraction of top-K whose year_idx  == y
   Averaged over all (x_c, y) pairs.

3. IMAGE + IMAGE  (query = f(x_c) + r_y,  r_y = f(x_y) − μ^ĉ)
   Reference image x_y is sampled uniformly from the train set.
   - Object Precision@K : fraction of top-K whose category == c(x_c)
   - Date   Precision@K : fraction of top-K whose year_idx  == year(x_y)
   Averaged over all query images.

Usage
-----
CUDA_VISIBLE_DEVICES=2 python evaluate_cir.py \
    --objects Window Man Building ... \
    --epoch 5 \
    --ckpt_folder /data/.../0be4e306/ \
    --rebuild
"""

import argparse
import json
import os
import random
from itertools import product as iterproduct

import numpy as np
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
from core_datautils import df as data_complete
from core_datautils import df_test as data_test
from models import SpecialistModel, ConditionedToYear, ViTSpecialistModel, VGGSpecialistModel, ResNetSpecialistModel
from train_experts_dataloader import SpecialistDataloaderWithClass
from torchvision import transforms

# ── constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1024
N_TREES       = 50
TOP_K         = 10

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
IMAGENET_TRANSFORMS_VAL = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def latest_epoch(ckpt_folder: str) -> int:
    import re
    epochs = [
        int(m.group(1))
        for f in os.listdir(ckpt_folder)
        if (m := re.match(r"model_epoch(\d+)\.pth", f))
    ]
    if not epochs:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_folder}")
    return max(epochs)

def load_model(dataset, args, device):
    if args.model == 'convnext':
        date_sensitive = SpecialistModel(len(dataset.available_labels)).to(device)
    elif args.model == 'vit':
        date_sensitive = ViTSpecialistModel(len(dataset.available_labels)).to(device)

    elif args.model == 'vgg':
        date_sensitive = VGGSpecialistModel(len(dataset.available_labels)).to(device)
    elif args.model == 'resnet':
        date_sensitive = ResNetSpecialistModel(len(dataset.available_labels)).to(device)

    else:
        raise NotImplementedError(f'Model {args.model} Not Implemented')

    model = ConditionedToYear(date_sensitive, output_dim=len(dataset.available_labels), origin_model=args.model).to(device)
    return model
def load_everything(ckpt_folder: str, objects: list, epoch: int, device, args):
    """Load model + loss-function state-dicts from checkpoint."""
    dataset0 = SpecialistDataloaderWithClass(
        data_complete, objects[0], transforms=IMAGENET_TRANSFORMS_VAL
    )
    avlabels = dataset0.available_labels
    n_years  = len(avlabels)

    model = load_model(dataset0, args, device)
    model.load_state_dict(
        torch.load(f"{ckpt_folder}/model_epoch{epoch}.pth", map_location=device)
    )
    model.eval()

    contrastive_loss_fn = losses.ProxyNCALoss(50, EMBEDDING_DIM, softmax_scale=1)
    contrastive_loss_fn.load_state_dict(
        torch.load(f"{ckpt_folder}/contrastive_loss_epoch{epoch}.pth", map_location=device)
    )

    year_loss_fn = losses.ProxyNCALoss(n_years, EMBEDDING_DIM, softmax_scale=1)
    year_loss_fn.load_state_dict(
        torch.load(f"{ckpt_folder}/year_loss_epoch{epoch}.pth", map_location=device)
    )

    return model, contrastive_loss_fn, year_loss_fn, avlabels


# ─────────────────────────────────────────────────────────────────────────────
# Annoy index  (built from TRAIN split)
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_annoy_index(
    objects: list,
    ckpt_folder: str,
    epoch: int,
    device,
    batch_size: int = 32,
    num_workers: int = 4,
    rebuild: bool = False,
    args = None
):
    """
    Build (or load) an Annoy index over the TRAIN split.
    Meta entries: { category, year_idx, norm }
    Embeddings stored un-normalised via the norm field so we can recover the
    original vector when sampling reference images for Image+Image retrieval.
    """
    annoy_path = os.path.join(ckpt_folder, f"qualitative_eval_{epoch}.ann")
    meta_path  = os.path.join(ckpt_folder, f"qualitative_eval_meta_{epoch}.json")

    if not rebuild and os.path.exists(annoy_path) and os.path.exists(meta_path):
        print("✔  Eval Annoy index already exists – loading.")
        ann = AnnoyIndex(EMBEDDING_DIM, "euclidean")
        ann.load(annoy_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return ann, meta

    print("Building eval Annoy index from TRAIN split…")
    model, contrastive_loss_fn, year_loss_fn, avlabels = load_everything(
        ckpt_folder, objects, epoch, device, args
    )

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

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)

    ann        = AnnoyIndex(EMBEDDING_DIM, "euclidean")
    meta       = {}
    global_idx = 0

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(loader, desc="Indexing train"):
            image_A = image_A.to(device)

            emb_raw, _ = model(image_A, None)          # un-normalised
            norms      = emb_raw.norm(p=2, dim=1)      # (B,)
            emb_norm   = F.normalize(emb_raw, p=2, dim=1)
            emb_np     = emb_norm.cpu().numpy()
            norms_np   = norms.cpu().numpy()

            for i in range(len(emb_np)):
                ann.add_item(global_idx, emb_np[i].tolist())
                cat = category[i] if isinstance(category[i], str) else category[i]
                meta[str(global_idx)] = {
                    "category": cat,
                    "year_idx": int(condition_B[i].item()),
                    "norm":     float(norms_np[i]),
                }
                global_idx += 1

    ann.build(N_TREES)
    ann.save(annoy_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"✔  Eval Annoy index built: {global_idx} train items → {annoy_path}")
    return ann, meta


# ─────────────────────────────────────────────────────────────────────────────
# Low-level query helpers
# ─────────────────────────────────────────────────────────────────────────────

def ann_query(ann: AnnoyIndex, query_vec: torch.Tensor, k: int):
    """L2-normalise query_vec, return k nearest neighbours."""
    qv = F.normalize(query_vec, p=2, dim=0).numpy().tolist()
    indices, distances = ann.get_nns_by_vector(qv, k, include_distances=True)
    return indices, distances


def get_neighbours(ann: AnnoyIndex, meta: dict, query_vec: torch.Tensor, k: int):
    indices, _ = ann_query(ann, query_vec, k)
    return [meta[str(idx)] for idx in indices]


def embed_batch(images: torch.Tensor, conditions: torch.Tensor,
                model, device, normalize: bool = True) -> torch.Tensor:
    images     = images.to(device)
    conditions = conditions.to(device)
    with torch.no_grad():
        emb, _ = model(images, conditions)
    if normalize:
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu()


def precision_at_k(neighbours: list, key: str, target_value) -> float:
    """Fraction of retrieved neighbours whose meta[key] == target_value."""
    return float(np.mean([int(n[key] == target_value) for n in neighbours]))


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL + LABEL   query = μ^c + μ^y
# ─────────────────────────────────────────────────────────────────────────────

def eval_label_label(
    ann: AnnoyIndex, meta: dict,
    cat_proxies: torch.Tensor,   # (n_obj, D)
    year_proxies: torch.Tensor,  # (n_yr,  D)
    objects: list,
    n_years: int,
    top_k: int,
):
    """
    For every (category c, year y) pair, issue query μ^c + μ^y and compute
    Object Precision@K and Date Precision@K.

    All (c, y) combinations are evaluated — the composed query does not require
    any actual image, only the two proxy vectors.
    """
    n_obj = len(objects)

    obj_prec_sum  = np.zeros((n_obj, n_years))
    date_prec_sum = np.zeros((n_obj, n_years))

    for c_idx, obj in enumerate(tqdm(objects, desc="Label+Label")):
        mu_c = cat_proxies[c_idx]                         # (D,)
        for y in range(n_years):
            mu_y      = year_proxies[y]                   # (D,)
            query_vec = mu_c + mu_y                       # (D,)

            neighbours = get_neighbours(ann, meta, query_vec, top_k)

            obj_prec_sum[c_idx, y]  = precision_at_k(neighbours, "category", obj)
            date_prec_sum[c_idx, y] = precision_at_k(neighbours, "year_idx", y)

    obj_prec_mean  = float(obj_prec_sum.mean())
    date_prec_mean = float(date_prec_sum.mean())

    results = {
        "label_label_object_precision_at_k":  obj_prec_mean,
        "label_label_date_precision_at_k":    date_prec_mean,
        # fine-grained matrices  [category][year] for deeper analysis
        "label_label_obj_prec_matrix":  {objects[c]: obj_prec_sum[c].tolist()  for c in range(n_obj)},
        "label_label_date_prec_matrix": {objects[c]: date_prec_sum[c].tolist() for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE + LABEL   query = f(x_c) + μ^y
# ─────────────────────────────────────────────────────────────────────────────

def eval_image_label(
    ann: AnnoyIndex, meta: dict,
    year_proxies: torch.Tensor,   # (n_yr, D)
    test_loader: DataLoader,
    model, device,
    objects: list,
    n_years: int,
    top_k: int,
):
    """
    For every test image x_c and every admissible year y:
        query = f(x_c) + μ^y
    Retrieve top-K and compute:
        Object Precision@K  – category of x_c preserved
        Date   Precision@K  – year_idx == y
    Averaged over all (x_c, y) pairs.
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_prec_sum  = np.zeros((n_obj, n_years))
    date_prec_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(
                test_loader, desc="Image+Label"):
            # un-normalised embeddings so we can add year proxy in embedding space
            embs      = embed_batch(image_A, condition_A, model, device, normalize=False)
            true_cats = list(category) if not isinstance(category, list) else category

            for i in range(len(embs)):
                emb      = embs[i]                        # (D,)  un-normalised
                true_cat = true_cats[i]
                cat_idx  = obj2idx[true_cat]

                for y in range(n_years):
                    query_vec  = emb + year_proxies[y]    # (D,)
                    neighbours = get_neighbours(ann, meta, query_vec, top_k)

                    obj_prec_sum[cat_idx, y]  += precision_at_k(neighbours, "category", true_cat)
                    date_prec_sum[cat_idx, y] += precision_at_k(neighbours, "year_idx", y)
                    count_matrix[cat_idx, y]  += 1

    safe = np.where(count_matrix > 0, count_matrix, 1)
    obj_mat  = obj_prec_sum  / safe
    date_mat = date_prec_sum / safe

    results = {
        "image_label_object_precision_at_k": float(obj_mat.mean()),
        "image_label_date_precision_at_k":   float(date_mat.mean()),
        # fine-grained breakdowns
        "image_label_obj_prec_by_category":  {objects[c]: float(obj_mat[c].mean())  for c in range(n_obj)},
        "image_label_date_prec_by_year":     {y: float(date_mat[:, y].mean())       for y in range(n_years)},
        "image_label_obj_prec_matrix":       {objects[c]: obj_mat[c].tolist()       for c in range(n_obj)},
        "image_label_date_prec_matrix":      {objects[c]: date_mat[c].tolist()      for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE + IMAGE   query = f(x_c) + r_y,  r_y = f(x_y) − μ^ĉ
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_train_ref(ann: AnnoyIndex, meta: dict, n_items: int):
    """
    Sample a random train embedding from the Annoy index and recover the
    un-normalised vector using the stored norm.
    Returns (ref_emb, ref_category, ref_year_idx).
    """
    idx     = random.randrange(n_items)
    m       = meta[str(idx)]
    ref_emb = torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)
    # Annoy stores the L2-normalised vector; restore original magnitude.
    ref_emb = ref_emb * m["norm"]
    return ref_emb, m["category"], m["year_idx"]


def eval_image_image(
    ann: AnnoyIndex, meta: dict,
    cat_proxies: torch.Tensor,    # (n_obj, D)
    n_train_items: int,
    test_loader: DataLoader,
    model, device,
    objects: list,
    n_years: int,
    top_k: int,
):
    """
    For every test image x_c, sample a random train image x_y as temporal
    reference. Build:
        ĉ         = argmin_c  ‖f(x_y) − μ^c‖
        r_y       = f(x_y) − μ^ĉ
        query_vec = f(x_c) + r_y
    Retrieve top-K and compute:
        Object Precision@K  – category of x_c preserved
        Date   Precision@K  – year_idx == year(x_y)
    Averaged over all query images.
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_prec_sum  = np.zeros((n_obj, n_years))
    date_prec_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    all_obj_prec  = []
    all_date_prec = []

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(
                test_loader, desc="Image+Image"):
            embs      = embed_batch(image_A, condition_A, model, device, normalize=False)
            true_cats = list(category) if not isinstance(category, list) else category

            for i in range(len(embs)):
                cat_emb  = embs[i]                        # (D,) un-normalised
                true_cat = true_cats[i]
                cat_idx  = obj2idx[true_cat]

                # ── temporal reference ───────────────────────────────────
                ref_emb, ref_cat, ref_year = sample_random_train_ref(
                    ann, meta, n_train_items
                )

                # infer category of reference image via nearest proxy
                dists_cat = torch.cdist(
                    ref_emb.unsqueeze(0), cat_proxies
                ).squeeze(0)                              # (n_obj,)
                closest_cat_idx = int(dists_cat.argmin())

                # extract temporal residual (strip category from ref embedding)
                r_y = ref_emb - cat_proxies[closest_cat_idx]  # (D,)

                # ── compose and retrieve ─────────────────────────────────
                query_vec  = cat_emb + r_y                # (D,)
                neighbours = get_neighbours(ann, meta, query_vec, top_k)

                op = precision_at_k(neighbours, "category", true_cat)
                dp = precision_at_k(neighbours, "year_idx", ref_year)

                all_obj_prec.append(op)
                all_date_prec.append(dp)

                obj_prec_sum[cat_idx, ref_year]  += op
                date_prec_sum[cat_idx, ref_year] += dp
                count_matrix[cat_idx, ref_year]  += 1

    safe = np.where(count_matrix > 0, count_matrix, 1)
    obj_mat  = obj_prec_sum  / safe
    date_mat = date_prec_sum / safe

    results = {
        "image_image_object_precision_at_k": float(np.mean(all_obj_prec)),
        "image_image_date_precision_at_k":   float(np.mean(all_date_prec)),
        # fine-grained breakdowns  (query_category  ×  ref_year)
        "image_image_obj_prec_by_category":  {objects[c]: float(obj_mat[c].mean())  for c in range(n_obj)},
        "image_image_date_prec_by_ref_year": {y: float(date_mat[:, y].mean())       for y in range(n_years)},
        "image_image_obj_prec_matrix":       {objects[c]: obj_mat[c].tolist()       for c in range(n_obj)},
        "image_image_date_prec_matrix":      {objects[c]: date_mat[c].tolist()      for c in range(n_obj)},
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test dataloader
# ─────────────────────────────────────────────────────────────────────────────

def build_test_loader(objects: list, avlabels, batch_size: int, num_workers: int):
    dataset = SpecialistDataloaderWithClass(
        data_test, objects[0], transforms=IMAGENET_TRANSFORMS_VAL, evaluate=True
    )
    dataset.available_labels = avlabels
    for obj in objects[1:]:
        extra = SpecialistDataloaderWithClass(
            data_test, obj, transforms=IMAGENET_TRANSFORMS_VAL, evaluate=True
        )
        dataset = dataset + extra
        dataset.available_labels = avlabels
    print(f"Test dataset size: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
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
    p = argparse.ArgumentParser(description="CIR offline evaluation")
    p.add_argument("--objects",     nargs="+", required=True)
    p.add_argument("--ckpt_folder", required=True)
    p.add_argument("--epoch",       type=int,  default=None)
    p.add_argument("--top_k",       type=int,  default=TOP_K)
    p.add_argument("--batch_size",  type=int,  default=32)
    p.add_argument("--num_workers", type=int,  default=16)
    p.add_argument("--rebuild",     action="store_true",
                   help="Force rebuild of the eval Annoy index.")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--model", type=str, default='convnext', choices=['convnext', 'vgg', 'vit', 'resnet'])

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    epoch = args.epoch if args.epoch is not None else latest_epoch(args.ckpt_folder)
    print(f"Epoch  : {epoch}")
    print(f"TOP_K  : {args.top_k}")

    # ── load model + proxies ──────────────────────────────────────────────────
    model, contrastive_loss_fn, year_loss_fn, avlabels = load_everything(
        args.ckpt_folder, args.objects, epoch, device, args
    )
    cat_proxies  = contrastive_loss_fn.proxies.detach().cpu()   # (n_obj, D)
    year_proxies = year_loss_fn.proxies.detach().cpu()          # (n_yr,  D)
    n_years      = year_proxies.shape[0]
    year_labels  = list(range(n_years))

    print(f"  Category proxies : {cat_proxies.shape}")
    print(f"  Year proxies     : {year_proxies.shape}")

    # ── build / load eval Annoy index ─────────────────────────────────────────
    ann, meta = build_eval_annoy_index(
        objects=args.objects, ckpt_folder=args.ckpt_folder, epoch=epoch,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        rebuild=args.rebuild, args = args
    )
    n_train_items = len(meta)
    print(f"  Annoy index size : {n_train_items} train items")

    # ── build test dataloader ─────────────────────────────────────────────────
    test_loader = build_test_loader(
        args.objects, avlabels, args.batch_size, args.num_workers
    )

    # ── 1. LABEL + LABEL ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1 / 3  LABEL + LABEL   (query = μ^c + μ^y)")
    print("=" * 60)
    ll_results = eval_label_label(
        ann, meta, cat_proxies, year_proxies,
        args.objects, n_years, args.top_k,
    )
    print_scalars(ll_results)
    print_matrix(ll_results["label_label_obj_prec_matrix"],  year_labels,
                 "Object Precision@K matrix [category × year]")
    print_matrix(ll_results["label_label_date_prec_matrix"], year_labels,
                 "Date   Precision@K matrix [category × year]")

    # ── 2. IMAGE + LABEL ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2 / 3  IMAGE + LABEL   (query = f(x_c) + μ^y)")
    print("=" * 60)
    il_results = eval_image_label(
        ann, meta, year_proxies, test_loader, model, device,
        args.objects, n_years, args.top_k,
    )
    print_scalars(il_results)
    print_matrix(il_results["image_label_obj_prec_matrix"],  year_labels,
                 "Object Precision@K matrix [category × target year]")
    print_matrix(il_results["image_label_date_prec_matrix"], year_labels,
                 "Date   Precision@K matrix [category × target year]")

    # ── 3. IMAGE + IMAGE ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3 / 3  IMAGE + IMAGE   (query = f(x_c) + r_y)")
    print("=" * 60)
    ii_results = eval_image_image(
        ann, meta, cat_proxies, n_train_items, test_loader, model, device,
        args.objects, n_years, args.top_k,
    )
    print_scalars(ii_results)
    print_matrix(ii_results["image_image_obj_prec_matrix"],  year_labels,
                 "Object Precision@K matrix [query category × ref year]")
    print_matrix(ii_results["image_image_date_prec_matrix"], year_labels,
                 "Date   Precision@K matrix [query category × ref year]")

    # ── scalar summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALAR SUMMARY")
    print("=" * 60)
    all_results = {**ll_results, **il_results, **ii_results}
    print_scalars(all_results)

    # ── save ──────────────────────────────────────────────────────────────────
    summary_path = os.path.join(
        args.ckpt_folder, f"eval_epoch{epoch}_topk{args.top_k}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔  Full results saved to {summary_path}")


if __name__ == "__main__":
    main()