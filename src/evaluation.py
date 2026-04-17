"""
evaluate_cir.py


CUDA_VISIBLE_DEVICES=2 python evaluation.py --objects  Window Man Building House Train Shirt Boy Tower Door "Street light" Trousers Tree "Vehicle registration plate" Dress Woman Skirt "Human head" Poster Tie Car Girl Belt Coat Jeans Tire Billboard Castle Shorts Wheel "Parking meter" Hat Glasses Bus Watch Footwear "Human face" Suit Taxi Ladder "Human ear" "Traffic sign" Jacket Person Stairs "Traffic light" Bench Sock Van Curtain Boat  --epoch 5 --ckpt_folder /data/113-2/users/amolina/cir_date/0be4e306/ --rebuild

---------------
Offline evaluation of a CIR (Composed Image Retrieval) checkpoint.

The Annoy index / meta JSON are built from the TRAIN split (df / data_complete)
and are stored as  <ckpt_folder>/qualitative_eval.ann  and
<ckpt_folder>/qualitative_eval_meta.json   (separate from the qualitative.py ones
so the two tools can co-exist).

Queries come from the TEST split (df_test).

─────────────────────────────────────────
Metrics computed
─────────────────────────────────────────

1. DATE ESTIMATION
   a) Proxy-based  : subtract closest category proxy → find closest year proxy
   b) KNN-based    : average year_idx of TOP_K nearest neighbours
   Reported as:  Accuracy (exact year match)  and  MAE (|true − pred| in year-index units)

2. OBJECT ESTIMATION
   a) Proxy-based  : find the category proxy closest to the raw embedding
   b) KNN-based    : majority vote over TOP_K nearest neighbours
   Reported as:  Accuracy

3. IMAGE + YEAR-PROXY TRANSLATION  (= "query by example" from the Flask demo)
   For every query image, translate to EVERY year index by adding year_proxy[y].
   For each (query, target_year) pair retrieve TOP_K images and measure:
   a) Object accuracy   : fraction of retrieved images whose category == query category
   b) Year consistency  : fraction of retrieved images whose year_idx == target_year

4. TWO-IMAGE TRANSLATION
   For every query image, pick a RANDOM train image as the "year reference".
   Build query = cat_emb(query) + (emb(ref) − closest_cat_proxy(emb(ref))).
   Retrieve TOP_K and measure:
   a) Object accuracy   : fraction retrieved with same category as query
   b) Year consistency  : fraction retrieved with same year_idx as reference image

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
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
from core_datautils import df as data_complete          # train split  → index
from core_datautils import df_test as data_test         # test  split  → queries
from models import ConditionedToYear, SpecialistModel
from train_experts_dataloader import SpecialistDataloaderWithClass
from torchvision import transforms

# ── constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1024
N_TREES       = 50       # Annoy trees  (more → better recall)
TOP_K         = 10       # global default, overridden by --top_k

# Standard ImageNet normalization
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


def load_everything(ckpt_folder: str, objects: list, epoch: int, device):
    """Load model + loss-function state-dicts from checkpoint."""
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
# Annoy index builder  (no b64 thumbnails – lightweight eval variant)
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_annoy_index(
    objects: list,
    ckpt_folder: str,
    epoch: int,
    device,
    batch_size: int = 32,
    num_workers: int = 4,
    rebuild: bool = False,
):
    """
    Build (or load) an Annoy index over the TRAIN split.
    Stored as qualitative_eval.ann / qualitative_eval_meta.json.
    Meta entries are lightweight: { category, year_idx } – no b64 image bytes.
    """
    annoy_path = os.path.join(ckpt_folder, "qualitative_eval.ann")
    meta_path  = os.path.join(ckpt_folder, "qualitative_eval_meta.json")

    if not rebuild and os.path.exists(annoy_path) and os.path.exists(meta_path):
        print("✔  Eval Annoy index already exists – loading.")
        ann = AnnoyIndex(EMBEDDING_DIM, "euclidean")
        ann.load(annoy_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return ann, meta

    print("Building eval Annoy index from TRAIN split…")
    model, contrastive_loss_fn, year_loss_fn, avlabels = load_everything(
        ckpt_folder, objects, epoch, device
    )

    # Merge all object datasets from the TRAIN split
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

    ann  = AnnoyIndex(EMBEDDING_DIM, "euclidean")
    meta = {}      # str(global_idx) → { category, year_idx }
    global_idx = 0

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(loader, desc="Indexing train"):
            image_A     = image_A.to(device)

            emb_agnostic, _ = model(image_A, None)
            norms            = emb_agnostic.norm(p=2, dim=1)          # (B,)
            emb_agnostic     = F.normalize(emb_agnostic, p=2, dim=1)
            emb_np           = emb_agnostic.cpu().numpy()
            norms_np         = norms.cpu().numpy()

            for i in range(len(emb_np)):
                ann.add_item(global_idx, emb_np[i].tolist())
                cat = category[i] if isinstance(category[i], str) else category[i]
                meta[str(global_idx)] = {
                    "category": cat,
                    "year_idx": int(condition_B[i].item()),
                    "norm":     float(norms_np[i]),                    # ← new
                }
                global_idx += 1
            
            # if len(meta) > 1000: break # TODO: ELiminar aixo a la versio final

    ann.build(N_TREES)
    ann.save(annoy_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"✔  Eval Annoy index built: {global_idx} train items → {annoy_path}")
    return ann, meta


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

def embed_batch(images: torch.Tensor, conditions: torch.Tensor,
                model, device, normalize: bool = True) -> torch.Tensor:
    """
    Forward a batch through the model.
    Returns (B, 1024) embeddings, optionally L2-normalised.
    """
    images     = images.to(device)
    conditions = conditions.to(device)
    with torch.no_grad():
        emb_agnostic, _ = model(images, conditions)
    if normalize:
        emb_agnostic = F.normalize(emb_agnostic, p=2, dim=1)
    return emb_agnostic.cpu()


def ann_query(ann: AnnoyIndex, query_vec: torch.Tensor, k: int):
    """
    query_vec: (1024,) tensor – will be L2-normalised before the query.
    Returns (indices, distances).
    """
    qv = F.normalize(query_vec, p=2, dim=0).numpy().tolist()
    indices, distances = ann.get_nns_by_vector(qv, k, include_distances=True)
    return indices, distances


def get_neighbours(ann: AnnoyIndex, meta: dict,
                   query_vec: torch.Tensor, k: int):
    """Return list of meta-dicts for the k nearest neighbours."""
    indices, _ = ann_query(ann, query_vec, k)
    return [meta[str(idx)] for idx in indices]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ROUTINES
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. DATE ESTIMATION ───────────────────────────────────────────────────────

def eval_date_estimation(
    ann: AnnoyIndex, meta: dict,
    cat_proxies: torch.Tensor,   # (n_obj, D)
    year_proxies: torch.Tensor,  # (n_yr,  D)
    test_loader: DataLoader,
    model, device, top_k: int,
    obj2idx: dict,
):
    """
    For each test image:
      - Embed with CNN (condition = true year).
      - Proxy-based year prediction:
          residual = emb − closest_cat_proxy
          pred_year = argmin_y  dist(residual, year_proxy[y])
      - KNN-based year prediction:
          mean( year_idx of TOP_K neighbours )  (rounded to nearest int)

    Returns:
      - Global accuracy + MAE for both methods.
      - Confusion matrices  proxy_confusion[true_y][pred_y]  and
        knn_confusion[true_y][pred_y]  (counts; normalise row-wise for recall).
    """
    n_years = year_proxies.shape[0]

    proxy_correct, proxy_abs_err = [], []
    knn_correct,   knn_abs_err   = [], []

    # confusion[true_year][pred_year] = count
    proxy_confusion = np.zeros((n_years, n_years), dtype=np.int64)
    knn_confusion   = np.zeros((n_years, n_years), dtype=np.int64)

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(test_loader, desc="Standalone date estimation"):
            embs = embed_batch(image_A, condition_A, model, device, normalize=False)
            true_years = condition_B.numpy()

            for i in range(len(embs)):
                emb       = embs[i]
                true_year = int(true_years[i])

                # ── proxy-based ──────────────────────────────────────────
                dists_cat       = torch.cdist(emb.unsqueeze(0), cat_proxies).squeeze(0)
                closest_cat_idx = int(dists_cat.argmin())
                residual        = emb - cat_proxies[closest_cat_idx]

                dists_year      = torch.cdist(residual.unsqueeze(0), year_proxies).squeeze(0)
                pred_year_proxy = int(dists_year.argmin())

                proxy_correct.append(int(pred_year_proxy == true_year))
                proxy_abs_err.append(abs(pred_year_proxy - true_year))
                proxy_confusion[true_year, pred_year_proxy] += 1

                # ── knn-based ────────────────────────────────────────────
                neighbours    = get_neighbours(ann, meta, emb, top_k)
                mean_year     = np.mean([n["year_idx"] for n in neighbours])
                pred_year_knn = int(round(mean_year))

                knn_correct.append(int(pred_year_knn == true_year))
                knn_abs_err.append(abs(pred_year_knn - true_year))
                knn_confusion[true_year, pred_year_knn] += 1

    evaluation_results = {
        # ── global scalars ──────────────────────────────────────────────
        "proxy_year_accuracy": float(np.mean(proxy_correct)),
        "proxy_year_mae":      float(np.mean(proxy_abs_err)),
        "knn_year_accuracy":   float(np.mean(knn_correct)),
        "knn_year_mae":        float(np.mean(knn_abs_err)),
        # ── confusion matrices (raw counts, row = true year, col = pred) ─
        "proxy_year_confusion": proxy_confusion.tolist(),
        "knn_year_confusion":   knn_confusion.tolist(),
    }
    print(evaluation_results)

    return evaluation_results


# ── 2. OBJECT ESTIMATION ─────────────────────────────────────────────────────

def eval_object_estimation(
    ann: AnnoyIndex, meta: dict,
    cat_proxies: torch.Tensor,   # (n_obj, D)
    test_loader: DataLoader,
    model, device, top_k: int,
    objects: list,
):
    """
    For each test image:
      - Embed with CNN.
      - Proxy-based category prediction:
          pred_cat = argmin_c  dist(emb, cat_proxy[c])
      - KNN-based category prediction:
          majority vote of TOP_K neighbours' category field.

    Returns:
      - Global accuracy for both methods.
      - Confusion matrices  proxy_confusion[true_obj][pred_obj]  and
        knn_confusion[true_obj][pred_obj]  (raw counts).
        Row/column order follows `objects`.
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    proxy_correct = []
    knn_correct   = []

    proxy_confusion = np.zeros((n_obj, n_obj), dtype=np.int64)
    knn_confusion   = np.zeros((n_obj, n_obj), dtype=np.int64)

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(test_loader, desc="Standalone object estimation"):
            embs      = embed_batch(image_A, condition_A, model, device, normalize=False)
            true_cats = list(category) if not isinstance(category, list) else category

            for i in range(len(embs)):
                emb      = embs[i]
                true_cat = true_cats[i]
                true_idx = obj2idx[true_cat]

                # ── proxy-based ──────────────────────────────────────────
                dists      = torch.cdist(emb.unsqueeze(0), cat_proxies).squeeze(0)
                pred_idx   = int(dists.argmin())
                pred_proxy = objects[pred_idx]

                proxy_correct.append(int(pred_proxy == true_cat))
                proxy_confusion[true_idx, pred_idx] += 1

                # ── knn-based ────────────────────────────────────────────
                neighbours = get_neighbours(ann, meta, emb, top_k)
                cats       = [n["category"] for n in neighbours]
                pred_knn   = Counter(cats).most_common(1)[0][0]
                pred_knn_idx = obj2idx.get(pred_knn, 0)

                knn_correct.append(int(pred_knn == true_cat))
                knn_confusion[true_idx, pred_knn_idx] += 1

    evaluate_results = {
        # ── global scalars ──────────────────────────────────────────────
        "proxy_object_accuracy": float(np.mean(proxy_correct)),
        "knn_object_accuracy":   float(np.mean(knn_correct)),
        # ── confusion matrices (raw counts, row = true obj, col = pred) ─
        "proxy_object_confusion": proxy_confusion.tolist(),
        "knn_object_confusion":   knn_confusion.tolist(),
    }
    print(evaluate_results)
    return evaluate_results


# ── 3. IMAGE + YEAR-PROXY TRANSLATION ────────────────────────────────────────

def eval_image_year_proxy_translation(
    ann: AnnoyIndex, meta: dict,
    year_proxies: torch.Tensor,  # (n_yr, D)
    test_loader: DataLoader,
    model, device, top_k: int,
    objects: list,
):
    """
    For every query image and EVERY possible year index y:
        query_vec = CNN(image) + year_proxy[y]
    Retrieve TOP_K and measure:
      - object_accuracy  : fraction retrieved with category == query category
      - year_consistency : fraction retrieved with year_idx == y

    Fine-grained breakdowns:
      - obj_acc_by_category[cat]       : mean object accuracy for queries of that category
                                         (averaged over all target years)
      - year_cons_by_year[y]           : mean year consistency when targeting year y
                                         (averaged over all query categories)
      - error_matrix[cat][y]           : mean object accuracy for (category=cat, target_year=y)
      - year_cons_matrix[cat][y]       : mean year consistency for (category=cat, target_year=y)
    """
    n_years = year_proxies.shape[0]
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_accs    = []
    year_consts = []

    # accumulators: shape (n_obj, n_years)
    obj_acc_sum    = np.zeros((n_obj, n_years))
    year_cons_sum  = np.zeros((n_obj, n_years))
    count_matrix   = np.zeros((n_obj, n_years), dtype=np.int64)

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(
                test_loader, desc="Image+YearProxy translation"):
            embs      = embed_batch(image_A, condition_A, model, device, normalize=False)
            true_cats = list(category) if not isinstance(category, list) else category

            for i in range(len(embs)):
                emb      = embs[i]
                true_cat = true_cats[i]
                cat_idx  = obj2idx[true_cat]

                for y in range(n_years):
                    query_vec  = emb + year_proxies[y]
                    neighbours = get_neighbours(ann, meta, query_vec, top_k)

                    obj_acc    = float(np.mean([int(n["category"] == true_cat) for n in neighbours]))
                    year_const = float(np.mean([int(n["year_idx"]  == y)       for n in neighbours]))

                    obj_accs.append(obj_acc)
                    year_consts.append(year_const)

                    obj_acc_sum[cat_idx, y]   += obj_acc
                    year_cons_sum[cat_idx, y] += year_const
                    count_matrix[cat_idx, y]  += 1

    # normalise
    safe_count = np.where(count_matrix > 0, count_matrix, 1)
    error_matrix     = obj_acc_sum   / safe_count   # (n_obj, n_years)
    year_cons_matrix = year_cons_sum / safe_count   # (n_obj, n_years)

    # per-category average (over all years)
    obj_acc_by_cat  = {objects[c]: float(error_matrix[c].mean())     for c in range(n_obj)}
    # per-year average (over all categories)
    year_cons_by_yr = {y: float(year_cons_matrix[:, y].mean())       for y in range(n_years)}

    evaluation_results =  {
        # ── global scalars ──────────────────────────────────────────────
        "translation_object_accuracy":  float(np.mean(obj_accs)),
        "translation_year_consistency": float(np.mean(year_consts)),
        # ── per-category object accuracy (averaged over target years) ───
        "translation_obj_acc_by_category": obj_acc_by_cat,
        # ── per-target-year consistency (averaged over query categories)─
        "translation_year_cons_by_year": year_cons_by_yr,
        # ── full error matrices [category][target_year] ─────────────────
        # object_accuracy at each (query_category, target_year) cell
        "translation_obj_acc_matrix":    {objects[c]: error_matrix[c].tolist()     for c in range(n_obj)},
        # year_consistency at each (query_category, target_year) cell
        "translation_year_cons_matrix":  {objects[c]: year_cons_matrix[c].tolist() for c in range(n_obj)},
    }
    print(evaluation_results)
    return evaluation_results


# ── 4. TWO-IMAGE TRANSLATION ─────────────────────────────────────────────────

def eval_two_image_translation(
    ann: AnnoyIndex, meta: dict,
    cat_proxies: torch.Tensor,
    year_proxies,# (n_obj, D)
    n_train_items: int,
    test_loader: DataLoader,
    model, device, top_k: int,
    objects: list,
    n_years: int,
):
    """
    For every query image q:
        1. Sample a RANDOM reference vector directly from the Annoy index.
        2. cat_emb  = CNN(q)
           ref_emb  = ann.get_item_vector(random_idx)
           closest_cat_proxy = argmin_c dist(ref_emb, cat_proxy[c])
           year_vec  = ref_emb − cat_proxy[closest_cat_proxy]
           query_vec = F.normalize(cat_emb + year_vec)
        3. Retrieve TOP_K, measure:
           - object_accuracy  : fraction with category == query category
           - year_consistency : fraction with year_idx == reference year_idx

    Fine-grained breakdowns:
      - obj_acc_by_category[cat]    : mean object accuracy for queries of that category
      - year_cons_by_ref_year[y]    : mean year consistency when the reference image is year y
      - obj_acc_matrix[cat][y]      : mean object accuracy for (query_cat=cat, ref_year=y)
      - year_cons_matrix[cat][y]    : mean year consistency for (query_cat=cat, ref_year=y)
    """
    n_obj   = len(objects)
    obj2idx = {o: i for i, o in enumerate(objects)}

    obj_accs    = []
    year_consts = []
    year_ignorarship = []
    year_mae_two_images = []

    # accumulators: (n_obj, n_years) indexed by [query_category, ref_year]
    obj_acc_sum   = np.zeros((n_obj, n_years))
    year_cons_sum = np.zeros((n_obj, n_years))
    count_matrix  = np.zeros((n_obj, n_years), dtype=np.int64)

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(
                test_loader, desc="Two-image translation..."):
            embs      = embed_batch(image_A, condition_A, model, device, normalize=False)
            true_cats = list(category) if not isinstance(category, list) else category

            for i in range(len(embs)):
                cat_emb  = embs[i]
                true_cat = true_cats[i]
                cat_idx  = obj2idx[true_cat]

                # Sample reference embedding directly from Annoy — zero extra cost
                ref_emb, ref_cat, ref_year, norm = sample_random_train_ref(ann, meta, n_train_items, return_norm=True)
                ref_emb = ref_emb * norm

                dists_cat         = torch.cdist(ref_emb.unsqueeze(0), cat_proxies).squeeze(0)
                closest_proxy_idx = int(dists_cat.argmin())
                year_vec          = ref_emb - cat_proxies[closest_proxy_idx]



                query_vec  = F.normalize(cat_emb + year_vec, p=2, dim=0)
                # query_vec  = cat_emb + year_vec
                neighbours = get_neighbours(ann, meta, query_vec, top_k)

                obj_acc    = float(np.mean([int(n["category"] == true_cat) for n in neighbours]))
                year_const = float(np.mean([int(n["year_idx"]  == ref_year) for n in neighbours]))
                year_mistakes = float(np.mean([int(n["year_idx"]  == condition_B[i].item()) for n in neighbours]))


                mean_year = np.mean([n["year_idx"] for n in neighbours])
                pred_year_knn = int(round(mean_year))
                year_mae_two_images.append(abs(pred_year_knn - ref_year))

                obj_accs.append(obj_acc)
                year_consts.append(year_const)
                year_ignorarship.append(year_mistakes)

                obj_acc_sum[cat_idx, ref_year]   += obj_acc
                year_cons_sum[cat_idx, ref_year] += year_const
                count_matrix[cat_idx, ref_year]  += 1

    safe_count = np.where(count_matrix > 0, count_matrix, 1)
    obj_acc_matrix   = obj_acc_sum   / safe_count   # (n_obj, n_years)
    year_cons_matrix = year_cons_sum / safe_count   # (n_obj, n_years)

    obj_acc_by_cat      = {objects[c]: float(obj_acc_matrix[c].mean())   for c in range(n_obj)}
    year_cons_by_yr     = {y: float(year_cons_matrix[:, y].mean())       for y in range(n_years)}

    evaluation_results =  {
        # ── global scalars ──────────────────────────────────────────────
        "two_image_object_accuracy":   float(np.mean(obj_accs)),
        "two_image_year_consistency":  float(np.mean(year_consts)),
        'two_images_year_UNCONSISTENCY': float(np.mean(year_ignorarship)),
        'two_images_year_mae': float(np.mean(year_mae_two_images)),
        # ── per-category object accuracy (averaged over ref years) ──────
        "two_image_obj_acc_by_category": obj_acc_by_cat,
        # ── per-ref-year consistency (averaged over query categories) ───
        "two_image_year_cons_by_ref_year": year_cons_by_yr,
        # ── full matrices [query_category][ref_year] ────────────────────
        "two_image_obj_acc_matrix":   {objects[c]: obj_acc_matrix[c].tolist()   for c in range(n_obj)},
        "two_image_year_cons_matrix": {objects[c]: year_cons_matrix[c].tolist() for c in range(n_obj)},
    }
    print(evaluation_results)
    return evaluation_results

# ─────────────────────────────────────────────────────────────────────────────
# Sample a random reference item directly from the Annoy index
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_train_ref(ann: AnnoyIndex, meta: dict, n_items: int, return_norm = False):
    """
    Pick one random index from the Annoy index and return its vector + meta.
    The embedding is already stored in Annoy — no re-encoding needed.

    Returns:
        ref_emb   : torch.Tensor  (D,)
        ref_cat   : str
        ref_year  : int
    """
    idx      = random.randrange(n_items)
    ref_emb  = torch.tensor(ann.get_item_vector(idx), dtype=torch.float32)
    m        = meta[str(idx)]
    if return_norm:
        return ref_emb, m["category"], m["year_idx"], m['norm']
    return ref_emb, m["category"], m["year_idx"]


# ─────────────────────────────────────────────────────────────────────────────
# Build test dataloader
# ─────────────────────────────────────────────────────────────────────────────

def build_test_loader(objects: list, avlabels, batch_size: int, num_workers: int):
    dataset = SpecialistDataloaderWithClass(
        data_test, objects[0], transforms=IMAGENET_TRANSFORMS_VAL, evaluate = True
    )
    dataset.available_labels = avlabels
    for obj in objects[1:]:
        extra = SpecialistDataloaderWithClass(
            data_test, obj, transforms=IMAGENET_TRANSFORMS_VAL, evaluate = True
        )
        dataset = dataset + extra
        dataset.available_labels = avlabels
    print("Test dataset loaded with size", len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CIR offline evaluation")
    p.add_argument("--objects",     nargs="+", required=True)
    p.add_argument("--ckpt_folder", required=True)
    p.add_argument("--epoch",       type=int,  default=None)
    p.add_argument("--top_k",       type=int,  default=TOP_K)
    p.add_argument("--batch_size",  type=int,  default=32)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--rebuild",     action="store_true",
                   help="Force rebuild of the eval Annoy index.")
    p.add_argument("--seed",        type=int,  default=42,
                   help="Random seed for reproducibility (two-image eval).")
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    epoch = args.epoch if args.epoch is not None else latest_epoch(args.ckpt_folder)
    print(f"Epoch  : {epoch}")
    print(f"TOP_K  : {args.top_k}")

    # ── load model + proxies ─────────────────────────────────────────────────
    model, contrastive_loss_fn, year_loss_fn, avlabels = load_everything(
        args.ckpt_folder, args.objects, epoch, device
    )
    cat_proxies  = contrastive_loss_fn.proxies.detach().cpu()   # (n_obj, D)
    year_proxies = year_loss_fn.proxies.detach().cpu()          # (n_yr,  D)
    obj2idx      = {obj: i for i, obj in enumerate(args.objects)}

    print(f"  Category proxies : {cat_proxies.shape}")
    print(f"  Year proxies     : {year_proxies.shape}")

    # ── build / load eval Annoy index ────────────────────────────────────────
    ann, meta = build_eval_annoy_index(
        objects     = args.objects,
        ckpt_folder = args.ckpt_folder,
        epoch       = epoch,
        device      = device,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        rebuild     = args.rebuild,
    )

    # ── build test dataloader ────────────────────────────────────────────────
    test_loader = build_test_loader(
        args.objects, avlabels, args.batch_size, args.num_workers
    )

    # Number of items in the Annoy index (keys are str(0)…str(N-1))
    n_train_items = len(meta)
    print(f"\n  Annoy index size: {n_train_items} train items")

    year_labels = list(range(year_proxies.shape[0]))

    # ── run evaluations ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1 / 4  DATE ESTIMATION")
    print("=" * 60)
    date_results = eval_date_estimation(
        ann, meta, cat_proxies, year_proxies,
        test_loader, model, device, args.top_k, obj2idx
    )
    _print_section(date_results, year_labels, args.objects, "date")

    print("\n" + "=" * 60)
    print("2 / 4  OBJECT ESTIMATION")
    print("=" * 60)
    obj_results = eval_object_estimation(
        ann, meta, cat_proxies,
        test_loader, model, device, args.top_k, args.objects
    )
    _print_section(obj_results, year_labels, args.objects, "object")

    print("\n" + "=" * 60)
    print("3 / 4  IMAGE + YEAR-PROXY TRANSLATION")
    print("=" * 60)
    trans_results = eval_image_year_proxy_translation(
        ann, meta, year_proxies,
        test_loader, model, device, args.top_k,
        objects=args.objects,
    )
    _print_section(trans_results, year_labels, args.objects, "translation")

    print("\n" + "=" * 60)
    print("4 / 4  TWO-IMAGE TRANSLATION")
    print("=" * 60)
    two_img_results = eval_two_image_translation(
        ann, meta, cat_proxies, year_proxies, n_train_items,
        test_loader, model, device, args.top_k,
        objects=args.objects,
        n_years=year_proxies.shape[0],
    )
    _print_section(two_img_results, year_labels, args.objects, "two_image")

    # ── save results ─────────────────────────────────────────────────────────
    all_results = {**date_results, **obj_results, **trans_results, **two_img_results}
    summary_path = os.path.join(
        args.ckpt_folder, f"eval_epoch{epoch}_topk{args.top_k}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✔  Full results (including matrices) saved to {summary_path}")

    # ── global scalar summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALAR SUMMARY")
    print("=" * 60)
    _print_scalars(all_results)


def _print_scalars(d: dict):
    """Print only the scalar (float) entries of a results dict."""
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<50s}: {v:.4f}")


def _print_confusion(matrix: list, labels: list, title: str):
    """Pretty-print a 2-D confusion matrix with row/col labels."""
    n = len(matrix)
    col_w = max(len(str(l)) for l in labels) + 2
    head  = " " * (col_w + 2) + "".join(f"{str(l):>{col_w}}" for l in labels)
    print(f"\n  {title}")
    print("  " + head)
    for i, row in enumerate(matrix):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {str(labels[i]):<{col_w}}  {row_str}")


def _print_matrix(matrix: dict, col_labels: list, title: str, fmt: str = ".3f"):
    """Pretty-print a category × year float matrix."""
    col_w = 8
    head  = " " * 22 + "".join(f"{str(l):>{col_w}}" for l in col_labels)
    print(f"\n  {title}")
    print("  " + head)
    for row_label, row in matrix.items():
        row_str = "".join(f"{v:{col_w}{fmt}}" for v in row)
        print(f"  {str(row_label):<22}{row_str}")


def _print_section(results: dict, year_labels: list, obj_labels: list, section: str):
    """Dispatch printing based on which eval section we're in."""
    _print_scalars(results)

    if section == "date":
        for key, title in [
            ("proxy_year_confusion", "Proxy year confusion (row=true, col=pred)"),
            ("knn_year_confusion",   "KNN   year confusion (row=true, col=pred)"),
        ]:
            if key in results:
                _print_confusion(results[key], year_labels, title)

    elif section == "object":
        for key, title in [
            ("proxy_object_confusion", "Proxy object confusion (row=true, col=pred)"),
            ("knn_object_confusion",   "KNN   object confusion (row=true, col=pred)"),
        ]:
            if key in results:
                _print_confusion(results[key], obj_labels, title)

    elif section in ("translation", "two_image"):
        prefix = "translation" if section == "translation" else "two_image"
        ref_label = "target year" if section == "translation" else "ref year"

        # per-category breakdown
        by_cat_key = f"{prefix}_obj_acc_by_category"
        if by_cat_key in results:
            print(f"\n  Object accuracy by query category:")
            for cat, v in results[by_cat_key].items():
                print(f"    {cat:<20s}: {v:.4f}")

        # per-year breakdown
        by_yr_key = f"{prefix}_year_cons_by_year" if section == "translation" else f"{prefix}_year_cons_by_ref_year"
        if by_yr_key in results:
            print(f"\n  Year consistency by {ref_label}:")
            for y, v in results[by_yr_key].items():
                print(f"    year {y:<4}: {v:.4f}")

        # full error matrices
        obj_mat_key  = f"{prefix}_obj_acc_matrix"
        year_mat_key = f"{prefix}_year_cons_matrix"
        if obj_mat_key in results:
            _print_matrix(results[obj_mat_key],  year_labels, f"Object accuracy matrix [category × {ref_label}]")
        if year_mat_key in results:
            _print_matrix(results[year_mat_key], year_labels, f"Year consistency matrix [category × {ref_label}]")



if __name__ == "__main__":
    main()
