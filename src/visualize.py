"""
visualize.py
------------
Extract embeddings from the TEST split, run 3-D t-SNE, save thumbnail images,
and write a CSV ready for tools like Embedding Projector or custom plotters.

Output layout
─────────────
  /data/113-2/amolina/CIR-Visualize/<run_id>/
      0.png, 1.png, 2.png, …          ← 64×64 thumbnail per test image
      embeddings.csv                   ← filename, embedding, x, y, z

CSV schema
──────────
  filename  : relative path to the saved thumbnail  (e.g. "42.png")
  embedding : raw 1024-d vector, comma-separated, quoted
  x, y, z   : 3-D t-SNE coordinates
  category  : object class string
  year_idx  : integer year label

Usage
─────
    python visualize.py \\
        --objects Dog Cat \\
        --ckpt_folder /data/.../cir_date/<run_id>/ \\
        --epoch 10 \\
        [--batch_size 32] \\
        [--num_workers 4] \\
        [--tsne_perplexity 30] \\
        [--tsne_iter 1000] \\
        [--out_root /data/113-2/amolina/CIR-Visualize]
"""

import argparse
import csv
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_metric_learning import losses
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
from core_datautils import df_test as data_test
from models import ConditionedToYear, SpecialistModel
from train_experts_dataloader import SpecialistDataloaderWithClass
from torchvision import transforms

# ── constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1024
THUMB_SIZE    = (64, 64)
OUT_ROOT_DEFAULT = "/data/113-2/users/amolina/CIR-Visualize"

# ImageNet stats for de-normalising tensors back to pixel values
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

IMAGENET_TRANSFORMS_VAL = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def latest_epoch(ckpt_folder: str) -> int:
    epochs = [
        int(m.group(1))
        for f in os.listdir(ckpt_folder)
        if (m := re.match(r"model_epoch(\d+)\.pth", f))
    ]
    if not epochs:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_folder}")
    return max(epochs)


def load_model(ckpt_folder: str, objects: list, epoch: int, device):
    """Load only the backbone model (we don't need the loss fns for embedding)."""
    # Need avlabels to know n_years
    dataset0 = SpecialistDataloaderWithClass(
        data_test, objects[0], transforms=IMAGENET_TRANSFORMS_VAL
    )
    avlabels = dataset0.available_labels
    n_years  = len(avlabels)

    date_sensitive = SpecialistModel(n_years).to(device)
    model = ConditionedToYear(date_sensitive, output_dim=n_years).to(device)
    model.load_state_dict(
        torch.load(f"{ckpt_folder}/model_epoch{epoch}.pth", map_location=device)
    )
    model.eval()
    return model, avlabels


def tensor_to_pil_thumb(tensor: torch.Tensor) -> Image.Image:
    """De-normalise a CHW ImageNet tensor and return a 64×64 PIL image."""
    img = tensor.cpu().float() * _STD + _MEAN
    img = img.clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr)
    return pil.resize(THUMB_SIZE, Image.LANCZOS)


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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(model, test_loader, device):
    """
    Returns:
        embeddings : np.ndarray  (N, 1024)   raw (un-normalised) embeddings
        images     : list of torch.Tensor     CHW normalised tensors (for thumbnails)
        categories : list of str
        year_idxs  : list of int
    """
    embeddings = []
    images     = []
    categories = []
    year_idxs  = []

    with torch.no_grad():
        for image_A, condition_A, image_B, condition_B, category in tqdm(
                test_loader, desc="Extracting embeddings"):
            image_A     = image_A.to(device)
            condition_A = condition_A.to(device)

            emb_agnostic, _ = model(image_A, condition_A)
            emb_np = emb_agnostic.cpu().numpy()          # (B, 1024) — raw, not normalised

            embeddings.append(emb_np)
            for i in range(len(emb_np)):
                images.append(image_A[i].cpu())
                cat = category[i] if isinstance(category[i], str) else str(category[i])
                categories.append(cat)
                year_idxs.append(int(condition_B[i].item()))

    return np.concatenate(embeddings, axis=0), images, categories, year_idxs


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(embeddings: np.ndarray, perplexity: int, n_iter: int) -> np.ndarray:
    """Run 3-component t-SNE. Returns (N, 3) array."""
    print(f"Running t-SNE on {embeddings.shape[0]} points "
          f"(perplexity={perplexity}, n_iter={n_iter}) …")
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        # n_iter=n_iter,
        init="pca",
        random_state=42,
        verbose=1,
    )
    return tsne.fit_transform(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Save thumbnails + CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    out_dir: str,
    embeddings: np.ndarray,
    tsne_coords: np.ndarray,
    images: list,
    categories: list,
    year_idxs: list,
):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "embeddings.csv")

    print(f"Saving {len(images)} thumbnails + CSV to {out_dir} …")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["filename", "embedding", "x", "y", "z", "category", "year_idx"])

        for idx, (emb, coords, img_tensor, cat, yr) in enumerate(
                tqdm(zip(embeddings, tsne_coords, images, categories, year_idxs),
                     total=len(images), desc="Saving")):

            # ── thumbnail ────────────────────────────────────────────────
            thumb_name = f"{idx}.png"
            thumb_path = os.path.join(out_dir, thumb_name)
            pil = tensor_to_pil_thumb(img_tensor)
            pil.save(thumb_path)

            # ── embedding string: comma-separated floats, quoted ─────────
            emb_str = ",".join(str(v) for v in emb.tolist())

            writer.writerow([
                thumb_name,
                emb_str,            # csv.writer will add surrounding quotes due to the comma
                float(coords[0]),
                float(coords[1]),
                float(coords[2]),
                cat,
                yr,
            ])

    print(f"✔  CSV saved  : {csv_path}")
    print(f"✔  Thumbnails : {out_dir}/<idx>.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="t-SNE visualisation of CIR test embeddings")
    p.add_argument("--objects",          nargs="+", required=True)
    p.add_argument("--ckpt_folder",      required=True,
                   help="Folder with model_epochN.pth checkpoints.")
    p.add_argument("--epoch",            type=int,  default=None,
                   help="Epoch to load (defaults to latest).")
    p.add_argument("--batch_size",       type=int,  default=32)
    p.add_argument("--num_workers",      type=int,  default=4)
    p.add_argument("--tsne_perplexity",  type=int,  default=30)
    p.add_argument("--tsne_iter",        type=int,  default=1000)
    p.add_argument("--out_root",         default=OUT_ROOT_DEFAULT,
                   help="Root output directory. A sub-folder named after the run "
                        "will be created inside.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    epoch = args.epoch if args.epoch is not None else latest_epoch(args.ckpt_folder)
    print(f"Epoch  : {epoch}")

    # Sub-folder name = random 6-digit number so repeated runs don't clobber each other
    run_id  = str(random.randint(100_000, 999_999))
    out_dir = os.path.join(args.out_root, run_id)
    print(f"Output : {out_dir}")

    # ── load model ───────────────────────────────────────────────────────────
    model, avlabels = load_model(args.ckpt_folder, args.objects, epoch, device)

    # ── build test dataloader ────────────────────────────────────────────────
    test_loader = build_test_loader(
        args.objects, avlabels, args.batch_size, args.num_workers
    )

    # ── extract embeddings ───────────────────────────────────────────────────
    embeddings, images, categories, year_idxs = extract_embeddings(
        model, test_loader, device
    )
    print(f"Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    tsne_coords = run_tsne(embeddings, args.tsne_perplexity, args.tsne_iter)

    # ── save ─────────────────────────────────────────────────────────────────
    save_results(out_dir, embeddings, tsne_coords, images, categories, year_idxs)

    print(f"\n✔  Done.  Run ID: {run_id}")
    print(f"   Load the CSV in Embedding Projector or your own plotter.")


if __name__ == "__main__":
    main()