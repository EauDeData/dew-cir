import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from train_experts_dataloader import SpecialistDataloaderWithClass, tqdm
from models import SpecialistModel, ConditionedToYear
from core_datautils import df as data_complete
from pytorch_metric_learning import losses
import torch.nn.functional as F
import os
import uuid

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

def save_checkpoint(model, contrastive_loss_fn, year_loss_fn, epoch, args):
    run_id = getattr(args, 'run_id', None)
    if run_id is None:
        args.run_id = str(uuid.uuid4())[:8]
        run_id = args.run_id

    save_dir = f"/data/113-2/users/amolina/cir_date/{run_id}/"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(),                f"{save_dir}/model_epoch{epoch}.pth")
    torch.save(contrastive_loss_fn.state_dict(),  f"{save_dir}/contrastive_loss_epoch{epoch}.pth")
    torch.save(year_loss_fn.state_dict(),         f"{save_dir}/year_loss_epoch{epoch}.pth")

    print(f"Checkpoint saved to {save_dir}")

def load_all_objects():
    return ['Dog']

def raw_texts_to_labels(raw_texts, label_map):
    labels = []

    for txt in raw_texts:
        labels.append(label_map[txt])
    return torch.tensor(labels, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SpecialistModel for target objects.")

    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        required=True,
        help="Target object classes (e.g. Tree Car Dog) or 'ALL'"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    if len(args.objects) == 1 and args.objects[0].upper() == "ALL":
        args.objects = load_all_objects()

    return args

def log_proxy_tsne(contrastive_loss_fn, year_loss_fn, obj2label, avlabels, epoch, logger):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.manifold import TSNE
    import numpy as np

    # --- Extract proxies ---
    obj_proxies  = contrastive_loss_fn.proxies.detach().cpu()   # (n_objects, 1024)
    year_proxies = year_loss_fn.proxies.detach().cpu()          # (n_years, 1024)

    n_objects = len(obj2label)
    n_years   = len(avlabels)

    # --- Build translated year proxies: one per (object, year) pair ---
    # Shape: (n_objects * n_years, 1024)
    translated = []
    translated_year_labels = []   # year index → brightness
    translated_obj_labels  = []   # object index → color hue

    for obj, obj_idx in obj2label.items():
        for year_idx in range(n_years):
            translated.append(obj_proxies[obj_idx] + year_proxies[year_idx])
            translated_year_labels.append(year_idx)
            translated_obj_labels.append(obj_idx)

    translated = torch.stack(translated).numpy()                # (n_objects*n_years, 1024)
    translated_year_labels = np.array(translated_year_labels)
    translated_obj_labels  = np.array(translated_obj_labels)

    # --- Combine all points for TSNE ---
    all_points = np.concatenate([obj_proxies.numpy(), translated], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_points) - 1))
    embedded = tsne.fit_transform(all_points)

    obj_embedded         = embedded[:n_objects]
    translated_embedded  = embedded[n_objects:]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#0e0e0e")
    fig.patch.set_facecolor("#0e0e0e")

    # Object proxies: one distinct color per object, large marker
    object_cmap = cm.get_cmap("tab10", n_objects)
    for obj, obj_idx in obj2label.items():
        ax.scatter(
            obj_embedded[obj_idx, 0], obj_embedded[obj_idx, 1],
            color=object_cmap(obj_idx),
            s=300, marker="*", zorder=5,
            label=f"[proxy] {obj}",
            edgecolors="white", linewidths=0.5,
        )

    # Translated year proxies: hue = object, brightness = year index
    year_norm = plt.Normalize(0, n_years - 1)
    for obj, obj_idx in obj2label.items():
        mask = translated_obj_labels == obj_idx
        sc = ax.scatter(
            translated_embedded[mask, 0],
            translated_embedded[mask, 1],
            c=translated_year_labels[mask],
            cmap="plasma",
            norm=year_norm,
            s=60, marker="o", zorder=3,
            alpha=0.85,
            edgecolors=object_cmap(obj_idx),
            linewidths=1.2,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Year index (brighter = later)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.legend(loc="upper left", framealpha=0.2, labelcolor="white", fontsize=9)
    ax.set_title(f"Proxy t-SNE — Epoch {epoch}", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()

    if logger:
        logger.log({"proxy_tsne": logger.Image(fig), "epoch": epoch})

    plt.close(fig)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logger ---
    logger = None
    if args.use_wandb:
        import wandb
        wandb.init(
            project="specialist-training",
            config=vars(args),
            name=f"{'_'.join(args.objects)}_bs{args.batch_size}_lr{args.lr}",
        )
        logger = wandb

    # --- Data ---
    dataset = SpecialistDataloaderWithClass(
        data_complete, args.objects[0], transforms=IMAGENET_TRANSFORMS_TRAIN,
    )
    avlabels = dataset.available_labels
    obj2label = {args.objects[0]: 0}
    for obj in args.objects[1:]:
        dataset = dataset + SpecialistDataloaderWithClass(
            data_complete, obj, transforms=IMAGENET_TRANSFORMS_TRAIN,
        )
        dataset.available_labels = avlabels
        obj2label[obj] = len(obj2label)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val   = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    val_set.dataset.es_trans = IMAGENET_TRANSFORMS_VAL

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # --- Losses ---
    contrastive_loss_fn = losses.ProxyNCALoss(len(args.objects), 1024, softmax_scale=1)
    year_loss_fn = losses.ProxyNCALoss(len(avlabels), 1024, softmax_scale=1)


    # --- Model ---
    date_sensitive = SpecialistModel(len(dataset.available_labels)).to(device)
    model = ConditionedToYear(date_sensitive, output_dim=len(dataset.available_labels)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(contrastive_loss_fn.parameters()) + list(year_loss_fn.parameters()), lr=args.lr)



    def transe_loss(hr, t):
        # h + r should land on t → minimize ||h + r - t||
        l2 = torch.norm(hr - t, dim=-1)
        return l2.mean()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        log_proxy_tsne(contrastive_loss_fn, year_loss_fn, obj2label, avlabels, epoch, logger)

        for image_A, condition_A, image_B, condition_B, category in tqdm(train_loader):
            image_A     = image_A.to(device)
            image_B     = image_B.to(device)
            condition_A = condition_A.to(device)
            condition_B = condition_B.to(device)
            category    = raw_texts_to_labels(category, obj2label).to(device)

            emb_A_agnostic, emb_A_conditioned = model(image_A, condition_A)
            emb_B_agnostic, emb_B_conditioned = model(image_B, condition_B)

            # Contrastive loss on year-agnostic embeddings
            all_agnostic = torch.cat([emb_A_agnostic, emb_B_agnostic], dim=0)

            all_labels   = torch.cat([category,       category],       dim=0)
            loss_contrastive = contrastive_loss_fn(all_agnostic, all_labels)

            proxies = contrastive_loss_fn.proxies  # (num_classes, 1024)
            proxy_vectors = proxies[category]  # shape: (batch_size, 1024)

            # Centrem els embeddigns restantli els centroides de les seves categories
            emb_A_agnostic_centered = emb_A_agnostic - proxy_vectors
            emb_A_conditioned_centered = emb_A_conditioned - proxy_vectors

            emb_B_agnostic_centered = emb_B_agnostic - proxy_vectors
            emb_B_conditioned_centered = emb_B_conditioned - proxy_vectors

            all_embeddings = torch.cat([emb_A_agnostic_centered, emb_B_agnostic_centered, emb_A_conditioned_centered, emb_B_conditioned_centered], dim=0)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

            year_labels = torch.cat([condition_B, condition_A, condition_A, condition_B], dim=-1)


            loss_trans = year_loss_fn(all_embeddings, year_labels)


            # Això funcionava:
            # TransE loss: emb_X_agnostic + year_encoding ≈ emb_Y_agnostic
            # loss_trans = (
            #     transe_loss(emb_A_conditioned, emb_B_agnostic.detach()) +
            #     transe_loss(emb_B_conditioned, emb_A_agnostic.detach())
            # ) / 2
            #
            loss = loss_contrastive + loss_trans

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if logger:
                logger.log({
                    "train/loss_total":       loss.item(),
                    "train/loss_object": loss_contrastive.item(),
                    "train/loss_date_geom":      loss_trans.item(),
                    "epoch": epoch,
                    "step":  global_step,
                })
            global_step += 1

        # --- Validation ---
        model.eval()
        val_metrics = {"val/loss_total": 0, "val/loss_object": 0, "val/loss_date_geom": 0}
        n_batches = 0
        with torch.no_grad():
            for image_A, condition_A, image_B, condition_B, category in val_loader:
                image_A = image_A.to(device)
                image_B = image_B.to(device)
                condition_A = condition_A.to(device)
                condition_B = condition_B.to(device)
                category = raw_texts_to_labels(category, obj2label).to(device)

                emb_A_agnostic, emb_A_conditioned = model(image_A, condition_A)
                emb_B_agnostic, emb_B_conditioned = model(image_B, condition_B)

                # Contrastive loss on year-agnostic embeddings
                all_agnostic = torch.cat([emb_A_agnostic, emb_B_agnostic], dim=0)

                all_labels = torch.cat([category, category], dim=0)
                loss_contrastive = contrastive_loss_fn(all_agnostic, all_labels)

                proxies = contrastive_loss_fn.proxies  # (num_classes, 1024)
                proxy_vectors = proxies[category]  # shape: (batch_size, 1024)

                # Centrem els embeddigns restantli els centroides de les seves categories
                emb_A_agnostic_centered = emb_A_agnostic - proxy_vectors
                emb_A_conditioned_centered = emb_A_conditioned - proxy_vectors

                emb_B_agnostic_centered = emb_B_agnostic - proxy_vectors
                emb_B_conditioned_centered = emb_B_conditioned - proxy_vectors

                all_embeddings = torch.cat(
                    [emb_A_agnostic_centered, emb_B_agnostic_centered, emb_A_conditioned_centered,
                     emb_B_conditioned_centered], dim=0)
                all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

                year_labels = torch.cat([condition_B, condition_A, condition_A, condition_B], dim=-1)

                loss_trans = year_loss_fn(all_embeddings, year_labels)

                val_metrics["val/loss_object"] += loss_contrastive.item()
                val_metrics["val/loss_date_geom"]      += loss_trans.item()
                val_metrics["val/loss_total"]       += (loss_contrastive + loss_trans).item()
                n_batches += 1

        val_metrics = {k: v / n_batches for k, v in val_metrics.items()}
        val_metrics["epoch"] = epoch
        print(f"Epoch {epoch} | " + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
        if logger:
            logger.log(val_metrics)
        save_checkpoint(model, contrastive_loss_fn, year_loss_fn, epoch, args)


main()