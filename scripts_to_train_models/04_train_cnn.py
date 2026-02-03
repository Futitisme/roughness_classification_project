#!/usr/bin/env python3
"""
04_train_cnn.py

Trains a CNN model (using PyTorch nn.Module) with grid search
for hyperparameter tuning on the Surface Roughness Classification dataset.

Architecture:
- 3 Conv blocks: Conv2d -> ReLU -> MaxPool2d
- Classifier: Flatten -> Linear -> ReLU -> Dropout -> Linear

Outputs:
- checkpoints/cnn_best.pt (best model checkpoint)
- results_cnn.csv (grid search results)
- Learning curves and confusion matrix plots
"""

import os
import sys
import time
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
IMAGE_SIZE = 64
NUM_CLASSES = 16

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 256
NUM_WORKERS = 2

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset
# ============================================================================
class SurfaceRoughnessDataset(Dataset):
    def __init__(self, df, image_size=64):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        y = int(row["label"])

        img = Image.open(path).convert("L")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ============================================================================
# Metrics
# ============================================================================
@torch.no_grad()
def confusion_matrix_torch(y_true, y_pred, num_classes):
    y_true = y_true.view(-1).long()
    y_pred = y_pred.view(-1).long()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    idx = y_true * num_classes + y_pred
    bc = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bc.reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def accuracy_top1(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()


@torch.no_grad()
def accuracy_topk(y_true, logits, k=2):
    topk = torch.topk(logits, k=k, dim=1).indices
    y_true = y_true.view(-1, 1)
    correct = (topk == y_true).any(dim=1)
    return correct.float().mean().item()


@torch.no_grad()
def per_class_prf_from_cm(cm, eps=1e-12):
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    macro_f1 = f1.mean().item()
    balanced_acc = recall.mean().item()

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc
    }


def logsumexp(x, dim=-1, keepdim=False):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    y = m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))
    return y if keepdim else y.squeeze(dim)


def cross_entropy_from_logits(logits, y):
    lse = logsumexp(logits, dim=1, keepdim=True)
    log_probs = logits - lse
    nll = -log_probs[torch.arange(logits.size(0), device=logits.device), y]
    return nll.mean()


@torch.no_grad()
def evaluate_model(loader, num_classes, device, model, topk=2):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_true, all_pred, all_logits = [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = cross_entropy_from_logits(logits, yb)

        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)

        preds = torch.argmax(logits, dim=1)
        all_true.append(yb.detach().cpu())
        all_pred.append(preds.detach().cpu())
        all_logits.append(logits.detach().cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    logits_all = torch.cat(all_logits, dim=0)

    cm = confusion_matrix_torch(y_true, y_pred, num_classes=num_classes)
    prf = per_class_prf_from_cm(cm)

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": accuracy_top1(y_true, y_pred),
        "top2_acc": accuracy_topk(y_true, logits_all, k=topk) if topk else None,
        "macro_f1": prf["macro_f1"],
        "balanced_acc": prf["balanced_acc"],
        "confusion_matrix": cm,
        "precision_per_class": prf["precision_per_class"],
        "recall_per_class": prf["recall_per_class"],
        "f1_per_class": prf["f1_per_class"],
    }


# ============================================================================
# CNN Model (PyTorch)
# ============================================================================
class CNNModel(nn.Module):
    """
    CNN for 64x64 grayscale images with 16 classes.
    
    Architecture:
    - Conv2d(1, c1, 3) -> ReLU -> MaxPool2d(2)  -> 32x32
    - Conv2d(c1, c2, 3) -> ReLU -> MaxPool2d(2) -> 16x16
    - Conv2d(c2, c3, 3) -> ReLU -> MaxPool2d(2) -> 8x8
    - Flatten -> Linear(c3*8*8, 256) -> ReLU -> Dropout -> Linear(256, 16)
    """
    def __init__(self, channels=(32, 64, 128), dropout=0.5):
        super().__init__()
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Input 64x64 -> after 3 pools: 8x8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# Training Functions
# ============================================================================
def train_cnn_one_run(train_loader, val_loader, cfg, device):
    """Train CNN with given config and early stopping."""
    model = CNNModel(channels=cfg["channels"], dropout=cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": [], "val_balanced_acc": [], "val_top2_acc": [],
        "sec_per_epoch": []
    }

    best_state = None
    best_score = -1.0
    best_epoch = -1
    patience = cfg.get("patience", 8)
    bad_epochs = 0

    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()

        loss_sum, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            loss_sum += float(loss.item()) * bs
            n_seen += bs

        train_loss = loss_sum / n_seen

        val_metrics = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, model=model)

        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_acc"].append(val_metrics["balanced_acc"])
        history["val_top2_acc"].append(val_metrics["top2_acc"])
        history["sec_per_epoch"].append(dt)

        score = val_metrics["macro_f1"]
        print(f"Epoch {ep:02d}/{cfg['epochs']} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f} | "
              f"val_macroF1={val_metrics['macro_f1']:.4f} | time={dt:.2f}s")

        # Early stopping on macro-F1
        if score > best_score:
            best_score = score
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs. "
                      f"Best epoch={best_epoch}, best_macroF1={best_score:.4f}")
                break

    # Load best weights
    model.load_state_dict(best_state)
    return model, history, best_score, best_epoch


def plot_learning_curves(history, title_prefix="CNN"):
    """Plot learning curves."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["val_acc"], marker="o")
    axes[1].set_title(f"{title_prefix}: Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    axes[2].plot(epochs, history["val_macro_f1"], marker="o")
    axes[2].set_title(f"{title_prefix}: Val Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(True)

    axes[3].plot(epochs, history["val_top2_acc"], marker="o")
    axes[3].set_title(f"{title_prefix}: Val Top-2 Accuracy")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Top-2 Acc")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(CHECKPOINTS_DIR / f"{title_prefix.lower()}_learning_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(cm, title, save_path=None):
    """Plot confusion matrix."""
    cm_np = cm.numpy() if isinstance(cm, torch.Tensor) else np.array(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_np, aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    print("=" * 60)
    print("CNN Training with Grid Search")
    print("=" * 60)

    # Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")

    mappings = torch.load(DATA_DIR / "class_mappings.pt")
    class_to_idx = mappings["class_to_idx"]
    idx_to_class = mappings["idx_to_class"]

    train_ds = SurfaceRoughnessDataset(df_train, image_size=IMAGE_SIZE)
    val_ds = SurfaceRoughnessDataset(df_val, image_size=IMAGE_SIZE)
    test_ds = SurfaceRoughnessDataset(df_test, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Grid search
    channels_grid = [(32, 64, 128), (64, 128, 256)]
    dropout_grid = [0.3, 0.5]
    lr_grid = [1e-3, 5e-4]
    wd_grid = [1e-4, 1e-5]

    runs = []
    best = {"score": -1.0, "cfg": None, "model": None, "history": None, "best_epoch": -1}

    run_id = 0
    total_runs = len(channels_grid) * len(dropout_grid) * len(lr_grid) * len(wd_grid)

    print(f"\nStarting grid search ({total_runs} runs)...")

    for channels in channels_grid:
        for dropout in dropout_grid:
            for lr in lr_grid:
                for wd in wd_grid:
                    run_id += 1
                    cfg = {
                        "channels": channels,
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": wd,
                        "epochs": 60,
                        "patience": 8,
                    }
                    print(f"\nRun {run_id}/{total_runs} | channels={channels} dropout={dropout} lr={lr} wd={wd}")

                    model, history, best_val_macro_f1, best_epoch = train_cnn_one_run(
                        train_loader, val_loader, cfg, device
                    )

                    val_metrics = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, model=model)
                    test_metrics = evaluate_model(test_loader, num_classes=NUM_CLASSES, device=device, model=model)

                    runs.append({
                        "run_id": run_id,
                        "channels": str(channels),
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": wd,
                        "epochs_run": len(history["sec_per_epoch"]),
                        "best_val_macro_f1": best_val_macro_f1,
                        "best_epoch": best_epoch,
                        "val_acc": val_metrics["acc"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "test_acc": test_metrics["acc"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "sec_total": sum(history["sec_per_epoch"]),
                    })

                    if best_val_macro_f1 > best["score"]:
                        best.update({
                            "score": best_val_macro_f1,
                            "cfg": cfg,
                            "model": model,
                            "history": history,
                            "best_epoch": best_epoch,
                            "val_metrics": val_metrics,
                            "test_metrics": test_metrics,
                        })

    # Save results
    results_df = pd.DataFrame(runs).sort_values(by="best_val_macro_f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(CHECKPOINTS_DIR / "results_cnn.csv", index=False)
    print(f"\nSaved grid results -> {CHECKPOINTS_DIR / 'results_cnn.csv'}")
    print("\nTop 5 runs:")
    print(results_df.head())

    # Best model info
    best_model = best["model"]
    best_val = best["val_metrics"]
    best_test = best["test_metrics"]

    print(f"\n{'='*60}")
    print("BEST CONFIG:", best["cfg"])
    print(f"Best epoch: {best['best_epoch']}")
    print("Best val metrics:", {k: best_val[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]})
    print("Best test metrics:", {k: best_test[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]})

    # Model stats
    cnn_params = sum(p.numel() for p in best_model.parameters())
    cnn_size_mb = cnn_params * 4 / (1024**2)
    print(f"\nCNN params: {cnn_params}")
    print(f"CNN model size (MB): {cnn_size_mb:.2f}")

    # Save checkpoint
    ckpt = {
        "state_dict": best_model.state_dict(),
        "config": best["cfg"],
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "history": best["history"],
        "val_confusion_matrix": best_val["confusion_matrix"],
        "test_confusion_matrix": best_test["confusion_matrix"],
        "val_metrics": {k: best_val[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]},
        "test_metrics": {k: best_test[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]},
        "best_epoch": best["best_epoch"],
    }
    ckpt_path = CHECKPOINTS_DIR / "cnn_best.pt"
    torch.save(ckpt, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")

    # Plot learning curves
    plot_learning_curves(best["history"], title_prefix="CNN")

    # Plot confusion matrices
    plot_confusion_matrix(best_val["confusion_matrix"], "CNN - Validation Confusion Matrix",
                          CHECKPOINTS_DIR / "cnn_val_cm.png")
    plot_confusion_matrix(best_test["confusion_matrix"], "CNN - Test Confusion Matrix",
                          CHECKPOINTS_DIR / "cnn_test_cm.png")

    print("\n" + "=" * 60)
    print("CNN training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
