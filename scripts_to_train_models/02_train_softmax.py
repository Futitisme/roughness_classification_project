#!/usr/bin/env python3
"""
02_train_softmax.py

Trains a Softmax Regression model (from scratch) with grid search
for hyperparameter tuning on the Surface Roughness Classification dataset.

Outputs:
- checkpoints/softmax_best.pt (best model checkpoint)
- results_softmax.csv (grid search results)
- Learning curves and confusion matrix plots
"""

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
IMAGE_SIZE = 64
NUM_CLASSES = 16
D = IMAGE_SIZE * IMAGE_SIZE  # input dimension

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
    """
    PyTorch Dataset for loading and preprocessing surface roughness images.
    
    Handles image loading, grayscale conversion, resizing, and normalization.
    """
    
    def __init__(self, df, image_size=64):
        """
        Initialize the dataset with a DataFrame containing image paths and labels.
        
        Args:
            df: DataFrame with 'path' and 'label' columns.
            image_size: Target size for image resizing (default: 64x64).
        """
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image sample.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            tuple: (image_tensor, label) where image is (1, H, W) float32.
        """
        row = self.df.iloc[idx]
        path = row["path"]
        y = int(row["label"])

        # Load image and convert to grayscale
        img = Image.open(path).convert("L")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        # Normalize pixel values to [0, 1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ============================================================================
# Metrics (from scratch)
# ============================================================================
@torch.no_grad()
def confusion_matrix_torch(y_true, y_pred, num_classes):
    """
    Compute confusion matrix using PyTorch operations.
    
    Args:
        y_true: Ground truth labels tensor.
        y_pred: Predicted labels tensor.
        num_classes: Total number of classes.
        
    Returns:
        torch.Tensor: Confusion matrix of shape (num_classes, num_classes).
    """
    y_true = y_true.view(-1).long()
    y_pred = y_pred.view(-1).long()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    # Flatten 2D index to 1D for efficient counting
    idx = y_true * num_classes + y_pred
    bc = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bc.reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def accuracy_top1(y_true, y_pred):
    """
    Compute top-1 accuracy (standard accuracy).
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        float: Accuracy as a value between 0 and 1.
    """
    return (y_true == y_pred).float().mean().item()


@torch.no_grad()
def accuracy_topk(y_true, logits, k=2):
    """
    Compute top-k accuracy (correct if true label is in top k predictions).
    
    Args:
        y_true: Ground truth labels.
        logits: Raw model outputs (before softmax).
        k: Number of top predictions to consider.
        
    Returns:
        float: Top-k accuracy as a value between 0 and 1.
    """
    topk = torch.topk(logits, k=k, dim=1).indices
    y_true = y_true.view(-1, 1)
    correct = (topk == y_true).any(dim=1)
    return correct.float().mean().item()


@torch.no_grad()
def per_class_prf_from_cm(cm, eps=1e-12):
    """
    Compute per-class precision, recall, F1-score from confusion matrix.
    
    Args:
        cm: Confusion matrix tensor.
        eps: Small constant to avoid division by zero.
        
    Returns:
        dict: Contains per-class metrics and macro-averaged scores.
    """
    cm = cm.to(torch.float32)
    # Extract true positives from diagonal
    tp = torch.diag(cm)
    # False positives: column sum minus diagonal
    fp = cm.sum(dim=0) - tp
    # False negatives: row sum minus diagonal
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Macro-averaged metrics (unweighted mean across classes)
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
    """
    Numerically stable log-sum-exp computation.
    
    Computes log(sum(exp(x))) in a numerically stable way by subtracting
    the maximum value before exponentiating.
    
    Args:
        x: Input tensor.
        dim: Dimension along which to compute.
        keepdim: Whether to keep the reduced dimension.
        
    Returns:
        torch.Tensor: Log-sum-exp result.
    """
    m, _ = torch.max(x, dim=dim, keepdim=True)
    y = m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))
    return y if keepdim else y.squeeze(dim)


def cross_entropy_from_logits(logits, y):
    """
    Compute cross-entropy loss from raw logits.
    
    Implements numerically stable cross-entropy using log-softmax.
    
    Args:
        logits: Raw model outputs of shape (batch_size, num_classes).
        y: Ground truth labels of shape (batch_size,).
        
    Returns:
        torch.Tensor: Mean cross-entropy loss.
    """
    lse = logsumexp(logits, dim=1, keepdim=True)
    log_probs = logits - lse
    nll = -log_probs[torch.arange(logits.size(0)), y]
    return nll.mean()


@torch.no_grad()
def evaluate_model(loader, num_classes, device, forward_fn, topk=2):
    """
    Evaluate model on a dataset and compute comprehensive metrics.
    
    Args:
        loader: DataLoader for the evaluation dataset.
        num_classes: Total number of classes.
        device: Device to run evaluation on (CPU/GPU).
        forward_fn: Function that takes input and returns logits.
        topk: K value for top-k accuracy computation.
        
    Returns:
        dict: Dictionary containing loss, accuracy, F1-scores, and confusion matrix.
    """
    total_loss = 0.0
    total_n = 0
    all_true, all_pred, all_logits = [], [], []

    # Iterate through all batches
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = forward_fn(xb)
        loss = cross_entropy_from_logits(logits, yb)

        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)

        preds = torch.argmax(logits, dim=1)
        all_true.append(yb.detach().cpu())
        all_pred.append(preds.detach().cpu())
        all_logits.append(logits.detach().cpu())

    # Concatenate all batches
    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    logits_all = torch.cat(all_logits, dim=0)

    # Compute metrics
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
# Softmax Regression Model (from scratch)
# ============================================================================
def stable_softmax(logits):
    """
    Numerically stable softmax computation.
    
    Subtracts max value before exponentiating to prevent overflow.
    
    Args:
        logits: Raw model outputs of shape (batch_size, num_classes).
        
    Returns:
        torch.Tensor: Softmax probabilities of same shape as input.
    """
    m, _ = torch.max(logits, dim=1, keepdim=True)
    exps = torch.exp(logits - m)
    return exps / torch.sum(exps, dim=1, keepdim=True)


def cross_entropy_from_probs(probs, y, eps=1e-12):
    """
    Compute cross-entropy loss from probability distribution.
    
    Args:
        probs: Softmax probabilities of shape (batch_size, num_classes).
        y: Ground truth labels of shape (batch_size,).
        eps: Small constant to avoid log(0).
        
    Returns:
        torch.Tensor: Mean cross-entropy loss.
    """
    p = probs[torch.arange(probs.size(0), device=probs.device), y]
    return (-torch.log(p + eps)).mean()


def one_hot(y, num_classes):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        y: Integer labels of shape (batch_size,).
        num_classes: Total number of classes.
        
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (batch_size, num_classes).
    """
    oh = torch.zeros((y.size(0), num_classes), device=y.device, dtype=torch.float32)
    oh[torch.arange(y.size(0), device=y.device), y] = 1.0
    return oh


class SoftmaxRegressionScratch:
    """
    Softmax Regression (Multinomial Logistic Regression) implemented from scratch.
    
    A linear classifier that maps flattened images directly to class logits.
    Training uses manual gradient computation and SGD optimization.
    
    Attributes:
        W: Weight matrix of shape (input_dim, num_classes).
        b: Bias vector of shape (num_classes,).
    """
    
    def __init__(self, input_dim, num_classes, seed, device):
        """
        Initialize model weights with small random values.
        
        Args:
            input_dim: Number of input features (flattened image size).
            num_classes: Number of output classes.
            seed: Random seed for reproducibility.
            device: Device to place tensors on (CPU/GPU).
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        # Initialize weights with small random values to break symmetry
        self.W = (torch.randn(input_dim, num_classes, generator=g) * 0.01).to(device)
        self.b = torch.zeros(num_classes, device=device)

    def forward_logits(self, x):
        """
        Compute raw logits (unnormalized scores) for input images.
        
        Args:
            x: Input tensor of shape (batch_size, 1, H, W).
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        X = x.view(x.size(0), -1)  # Flatten to (batch_size, input_dim)
        return X @ self.W + self.b

    def train_step(self, x, y, lr, weight_decay):
        """
        Perform one training step with manual gradient computation.
        
        Implements forward pass, loss computation, backward pass (gradient
        computation), and parameter update using SGD.
        
        Args:
            x: Input batch of shape (batch_size, 1, H, W).
            y: Target labels of shape (batch_size,).
            lr: Learning rate.
            weight_decay: L2 regularization strength.
            
        Returns:
            float: Loss value for this batch.
        """
        B = x.size(0)
        X = x.view(B, -1)  # Flatten input
        
        # Forward pass
        logits = X @ self.W + self.b
        probs = stable_softmax(logits)
        loss = cross_entropy_from_probs(probs, y)

        # Backward pass: compute gradients analytically
        # Gradient of cross-entropy w.r.t. logits is (probs - one_hot)
        dlogits = (probs - one_hot(y, NUM_CLASSES)) / B
        dW = X.t() @ dlogits  # Gradient for weights
        db = dlogits.sum(dim=0)  # Gradient for bias

        # Add L2 regularization gradient
        if weight_decay > 0:
            dW = dW + weight_decay * self.W

        # SGD parameter update
        self.W = self.W - lr * dW
        self.b = self.b - lr * db

        return float(loss.item())


# ============================================================================
# Training Functions
# ============================================================================
def train_one_run(train_loader, val_loader, epochs, lr, weight_decay, seed, device):
    """
    Train a Softmax Regression model for a specified number of epochs.
    
    Performs full training loop including forward pass, loss computation,
    gradient calculation, and parameter updates for each batch.
    
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        lr: Learning rate for SGD.
        weight_decay: L2 regularization coefficient.
        seed: Random seed for model initialization.
        device: Device to run training on (CPU/GPU).
        
    Returns:
        tuple: (model, history, total_time, best_val_macro_f1) containing
            trained model, training history dict, total training time,
            and best validation macro-F1 score achieved.
    """
    model = SoftmaxRegressionScratch(D, NUM_CLASSES, seed=seed, device=device)

    # Dictionary to track training metrics over epochs
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_balanced_acc": [],
        "val_top2_acc": [],
        "sec_per_epoch": [],
    }

    t_total0 = time.time()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0

        # Training loop over batches
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            l = model.train_step(xb, yb, lr=lr, weight_decay=weight_decay)
            bs = xb.size(0)
            loss_sum += l * bs
            n_seen += bs

        train_loss = loss_sum / n_seen

        # Evaluate on validation set
        forward_fn = lambda x: model.forward_logits(x)
        val_metrics = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, forward_fn=forward_fn)

        dt = time.time() - t0

        # Record metrics for this epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_acc"].append(val_metrics["balanced_acc"])
        history["val_top2_acc"].append(val_metrics["top2_acc"])
        history["sec_per_epoch"].append(dt)

    total_time = time.time() - t_total0
    best_val_macro_f1 = max(history["val_macro_f1"])

    return model, history, total_time, best_val_macro_f1


def plot_learning_curves(history, title_prefix="Softmax"):
    """
    Plot training and validation metrics over epochs.
    
    Creates a 4-panel figure showing loss, accuracy, macro-F1, and top-2
    accuracy curves to visualize training progress and detect overfitting.
    
    Args:
        history: Dictionary containing lists of metrics per epoch.
        title_prefix: Prefix for plot titles and output filename.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Loss curves (train vs validation)
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title(f"{title_prefix}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Validation accuracy
    axes[1].plot(epochs, history["val_acc"], marker="o")
    axes[1].set_title(f"{title_prefix}: Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    # Macro-F1 score (important for imbalanced classes)
    axes[2].plot(epochs, history["val_macro_f1"], marker="o")
    axes[2].set_title(f"{title_prefix}: Val Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(True)

    # Top-2 accuracy (useful for fine-grained classification)
    axes[3].plot(epochs, history["val_top2_acc"], marker="o")
    axes[3].set_title(f"{title_prefix}: Val Top-2 Accuracy")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Top-2 Acc")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(CHECKPOINTS_DIR / f"{title_prefix.lower()}_learning_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(cm, title, save_path=None):
    """
    Visualize confusion matrix as a heatmap.
    
    Displays the confusion matrix with color intensity representing
    the number of predictions for each true/predicted class pair.
    
    Args:
        cm: Confusion matrix (torch.Tensor or numpy array).
        title: Title for the plot.
        save_path: Optional path to save the figure.
    """
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
    """
    Main entry point for Softmax Regression training with grid search.
    
    Performs hyperparameter grid search over learning rates, weight decay,
    and number of epochs. Saves the best model checkpoint and generates
    visualization plots.
    """
    print("=" * 60)
    print("Softmax Regression Training with Grid Search")
    print("=" * 60)

    # Setup reproducibility and device
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data from data preparation script
    print("\nLoading data...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")

    # Load class mappings for label interpretation
    mappings = torch.load(DATA_DIR / "class_mappings.pt")
    class_to_idx = mappings["class_to_idx"]
    idx_to_class = mappings["idx_to_class"]

    # Create datasets and dataloaders
    train_ds = SurfaceRoughnessDataset(df_train, image_size=IMAGE_SIZE)
    val_ds = SurfaceRoughnessDataset(df_val, image_size=IMAGE_SIZE)
    test_ds = SurfaceRoughnessDataset(df_test, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Grid search
    grid_epochs = [5, 10, 20, 35]
    grid_lr = [1e-2, 3e-3, 1e-3]
    grid_wd = [0.0, 1e-4, 1e-3]

    runs = []
    best = {"score": -1.0, "cfg": None, "model": None, "history": None}

    run_id = 0
    total_runs = len(grid_epochs) * len(grid_lr) * len(grid_wd)

    print(f"\nStarting grid search ({total_runs} runs)...")

    for epochs in grid_epochs:
        for lr in grid_lr:
            for wd in grid_wd:
                run_id += 1
                print(f"\nRun {run_id}/{total_runs} | epochs={epochs} lr={lr} wd={wd}")

                model, history, total_time, best_val_macro_f1 = train_one_run(
                    train_loader, val_loader, epochs, lr, wd, seed=SEED, device=device
                )

                forward_fn = lambda x: model.forward_logits(x)
                val_metrics_last = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, forward_fn=forward_fn)

                runs.append({
                    "run_id": run_id,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": wd,
                    "best_val_macro_f1": best_val_macro_f1,
                    "val_macro_f1_last": val_metrics_last["macro_f1"],
                    "val_acc_last": val_metrics_last["acc"],
                    "val_balanced_acc_last": val_metrics_last["balanced_acc"],
                    "val_loss_last": val_metrics_last["loss"],
                    "sec_total": total_time,
                })

                if best_val_macro_f1 > best["score"]:
                    best.update({
                        "score": best_val_macro_f1,
                        "cfg": {"epochs": epochs, "lr": lr, "weight_decay": wd, "seed": SEED, "image_size": IMAGE_SIZE},
                        "model": model,
                        "history": history,
                    })

    # Save results
    results_df = pd.DataFrame(runs).sort_values(by="best_val_macro_f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(CHECKPOINTS_DIR / "results_softmax.csv", index=False)
    print(f"\nSaved grid results -> {CHECKPOINTS_DIR / 'results_softmax.csv'}")
    print("\nTop 5 runs:")
    print(results_df.head())

    # Evaluate best model
    best_model = best["model"]
    best_forward = lambda x: best_model.forward_logits(x)

    best_val = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, forward_fn=best_forward)
    best_test = evaluate_model(test_loader, num_classes=NUM_CLASSES, device=device, forward_fn=best_forward)

    print(f"\n{'='*60}")
    print("BEST CONFIG:", best["cfg"])
    print("Best val metrics:", {k: best_val[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]})
    print("Best test metrics:", {k: best_test[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]})

    # Save checkpoint
    ckpt = {
        "W": best_model.W.detach().cpu(),
        "b": best_model.b.detach().cpu(),
        "config": best["cfg"],
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "history": best["history"],
        "val_confusion_matrix": best_val["confusion_matrix"],
        "test_confusion_matrix": best_test["confusion_matrix"],
        "val_metrics": {k: best_val[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]},
        "test_metrics": {k: best_test[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]},
    }
    ckpt_path = CHECKPOINTS_DIR / "softmax_best.pt"
    torch.save(ckpt, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")

    # Plot learning curves
    plot_learning_curves(best["history"], title_prefix="Softmax")

    # Plot confusion matrices
    plot_confusion_matrix(best_val["confusion_matrix"], "Softmax - Validation Confusion Matrix",
                          CHECKPOINTS_DIR / "softmax_val_cm.png")
    plot_confusion_matrix(best_test["confusion_matrix"], "Softmax - Test Confusion Matrix",
                          CHECKPOINTS_DIR / "softmax_test_cm.png")

    print("\n" + "=" * 60)
    print("Softmax Regression training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
