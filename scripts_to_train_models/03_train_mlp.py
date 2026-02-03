#!/usr/bin/env python3
"""
03_train_mlp.py

Trains a Deep MLP model (from scratch with manual backpropagation)
with grid search for hyperparameter tuning on the Surface Roughness Classification dataset.

Architecture: Flatten -> [Linear -> Sigmoid -> Dropout] x K -> Linear -> logits

Outputs:
- checkpoints/mlp_deep_best.pt (best model checkpoint)
- results_mlp_deep.csv (grid search results)
- Learning curves and confusion matrix plots
"""

import os
import sys
import time
import math
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
        Initialize the dataset.
        
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

        # Load and convert to grayscale
        img = Image.open(path).convert("L")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        # Normalize to [0, 1] range
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
    idx = y_true * num_classes + y_pred
    bc = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bc.reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def accuracy_top1(y_true, y_pred):
    """
    Compute top-1 accuracy.
    
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
    Compute top-k accuracy.
    
    Args:
        y_true: Ground truth labels.
        logits: Raw model outputs.
        k: Number of top predictions to consider.
        
    Returns:
        float: Top-k accuracy.
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
    """
    Numerically stable log-sum-exp computation.
    
    Args:
        x: Input tensor.
        dim: Dimension to reduce.
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
    
    Args:
        logits: Raw model outputs of shape (batch_size, num_classes).
        y: Ground truth labels of shape (batch_size,).
        
    Returns:
        torch.Tensor: Mean cross-entropy loss.
    """
    lse = logsumexp(logits, dim=1, keepdim=True)
    log_probs = logits - lse
    nll = -log_probs[torch.arange(logits.size(0), device=logits.device), y]
    return nll.mean()


@torch.no_grad()
def evaluate_model(loader, num_classes, device, forward_fn, topk=2):
    """
    Evaluate model on a dataset and compute comprehensive metrics.
    
    Args:
        loader: DataLoader for the evaluation dataset.
        num_classes: Total number of classes.
        device: Device to run evaluation on.
        forward_fn: Function that takes input and returns logits.
        topk: K value for top-k accuracy.
        
    Returns:
        dict: Dictionary containing loss, accuracy, F1-scores, and confusion matrix.
    """
    total_loss = 0.0
    total_n = 0
    all_true, all_pred, all_logits = [], [], []

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
# MLP Model (from scratch with manual backpropagation)
# ============================================================================
def stable_softmax(logits):
    """
    Numerically stable softmax computation.
    
    Subtracts max value before exponentiating to prevent overflow.
    
    Args:
        logits: Raw model outputs of shape (batch_size, num_classes).
        
    Returns:
        torch.Tensor: Softmax probabilities.
    """
    m, _ = torch.max(logits, dim=1, keepdim=True)
    exps = torch.exp(logits - m)
    return exps / torch.sum(exps, dim=1, keepdim=True)


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


def center_input(x_img):
    """
    Transform input from [0,1] to [-1,1] range.
    
    Centering input around zero can help with gradient flow
    and faster convergence in neural networks.
    
    Args:
        x_img: Input tensor with values in [0,1].
        
    Returns:
        torch.Tensor: Centered input with values in [-1,1].
    """
    return x_img * 2.0 - 1.0


class LinearScratch:
    """
    Linear (fully connected) layer with manual forward/backward passes.
    
    Implements y = x @ W + b with Xavier initialization and stores
    activations for backpropagation.
    
    Attributes:
        W: Weight matrix of shape (in_dim, out_dim).
        b: Bias vector of shape (out_dim,).
        x: Cached input for backward pass.
        dW, db: Computed gradients.
    """
    
    def __init__(self, in_dim, out_dim, seed, device):
        """
        Initialize layer with Xavier uniform initialization.
        
        Args:
            in_dim: Number of input features.
            out_dim: Number of output features.
            seed: Random seed for reproducibility.
            device: Device to place tensors on.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        # Xavier uniform initialization for better gradient flow
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.W = (torch.empty(in_dim, out_dim).uniform_(-limit, limit, generator=g)).to(device)
        self.b = torch.zeros(out_dim, device=device)
        self.x = None  # Cache for backward pass
        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)

    def forward(self, x):
        """
        Forward pass: compute linear transformation.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim).
            
        Returns:
            torch.Tensor: Output of shape (batch_size, out_dim).
        """
        self.x = x  # Cache input for backward pass
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Backward pass: compute gradients w.r.t. input, weights, and bias.
        
        Args:
            dout: Gradient from upstream of shape (batch_size, out_dim).
            
        Returns:
            torch.Tensor: Gradient w.r.t. input of shape (batch_size, in_dim).
        """
        x = self.x
        self.dW = x.t() @ dout  # Gradient for weights
        self.db = dout.sum(dim=0)  # Gradient for bias
        dx = dout @ self.W.t()  # Gradient for input
        return dx

    def sgd_step(self, lr, weight_decay):
        """
        Update parameters using SGD with optional L2 regularization.
        
        Args:
            lr: Learning rate.
            weight_decay: L2 regularization coefficient.
        """
        if weight_decay > 0:
            self.dW = self.dW + weight_decay * self.W
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db


class SigmoidScratch:
    """
    Sigmoid activation function with manual forward/backward passes.
    
    Implements s = 1 / (1 + exp(-z)) and its derivative s * (1 - s).
    """
    
    def __init__(self):
        """Initialize sigmoid layer."""
        self.s = None  # Cache for backward pass

    def forward(self, z):
        """
        Forward pass: apply sigmoid activation.
        
        Args:
            z: Input tensor.
            
        Returns:
            torch.Tensor: Sigmoid output in range (0, 1).
        """
        s = 1.0 / (1.0 + torch.exp(-z))
        self.s = s  # Cache for backward
        return s

    def backward(self, dout):
        """
        Backward pass: compute gradient through sigmoid.
        
        Uses the cached sigmoid output: dsigmoid/dz = s * (1 - s).
        
        Args:
            dout: Gradient from upstream.
            
        Returns:
            torch.Tensor: Gradient w.r.t. input.
        """
        s = self.s
        return dout * s * (1.0 - s)


class DropoutScratch:
    """
    Inverted dropout layer implemented from scratch.
    
    During training, randomly zeros out elements and scales remaining
    by 1/(1-p) to maintain expected values. No scaling needed at inference.
    
    Attributes:
        p: Dropout probability (fraction of elements to zero).
        mask: Binary mask used during forward pass.
        training: Whether in training mode.
    """
    
    def __init__(self, p):
        """
        Initialize dropout layer.
        
        Args:
            p: Dropout probability (0 <= p < 1).
        """
        assert 0.0 <= p < 1.0
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        """
        Forward pass: apply dropout if training.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output with dropped units (if training).
        """
        if (not self.training) or self.p == 0.0:
            self.mask = None
            return x
        keep_prob = 1.0 - self.p
        # Generate random mask and scale to maintain expected value
        self.mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        return x * self.mask / keep_prob

    def backward(self, dout):
        """
        Backward pass: apply same mask to gradient.
        
        Args:
            dout: Gradient from upstream.
            
        Returns:
            torch.Tensor: Gradient with same dropout pattern applied.
        """
        if self.mask is None:
            return dout
        keep_prob = 1.0 - self.p
        return dout * self.mask / keep_prob


class MLPScratch:
    """
    Deep Multi-Layer Perceptron implemented from scratch.
    
    Architecture: Flatten -> [Linear -> Sigmoid -> Dropout] x K -> Linear -> logits
    
    This implementation uses manual forward and backward passes without
    PyTorch autograd, demonstrating the fundamentals of backpropagation.
    
    Attributes:
        hidden_sizes: List of hidden layer dimensions.
        dropout_p: Dropout probability between layers.
        layers: List of layer objects (Linear, Sigmoid, Dropout).
        out: Final output linear layer.
    """
    
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout_p, seed, device):
        """
        Build the MLP architecture.
        
        Args:
            input_dim: Flattened input dimension (e.g., 64*64 for images).
            hidden_sizes: Tuple/list of hidden layer sizes.
            num_classes: Number of output classes.
            dropout_p: Dropout probability.
            seed: Random seed for weight initialization.
            device: Device to place tensors on.
        """
        self.hidden_sizes = list(hidden_sizes)
        self.dropout_p = float(dropout_p)
        self.layers = []
        cur = input_dim
        s = seed

        # Build hidden layers: Linear -> Sigmoid -> Dropout
        for h in self.hidden_sizes:
            self.layers.append(LinearScratch(cur, h, seed=s, device=device))
            self.layers.append(SigmoidScratch())
            self.layers.append(DropoutScratch(self.dropout_p))
            cur = h
            s += 1

        # Output layer (no activation - logits)
        self.out = LinearScratch(cur, num_classes, seed=s, device=device)

    def set_training(self, mode):
        """
        Set training mode for dropout layers.
        
        Args:
            mode: True for training, False for evaluation.
        """
        for layer in self.layers:
            if isinstance(layer, DropoutScratch):
                layer.training = mode

    def forward_logits(self, x_img):
        """
        Forward pass: compute logits from input images.
        
        Args:
            x_img: Input images of shape (batch_size, 1, H, W).
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes).
        """
        x_img = center_input(x_img)  # Normalize to [-1, 1]
        x = x_img.view(x_img.size(0), -1)  # Flatten
        for layer in self.layers:
            x = layer.forward(x)
        logits = self.out.forward(x)
        return logits

    def train_step(self, x_img, y, lr, weight_decay):
        """
        Perform one training step with manual backpropagation.
        
        Computes forward pass, loss, backward pass, and parameter updates.
        
        Args:
            x_img: Input batch of images.
            y: Target labels.
            lr: Learning rate.
            weight_decay: L2 regularization coefficient.
            
        Returns:
            float: Loss value for this batch.
        """
        self.set_training(True)
        B = x_img.size(0)

        # Forward pass
        x_img = center_input(x_img)
        x = x_img.view(B, -1)
        for layer in self.layers:
            x = layer.forward(x)
        logits = self.out.forward(x)

        # Compute loss
        probs = stable_softmax(logits)
        p = probs[torch.arange(B, device=probs.device), y]
        loss = (-torch.log(p + 1e-12)).mean()

        # Backward pass: compute gradients
        dlogits = (probs - one_hot(y, NUM_CLASSES)) / B
        dx = self.out.backward(dlogits)

        # Backpropagate through all hidden layers in reverse order
        for layer in reversed(self.layers):
            dx = layer.backward(dx)

        # SGD parameter updates
        for layer in self.layers:
            if isinstance(layer, LinearScratch):
                layer.sgd_step(lr=lr, weight_decay=weight_decay)
        self.out.sgd_step(lr=lr, weight_decay=weight_decay)

        return float(loss.item())

    def get_state(self):
        """
        Extract model state for checkpointing.
        
        Returns:
            dict: Dictionary containing all layer weights and configuration.
        """
        state = {
            "hidden_sizes": self.hidden_sizes,
            "dropout_p": self.dropout_p,
            "layers": [],
            "out": {}
        }
        for layer in self.layers:
            if isinstance(layer, LinearScratch):
                state["layers"].append({"type": "linear", "W": layer.W.detach().cpu(), "b": layer.b.detach().cpu()})
            elif isinstance(layer, SigmoidScratch):
                state["layers"].append({"type": "sigmoid"})
            elif isinstance(layer, DropoutScratch):
                state["layers"].append({"type": "dropout", "p": layer.p})
        state["out"] = {"W": self.out.W.detach().cpu(), "b": self.out.b.detach().cpu()}
        state["input_preprocess"] = "x in [0,1] -> [-1,1] via (2x-1)"
        return state


# ============================================================================
# Training Functions
# ============================================================================
def train_one_run(train_loader, val_loader, hidden_sizes, dropout_p, epochs, lr, weight_decay, seed, device):
    """
    Train a Deep MLP model for a specified number of epochs.
    
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        hidden_sizes: Tuple of hidden layer dimensions.
        dropout_p: Dropout probability.
        epochs: Number of training epochs.
        lr: Learning rate for SGD.
        weight_decay: L2 regularization coefficient.
        seed: Random seed for model initialization.
        device: Device to run training on.
        
    Returns:
        tuple: (model, history, total_time, best_val_macro_f1)
    """
    model = MLPScratch(D, hidden_sizes, NUM_CLASSES, dropout_p=dropout_p, seed=seed, device=device)

    # Track training metrics
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

        # Evaluate on validation set (disable dropout)
        model.set_training(False)
        forward_fn = lambda x: model.forward_logits(x)
        val_metrics = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, forward_fn=forward_fn)

        dt = time.time() - t0

        # Record metrics
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


def plot_learning_curves(history, title_prefix="MLP"):
    """
    Plot training and validation metrics over epochs.
    
    Creates a 4-panel figure showing loss, accuracy, macro-F1, and top-2
    accuracy curves to visualize training progress.
    
    Args:
        history: Dictionary containing lists of metrics per epoch.
        title_prefix: Prefix for plot titles and output filename.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Loss curves
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

    # Macro-F1 score
    axes[2].plot(epochs, history["val_macro_f1"], marker="o")
    axes[2].set_title(f"{title_prefix}: Val Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(True)

    # Top-2 accuracy
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
    Main entry point for Deep MLP training with grid search.
    
    Performs hyperparameter grid search over architecture (hidden sizes),
    dropout, learning rate, weight decay, and epochs. Saves the best model
    checkpoint and generates visualization plots.
    """
    print("=" * 60)
    print("Deep MLP Training with Grid Search")
    print("=" * 60)

    # Setup reproducibility and device
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    print("\nLoading data...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")

    # Load class mappings
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
    hidden_grid = [
        (512, 256),
        (512, 512, 256),
        (1024, 512),
        (1024, 512, 256),
    ]
    dropout_grid = [0.0, 0.3, 0.5]
    lr_grid = [1e-3, 3e-4]
    wd_grid = [0.0, 1e-4]
    epochs_grid = [40, 60]

    runs = []
    best = {"score": -1.0, "cfg": None, "model": None, "history": None}

    run_id = 0
    total_runs = len(hidden_grid) * len(dropout_grid) * len(lr_grid) * len(wd_grid) * len(epochs_grid)

    print(f"\nStarting grid search ({total_runs} runs)...")

    for hidden_sizes in hidden_grid:
        for dropout_p in dropout_grid:
            for lr in lr_grid:
                for wd in wd_grid:
                    for epochs in epochs_grid:
                        run_id += 1
                        print(f"\nRun {run_id}/{total_runs} | hidden={hidden_sizes} dropout={dropout_p} epochs={epochs} lr={lr} wd={wd}")

                        model, history, total_time, best_val_macro_f1 = train_one_run(
                            train_loader, val_loader, hidden_sizes, dropout_p, epochs, lr, wd, seed=SEED, device=device
                        )

                        model.set_training(False)
                        forward_fn = lambda x: model.forward_logits(x)
                        val_last = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, forward_fn=forward_fn)

                        runs.append({
                            "run_id": run_id,
                            "hidden_sizes": str(hidden_sizes),
                            "dropout": dropout_p,
                            "epochs": epochs,
                            "lr": lr,
                            "weight_decay": wd,
                            "best_val_macro_f1": best_val_macro_f1,
                            "val_macro_f1_last": val_last["macro_f1"],
                            "val_acc_last": val_last["acc"],
                            "sec_total": total_time,
                        })

                        if best_val_macro_f1 > best["score"]:
                            best.update({
                                "score": best_val_macro_f1,
                                "cfg": {
                                    "hidden_sizes": hidden_sizes,
                                    "dropout": dropout_p,
                                    "epochs": epochs,
                                    "lr": lr,
                                    "weight_decay": wd,
                                    "seed": SEED,
                                    "image_size": IMAGE_SIZE,
                                    "input_preprocess": "x in [0,1] -> [-1,1] via (2x-1)"
                                },
                                "model": model,
                                "history": history
                            })

    # Save results
    results_df = pd.DataFrame(runs).sort_values(by="best_val_macro_f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(CHECKPOINTS_DIR / "results_mlp_deep.csv", index=False)
    print(f"\nSaved grid results -> {CHECKPOINTS_DIR / 'results_mlp_deep.csv'}")
    print("\nTop 5 runs:")
    print(results_df.head())

    # Evaluate best model
    best_model = best["model"]
    best_model.set_training(False)
    best_forward = lambda x: best_model.forward_logits(x)

    best_val = evaluate_model(val_loader, num_classes=NUM_CLASSES, device=device, forward_fn=best_forward)
    best_test = evaluate_model(test_loader, num_classes=NUM_CLASSES, device=device, forward_fn=best_forward)

    print(f"\n{'='*60}")
    print("BEST CONFIG:", best["cfg"])
    print("Best val metrics:", {k: best_val[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]})
    print("Best test metrics:", {k: best_test[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]})

    # Save checkpoint
    ckpt = {
        "state": best_model.get_state(),
        "config": best["cfg"],
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "history": best["history"],
        "val_confusion_matrix": best_val["confusion_matrix"],
        "test_confusion_matrix": best_test["confusion_matrix"],
        "val_metrics": {k: best_val[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]},
        "test_metrics": {k: best_test[k] for k in ["loss", "acc", "macro_f1", "balanced_acc", "top2_acc"]},
    }
    ckpt_path = CHECKPOINTS_DIR / "mlp_deep_best.pt"
    torch.save(ckpt, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")

    # Plot learning curves
    plot_learning_curves(best["history"], title_prefix="MLP")

    # Plot confusion matrices
    plot_confusion_matrix(best_val["confusion_matrix"], "MLP - Validation Confusion Matrix",
                          CHECKPOINTS_DIR / "mlp_val_cm.png")
    plot_confusion_matrix(best_test["confusion_matrix"], "MLP - Test Confusion Matrix",
                          CHECKPOINTS_DIR / "mlp_test_cm.png")

    print("\n" + "=" * 60)
    print("Deep MLP training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
