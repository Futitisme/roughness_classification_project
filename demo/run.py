#!/usr/bin/env python3
"""
run.py - Surface Roughness Classification Demo

This script demonstrates the trained CNN model for surface roughness classification.
It loads a pre-trained model and runs inference on sample images.

Usage:
    python3 run.py

Author: Neural Networks Project
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "cnn_final.pt"
SAMPLE_IMAGES_DIR = SCRIPT_DIR / "sample_images"

IMAGE_SIZE = 64
NUM_CLASSES = 16

# Class names (surface roughness values in micrometers)
CLASS_NAMES = ['00', '03', '06', '09', '12', '15', '18', '21', '24', '27', '30', '33', '36', '39', '42', '45']


# ============================================================================
# CNN Model Definition (same as training)
# ============================================================================
class CNNModel(nn.Module):
    """
    CNN for 64x64 grayscale images with 16 classes.
    Architecture: 3x(Conv2d -> ReLU -> MaxPool2d) -> Linear -> ReLU -> Dropout -> Linear
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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================================
# Helper Functions
# ============================================================================
def load_image(path, image_size=IMAGE_SIZE):
    """Load and preprocess a single image."""
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def get_prediction(model, image_tensor, device):
    """Get model prediction for an image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        # Get top-3 predictions
        top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)
        top3 = [(CLASS_NAMES[idx.item()], prob.item()) for idx, prob in zip(top3_indices[0], top3_probs[0])]
        
    return pred_idx, confidence, top3


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("   SURFACE ROUGHNESS CLASSIFICATION - CNN MODEL DEMO")
    print("=" * 70)
    print("\nTask: Classify surface roughness from microscope images")
    print("Classes: 16 roughness levels (00, 03, 06, ... 45 micrometers)")
    print("Model: CNN with 3 convolutional layers")
    print("-" * 70)


def print_model_info(checkpoint):
    """Print model information."""
    config = checkpoint.get("config", {})
    val_metrics = checkpoint.get("val_metrics", {})
    test_metrics = checkpoint.get("test_metrics", {})
    
    print("\n[MODEL INFORMATION]")
    print(f"  Architecture: {config.get('channels', 'N/A')}")
    print(f"  Parameters: {checkpoint.get('num_params', 'N/A'):,}")
    print(f"  Model size: {checkpoint.get('model_size_mb', 'N/A'):.2f} MB")
    print(f"  Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
    
    print("\n[TRAINING RESULTS]")
    print(f"  Validation Accuracy:  {val_metrics.get('acc', 0)*100:.2f}%")
    print(f"  Validation Macro-F1:  {val_metrics.get('macro_f1', 0)*100:.2f}%")
    print(f"  Test Accuracy:        {test_metrics.get('acc', 0)*100:.2f}%")
    print(f"  Test Macro-F1:        {test_metrics.get('macro_f1', 0)*100:.2f}%")
    print(f"  Test Top-2 Accuracy:  {test_metrics.get('top2_acc', 0)*100:.2f}%")


def run_demo():
    """Main demo function."""
    print_header()
    
    # Check model file
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model file not found: {MODEL_PATH}")
        print("Please run the training script first:")
        print("  cd scripts && python 01_data_preparation.py && python 05_train_cnn_best.py")
        sys.exit(1)
    
    # Check sample images
    if not SAMPLE_IMAGES_DIR.exists():
        print(f"\n[ERROR] Sample images directory not found: {SAMPLE_IMAGES_DIR}")
        sys.exit(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Using: {device}")
    
    # Load model
    print(f"\n[LOADING] Model from: {MODEL_PATH.name}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {"channels": (32, 64, 128), "dropout": 0.5})
    model = CNNModel(channels=config["channels"], dropout=config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("[LOADED] Model loaded successfully!")
    
    # Print model info
    print_model_info(checkpoint)
    
    # Get sample images (skip hidden macOS files starting with ._)
    image_files = sorted([f for f in SAMPLE_IMAGES_DIR.glob("*.jpg") if not f.name.startswith(".")])
    if not image_files:
        print(f"\n[ERROR] No .jpg images found in {SAMPLE_IMAGES_DIR}")
        sys.exit(1)
    
    print(f"\n[DEMO] Running inference on {len(image_files)} sample images...")
    print("-" * 70)
    
    # Run inference
    correct = 0
    total = 0
    results = []
    
    total_time = 0
    
    for img_path in image_files:
        # Extract true class from filename (e.g., "class_09_sample1.jpg" -> "09")
        filename = img_path.stem
        parts = filename.split("_")
        true_class = parts[1] if len(parts) >= 2 else "??"
        
        # Load and predict
        img_tensor = load_image(img_path)
        
        start_time = time.perf_counter()
        pred_idx, confidence, top3 = get_prediction(model, img_tensor, device)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        total_time += inference_time
        
        pred_class = CLASS_NAMES[pred_idx]
        is_correct = pred_class == true_class
        
        if true_class in CLASS_NAMES:
            total += 1
            if is_correct:
                correct += 1
        
        status = "✓" if is_correct else "✗"
        results.append({
            "file": img_path.name,
            "true": true_class,
            "pred": pred_class,
            "conf": confidence,
            "correct": is_correct,
            "time_ms": inference_time,
            "top3": top3
        })
    
    # Print results table
    print(f"\n{'Image':<30} {'True':>6} {'Pred':>6} {'Conf':>8} {'Time':>8} {'Status':>8}")
    print("-" * 70)
    
    for r in results:
        status = "✓ CORRECT" if r["correct"] else "✗ WRONG"
        print(f"{r['file']:<30} {r['true']:>6} {r['pred']:>6} {r['conf']*100:>7.1f}% {r['time_ms']:>6.1f}ms {status:>8}")
    
    print("-" * 70)
    
    # Print detailed predictions for a few samples
    print("\n[DETAILED PREDICTIONS - Top 3 classes for each image]")
    print("-" * 70)
    for r in results[:5]:  # Show first 5
        print(f"\n  {r['file']}:")
        print(f"    True class: {r['true']} µm")
        for i, (cls, prob) in enumerate(r['top3'], 1):
            marker = " ← predicted" if i == 1 else ""
            print(f"    {i}. Class {cls}: {prob*100:5.1f}%{marker}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("[DEMO SUMMARY]")
    print("=" * 70)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = total_time / len(results) if results else 0
    avg_conf = np.mean([r["conf"] for r in results]) * 100 if results else 0
    
    print(f"\n  Total images processed:  {len(results)}")
    print(f"  Correct predictions:     {correct}/{total}")
    print(f"  Demo Accuracy:           {accuracy:.1f}%")
    print(f"  Average confidence:      {avg_conf:.1f}%")
    print(f"  Average inference time:  {avg_time:.2f} ms/image")
    print(f"  Total inference time:    {total_time:.2f} ms")
    
    # Performance indicator
    print("\n[PERFORMANCE INDICATOR]")
    if accuracy >= 60:
        print("  ████████████████████ EXCELLENT - Model performing well!")
    elif accuracy >= 40:
        print("  ██████████████░░░░░░ GOOD - Reasonable performance")
    else:
        print("  ██████░░░░░░░░░░░░░░ MODERATE - Room for improvement")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
