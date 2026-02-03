#!/usr/bin/env python3
"""
01_data_preparation.py

Downloads the Surface Roughness Classification dataset via kagglehub,
builds DataFrame with image paths and labels, performs stratified train/val/test split,
creates PyTorch Datasets and DataLoaders, and saves processed data for other scripts.
"""

import os
import sys
import random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 256
NUM_WORKERS = 2

DATASET_ID = "sahini/surface-roughness-classification"
TARGET_DIRNAME = "Cropped100x"

# Output directory for processed data
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_dataset():
    """Download dataset via kagglehub and locate Cropped100x folder."""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system(f"{sys.executable} -m pip install -q kagglehub")
        import kagglehub

    print(f"Downloading dataset: {DATASET_ID}")
    root = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"Downloaded dataset root: {root}")

    # Find Cropped100x anywhere inside the downloaded directory
    candidates = [p for p in root.rglob(TARGET_DIRNAME) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find folder '{TARGET_DIRNAME}' under: {root}")

    cropped_dir = candidates[0]
    print(f"Cropped100x path: {cropped_dir}")

    return cropped_dir


def verify_dataset(cropped_dir):
    """Verify dataset structure and show basic statistics."""
    image_exts = {".jpg", ".jpeg", ".png"}
    class_stats = OrderedDict()

    for class_dir in sorted(cropped_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in image_exts]
        class_stats[class_dir.name] = len(images)

    total_images = sum(class_stats.values())

    print(f"\nTotal images: {total_images}\n")
    print("Images per class:")
    for cls, cnt in class_stats.items():
        print(f"  class {cls}: {cnt}")

    # Sanity checks
    assert len(class_stats) == 16, "Expected exactly 16 classes"
    assert all(v > 0 for v in class_stats.values()), "Some class has zero images"

    return class_stats


def build_dataframe(cropped_dir):
    """Build DataFrame with image paths and labels."""
    image_exts = {".jpg", ".jpeg", ".png"}
    rows = []

    for class_dir in sorted(cropped_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in image_exts:
                rows.append({
                    "path": str(img_path),
                    "class_name": class_name
                })

    df = pd.DataFrame(rows)

    # Create class mappings (sorted for reproducibility)
    class_names = sorted(df["class_name"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    df["label"] = df["class_name"].map(class_to_idx)

    print(f"\nDataFrame shape: {df.shape}")
    print("\nClass mapping:")
    for k, v in class_to_idx.items():
        print(f"  {k} -> {v}")

    return df, class_to_idx, idx_to_class


def stratified_split(df, seed=SEED):
    """
    Stratified train/val/test split (70/15/15).
    """
    np.random.seed(seed)

    # 1) test split: 15% of total
    df_trainval, df_test = train_test_split(
        df,
        test_size=0.15,
        random_state=seed,
        stratify=df["label"]
    )

    # 2) val split: 15% of total => 0.15 / 0.85 of trainval
    val_ratio_within_trainval = 0.15 / 0.85

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_within_trainval,
        random_state=seed,
        stratify=df_trainval["label"]
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("\nSplit sizes:")
    print(f"  train: {len(df_train)}")
    print(f"  val:   {len(df_val)}")
    print(f"  test:  {len(df_test)}")
    print(f"  total: {len(df_train) + len(df_val) + len(df_test)}")

    return df_train, df_val, df_test


class SurfaceRoughnessDataset(Dataset):
    """
    PyTorch Dataset for Surface Roughness images.
    Converts to grayscale, resizes, and normalizes to [0,1].
    Output: (1, H, W) float32 tensor.
    """
    def __init__(self, df, image_size=64):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        y = int(row["label"])

        # Read image
        img = Image.open(path).convert("L")  # grayscale

        # Resize
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        # To numpy [0,1]
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W)

        # To torch tensor (1,H,W)
        x = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W), float32
        y = torch.tensor(y, dtype=torch.long)   # scalar long

        return x, y


def create_dataloaders(df_train, df_val, df_test, image_size=IMAGE_SIZE):
    """Create PyTorch DataLoaders."""
    train_ds = SurfaceRoughnessDataset(df_train, image_size=image_size)
    val_ds = SurfaceRoughnessDataset(df_val, image_size=image_size)
    test_ds = SurfaceRoughnessDataset(df_test, image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Quick sanity check
    xb, yb = next(iter(train_loader))
    print(f"\nDataLoader sanity check:")
    print(f"  x batch: {xb.shape}, {xb.dtype}, min/max: {float(xb.min()):.4f}/{float(xb.max()):.4f}")
    print(f"  y batch: {yb.shape}, {yb.dtype}, min/max: {int(yb.min())}/{int(yb.max())}")

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def save_processed_data(df_train, df_val, df_test, class_to_idx, idx_to_class, output_dir):
    """Save processed DataFrames and mappings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)

    # Save class mappings
    mappings = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }
    torch.save(mappings, output_dir / "class_mappings.pt")

    print(f"\nSaved processed data to: {output_dir}")
    print(f"  - train.csv ({len(df_train)} samples)")
    print(f"  - val.csv ({len(df_val)} samples)")
    print(f"  - test.csv ({len(df_test)} samples)")
    print(f"  - class_mappings.pt")


def main():
    print("=" * 60)
    print("Surface Roughness Classification - Data Preparation")
    print("=" * 60)

    # Set seeds
    set_seed(SEED)
    print(f"\nSeed set to: {SEED}")

    # Download dataset
    cropped_dir = download_dataset()

    # Verify dataset
    class_stats = verify_dataset(cropped_dir)

    # Build DataFrame
    df, class_to_idx, idx_to_class = build_dataframe(cropped_dir)

    # Stratified split
    df_train, df_val, df_test = stratified_split(df)

    # Create DataLoaders
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_dataloaders(
        df_train, df_val, df_test
    )

    # Save processed data
    save_processed_data(df_train, df_val, df_test, class_to_idx, idx_to_class, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)

    return {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


if __name__ == "__main__":
    main()
