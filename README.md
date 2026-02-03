# Surface Roughness Classification - Python Scripts

This folder contains standalone Python scripts that replicate the full pipeline from the Jupyter notebook `neural_networks_project-5.ipynb`.

## Project Structure

```
scripts/
├── 01_data_preparation.py    # Download and prepare dataset
├── 02_train_softmax.py       # Train Softmax Regression (baseline)
├── 03_train_mlp.py           # Train Deep MLP with grid search
├── 04_train_cnn.py           # Train CNN with grid search
├── 05_train_cnn_best.py      # Train CNN with best hyperparameters
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Generated after running 01_data_preparation.py
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── class_mappings.pt
└── checkpoints/              # Generated after training
    ├── softmax_best.pt
    ├── mlp_deep_best.pt
    ├── cnn_best.pt
    ├── cnn_final.pt
    └── *.png (plots)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have Kaggle credentials set up for `kagglehub`:
   - Either set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`
   - Or have `~/.kaggle/kaggle.json` configured

## Running the Scripts

**Important:** Scripts must be run in order since they depend on outputs from previous scripts.

### Step 1: Data Preparation
```bash
python 01_data_preparation.py
```
This script:
- Downloads the Surface Roughness Classification dataset from Kaggle
- Creates train/val/test splits (70/15/15)
- Saves processed CSVs and class mappings to `data/` folder

### Step 2: Train Softmax Regression (Baseline)
```bash
python 02_train_softmax.py
```
This script:
- Implements Softmax Regression from scratch (no PyTorch autograd)
- Runs grid search over epochs, learning rate, and weight decay
- Best config found: epochs=35, lr=0.01, weight_decay=0.0
- Saves model to `checkpoints/softmax_best.pt`

### Step 3: Train Deep MLP
```bash
python 03_train_mlp.py
```
This script:
- Implements MLP from scratch with manual backpropagation
- Architecture: Linear -> Sigmoid -> Dropout (repeated) -> Linear
- Runs grid search over hidden sizes, dropout, epochs, lr, weight decay
- Best config found: hidden_sizes=(1024, 512), dropout=0.0, epochs=60, lr=0.001
- Saves model to `checkpoints/mlp_deep_best.pt`

### Step 4: Train CNN with Grid Search
```bash
python 04_train_cnn.py
```
This script:
- Uses PyTorch nn.Module for CNN implementation
- Architecture: 3x(Conv2d -> ReLU -> MaxPool2d) -> Linear -> ReLU -> Dropout -> Linear
- Runs grid search over channel sizes, dropout, lr, weight decay
- Uses early stopping with patience=8
- Saves model to `checkpoints/cnn_best.pt`

## Model Performance Summary

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|-------|--------------|--------------|---------------|---------------|
| Softmax Regression | ~10% | ~7% | ~10% | ~6% |
| Deep MLP | ~16-18% | ~13-15% | ~13-16% | ~10-12% |
| CNN | ~61% | ~61% | ~63% | ~63% |

## Key Features

- **Reproducibility**: All scripts use SEED=42 for reproducible results
- **From Scratch Implementation**: Softmax and MLP use manual gradient computation
- **Metrics**: Accuracy, Macro-F1, Balanced Accuracy, Top-2 Accuracy, Confusion Matrix
- **Visualization**: Learning curves and confusion matrices saved as PNG files
- **Early Stopping**: CNN uses early stopping to prevent overfitting

## Notes

- Dataset: Surface Roughness Classification (16 classes, 3840 images total)
- Image size: 64x64 grayscale
- Split: 70% train, 15% val, 15% test (stratified)
- All training uses SGD (Softmax/MLP) or Adam (CNN) optimizer
