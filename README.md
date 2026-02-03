# Surface Roughness Classification using Neural Networks

A deep learning project for classifying surface roughness from microscope images into 16 classes (0-45 µm).

## Project Structure

```
├── demo/                          # Ready-to-run demo
├── scripts_to_train_models/       # Training scripts
└── surface_roughness_presentation.pdf
```

## Quick Demo

```bash
cd demo

python run.py
```

## Results

| Model | Test Accuracy | Test Macro-F1 |
|-------|---------------|---------------|
| Softmax Regression | ~10% | ~6% |
| Deep MLP | ~16% | ~12% |
| **CNN** | **60%** | **60%** |

## Dataset

[Surface Roughness Classification](https://www.kaggle.com/datasets/sahini/surface-roughness-classification) - 3,840 images, 16 classes
