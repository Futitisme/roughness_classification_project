# Surface Roughness Classification - Demo

This folder contains a demonstration script for the trained CNN model.

## Quick Start

```bash
python3 run.py
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pillow

Install with:
```bash
pip install torch numpy pillow
```

## What it does

The `run.py` script:
1. Loads the pre-trained CNN model from `cnn_final.pt`
2. Runs inference on 16 sample images (one from each class)
3. Displays predictions with confidence scores
4. Shows accuracy metrics and inference timing

## Sample Output

```
======================================================================
   SURFACE ROUGHNESS CLASSIFICATION - CNN MODEL DEMO
======================================================================

Task: Classify surface roughness from microscope images
Classes: 16 roughness levels (00, 03, 06, ... 45 micrometers)
Model: CNN with 3 convolutional layers
----------------------------------------------------------------------

[MODEL INFORMATION]
  Architecture: (32, 64, 128)
  Parameters: 2,194,192
  Model size: 8.37 MB

[DEMO] Running inference on 16 sample images...
----------------------------------------------------------------------

Image                          True   Pred     Conf     Time   Status
----------------------------------------------------------------------
class_00_sample1.jpg             00     00    85.3%   12.5ms ✓ CORRECT
class_09_sample1.jpg             09     09    72.1%   11.2ms ✓ CORRECT
...

======================================================================
[DEMO SUMMARY]
======================================================================
  Demo Accuracy:           62.5%
  Average inference time:  11.8 ms/image
======================================================================
```

## Files

- `run.py` - Main demo script
- `cnn_final.pt` - Trained CNN model (8.37 MB)
- `sample_images/` - 16 test images (one per class)
- `README.md` - This file
