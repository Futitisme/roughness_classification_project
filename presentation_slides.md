# Presentation Slides: Surface Roughness Classification

---

## Slide 1: Problem & Dataset

### Problem Statement
- **Task**: Multi-class image classification
- **Goal**: Classify microscopic grayscale images of material surfaces into **16 roughness categories**
- **Challenge**: Distinguish subtle texture differences between surface roughness levels

### Dataset: Surface Roughness Classification (Kaggle)

| Property | Value |
|----------|-------|
| Total images | 3840 |
| Number of classes | 16 |
| Images per class | 240 (balanced) |
| Image format | JPEG (grayscale) |
| Class labels | `00`, `03`, `06`, ..., `45` |

### Three Models Compared
1. **Softmax Regression** — linear baseline (from scratch)
2. **MLP with Sigmoid** — non-linear capacity (from scratch)
3. **CNN** — spatial feature extraction (PyTorch)

---

## Slide 2: Data Preparation Pipeline

### 1. Label Encoding
- Class names sorted and mapped to integers 0-15

### 2. Train / Validation / Test Split
- **Stratified sampling** to preserve class distribution

| Subset | Proportion | Images | Per class |
|--------|------------|--------|-----------|
| Train | 70% | 2688 | 168 |
| Validation | 15% | 576 | 36 |
| Test | 15% | 576 | 36 |

### 3. Image Preprocessing Pipeline
1. **Grayscale conversion** — color discarded (texture is primary feature)
2. **Resize to 64×64** — fixed resolution for batching
3. **Normalize to [0, 1]** — pixel values from [0, 255] to [0, 1]
4. **Add channel dimension** — shape: (1, 64, 64) for PyTorch

---

## Slide 3: Model 1 — Softmax Regression (Math)

### Motivation
- Simplest discriminative model for multi-class classification
- Assumes classes are **linearly separable** in pixel space
- Establishes a **lower bound** on performance

### Forward Pass
- Input: flattened image $\mathbf{x} \in \mathbb{R}^{4096}$ (64×64 pixels)
- Logits: $\mathbf{z} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}$
- Parameters: $\mathbf{W} \in \mathbb{R}^{4096 \times 16}$, $\mathbf{b} \in \mathbb{R}^{16}$

### Softmax Function
$$p(y=c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{16} e^{z_j}}$$

### Cross-Entropy Loss
$$\mathcal{L} = -\frac{1}{B} \sum_{n=1}^{B} \log p(y_n \mid \mathbf{x}_n)$$

### Gradient (Manual Backpropagation)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top (\mathbf{P} - \mathbf{Y})$$

**Implemented fully from scratch** (no nn.Linear, no autograd)

---

## Slide 4: Model 1 — Softmax Regression (Results)

### Hyperparameter Search
- **36 configurations** tested
- Grid: epochs {5, 10, 20, 35}, lr {0.01, 0.003, 0.001}, weight decay {0, 1e-4, 1e-3}

**Best config**: epochs=35, lr=0.01, weight_decay=1e-4

### Learning Curves
<img width="1131" alt="Softmax learning curves" src="https://github.com/user-attachments/assets/7883a14b-91e5-4c5c-8571-9bdff6d8bbc3" />

### Final Performance

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | ~0.11 | **0.13** |
| Macro-F1 | ~0.08 | **0.10** |

### Confusion Matrix
<img width="455" alt="Softmax confusion matrix" src="https://github.com/user-attachments/assets/2c985f07-3f8a-4ec9-ada5-6a9eca631dd8" />

### Conclusion
- Fast convergence but **limited capacity**
- **Linear decision boundaries are insufficient** for texture classification

---

## Slide 5: Model 2 — MLP with Sigmoid (Math)

### Motivation
- Adds **non-linear capacity** through hidden layers
- Can learn **non-linear decision boundaries**
- Still treats image as **flat vector** (no spatial structure)

### Architecture
```
Input (4096) → Linear(1024) → Sigmoid → Linear(512) → Sigmoid → Linear(16) → Softmax
```

### Sigmoid Activation
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Backpropagation (Manual)
- **Linear layer gradient**: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}$
- **Sigmoid gradient**: $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \odot \mathbf{S}(1 - \mathbf{S})$

### Input Preprocessing
$$x' = 2x - 1 \quad \text{(maps [0,1] to [-1,1] for better gradient flow)}$$

**Implemented fully from scratch** (manual forward/backward for all layers)

---

## Slide 6: Model 2 — MLP with Sigmoid (Results)

### Hyperparameter Search
- **96 configurations** tested
- Grid: hidden layouts, dropout {0, 0.3, 0.5}, lr {0.001, 0.0003}, epochs {40, 60}

**Best config**: hidden=(1024, 512), dropout=0.0, lr=0.001, epochs=60

### Learning Curves
<img width="1068" alt="MLP learning curves" src="https://github.com/user-attachments/assets/f1da18aa-282e-4ea3-84e6-73300642c96c" />

### Final Performance

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 0.160 | **0.130** |
| Macro-F1 | 0.127 | **0.104** |

### Confusion Matrix
<img width="517" alt="MLP test confusion matrix" src="https://github.com/user-attachments/assets/49c7ec88-32f7-4f57-917e-ae3e24b659bb" />

### Why MLP Struggles
1. **Flattening destroys spatial structure**
2. **No translation invariance** — patterns position-dependent
3. **Sigmoid saturation** — vanishing gradients

---

## Slide 7: Model 3 — CNN (Math)

### Motivation
CNNs exploit image-specific properties:
- **Spatial locality** — nearby pixels are related
- **Translation invariance** — detect patterns regardless of position
- **Hierarchical features** — edges → textures → patterns

### Architecture
```
Conv2d(1→32, 3×3) → ReLU → MaxPool(2)
Conv2d(32→64, 3×3) → ReLU → MaxPool(2)
Conv2d(64→128, 3×3) → ReLU → MaxPool(2)
Flatten → Linear(8192→256) → ReLU → Dropout(0.5) → Linear(256→16)
```

### Convolution Operation
$$z_k(i,j) = \sum_{c}\sum_{u,v} w_{k,c}(u,v) \cdot x_c(i+u, j+v) + b_k$$

### ReLU Activation (vs Sigmoid)
$$\text{ReLU}(z) = \max(0, z)$$
- Avoids vanishing gradient problem
- Allows training deeper networks

### Spatial Dimension Evolution
$$64 \times 64 \xrightarrow{\text{pool}} 32 \times 32 \xrightarrow{\text{pool}} 16 \times 16 \xrightarrow{\text{pool}} 8 \times 8$$

---

## Slide 8: Model 3 — CNN (Results)

### Training Setup
- **Optimizer**: Adam (adaptive learning rates)
- **Regularization**: Dropout (0.5), Weight decay (1e-4)
- **Early stopping**: patience=8 epochs
- Training stopped at epoch 39 (best: epoch 31)

### Learning Curves
<img width="1065" alt="CNN learning curves" src="https://github.com/user-attachments/assets/c79e2fef-5e06-4286-bd5b-d41cd791d275" />

### Final Performance

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 0.597 | **0.592** |
| Macro-F1 | 0.594 | **0.590** |
| Top-2 Accuracy | 0.837 | **0.825** |

### Confusion Matrix
<img width="479" alt="CNN test confusion matrix" src="https://github.com/user-attachments/assets/527aeada-bb08-4ac6-9112-9ee48f4f2e78" />

### Per-Class F1 Scores
<img width="1063" alt="CNN per-class F1" src="https://github.com/user-attachments/assets/603f0d10-7faa-482c-99f7-790a0bcbe049" />

---

## Slide 9: Results — Models Comparison

### Performance Comparison
<img width="1074" alt="All models comparison" src="https://github.com/user-attachments/assets/c63971c8-b1db-4bcf-8747-d0346ec5e9e0" />

### Summary Table

| Model | Test Accuracy | Test Macro-F1 | Parameters | Latency |
|-------|---------------|---------------|------------|---------|
| Softmax Regression | 0.13 | 0.10 | ~65K | <1 ms |
| MLP (1024, 512) | 0.16 | 0.12 | ~5M | 1.2 ms |
| **CNN (32-64-128)** | **0.59** | **0.59** | ~2.2M | 4 ms |

### Per-Class F1 Across All Models
<img width="1066" alt="Per-class F1 all models" src="https://github.com/user-attachments/assets/f7c126e7-40db-4c22-af5c-6c9b6146312b" />

### Quality vs Performance Trade-off
<img width="626" alt="Quality vs performance" src="https://github.com/user-attachments/assets/7b2da291-e7f1-4017-a291-be3b62e9913a" />

---

## Slide 10: Conclusion

### Key Findings

| Model | Why it performs this way |
|-------|--------------------------|
| **Softmax** | Linear boundaries insufficient for texture data |
| **MLP** | Flattening destroys spatial structure; no translation invariance |
| **CNN** | Local receptive fields + shared weights = best for images |

### Theoretical Takeaways

1. **Linear models are insufficient** for complex visual tasks requiring texture discrimination

2. **Fully connected networks** benefit from non-linearity but struggle without spatial inductive bias

3. **Convolutional architectures** provide the best balance between representation power and generalization for image data

### Final Conclusion

CNN is the **most appropriate model** for surface roughness classification:
- **~6× better accuracy** than linear/MLP baselines
- Exploits spatial structure through local filters
- Achieves ~60% accuracy on 16-class problem

Softmax & MLP serve as **valuable baselines** demonstrating the importance of **architectural inductive biases** in deep learning.
