# What problem I'm solving?

The objective of this project is to solve a **multi-class image classification problem**, where microscopic grayscale images of material surfaces must be assigned to one of several discrete surface roughness categories. The task is approached by training and comparing neural network models with progressively increasing representational capacity, starting from a **linear baseline**, moving to fully connected **feed-forward network**, and finally employing **convolutional neural network** designed to capture spatial and textural patterns. Each model is trained and evaluated under a unified experimental protocol, with systematic hyperparameter tuning based on validation performance, in order to analyze the trade-off between model complexity, classification accuracy, and computational cost.


# Data Preparation

This section describes the input data and the preprocessing pipeline applied before training the models. The objective is to obtain a clean, reproducible, and architecture-agnostic data representation suitable for all model types.

---

## 1. Raw Data

The dataset used in this project is the **Surface Roughness Classification** dataset from Kaggle. It contains microscopy images of machined surfaces organized into folders by roughness class.

**Dataset characteristics:**

| Property | Value |
|----------|-------|
| Total images | 3840 |
| Number of classes | 16 |
| Images per class | 240 (perfectly balanced) |
| Image format | JPEG |
| Class labels | `00`, `03`, `06`, ..., `45` |

The folder-based structure allows class labels to be inferred directly from file paths. Each folder name represents a discrete roughness level, making this a **multi-class classification problem** where each image belongs to exactly one class.

---

## 2. Dataset Indexing and Label Encoding

To ensure reproducibility and simplify downstream processing, all images were indexed into a single Pandas DataFrame containing:

- `path` — absolute path to the image file
- `class_name` — string class identifier (folder name)
- `label` — integer-encoded class index $\in [0, 15]$

### Label Mapping

Class names are sorted alphabetically and mapped to consecutive integers:

```python
class_names = sorted(df["class_name"].unique())
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
df["label"] = df["class_name"].map(class_to_idx)
```

This deterministic mapping ensures that all models operate on identically labeled data across experiments.

---

## 3. Train / Validation / Test Split

The dataset was divided into three disjoint subsets with a **stratified split** to preserve the class distribution in each subset:

| Subset | Proportion | Images | Per class |
|--------|------------|--------|-----------|
| Train | 70% | 2688 | 168 |
| Validation | 15% | 576 | 36 |
| Test | 15% | 576 | 36 |

### Split Procedure

The split is performed in two stages:

1. **Extract test set** (15% of total):

$$
\mathcal{D}_{\text{test}} \sim 0.15 \cdot \mathcal{D}
$$

2. **Split remaining data** into training and validation sets. Since the validation set should be 15% of the original dataset and 85% remains after extracting the test set:

$$
\frac{|\mathcal{D}_{\text{val}}|}{|\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}}|} = \frac{0.15}{0.85} \approx 0.176
$$

```python
df_trainval, df_test = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=SEED
)

val_ratio = 0.15 / 0.85
df_train, df_val = train_test_split(
    df_trainval, test_size=val_ratio, stratify=df_trainval["label"], random_state=SEED
)
```

**Stratification** ensures that each subset contains the same proportion of samples from every class, preventing biased evaluation and enabling fair model comparison.

---

## 4. Image Preprocessing

All images undergo a uniform preprocessing pipeline to produce tensors compatible with any model architecture.

### Preprocessing Steps

For each image $I$:

**Step 1. Grayscale conversion**

$$
I \leftarrow \texttt{grayscale}(I)
$$

Color information is discarded since surface texture is the primary discriminative feature.

**Step 2. Resize to fixed resolution**

$$
I \leftarrow \texttt{resize}(I, (H, W)), \quad H = W = 64
$$

A fixed spatial resolution of $64 \times 64$ pixels is required for batching and ensures consistent input dimensions across all models.

**Step 3. Intensity normalization**

$$
X = \frac{I}{255}, \quad X \in [0, 1]^{H \times W}
$$

Pixel values are scaled from integer range $[0, 255]$ to floating-point range $[0, 1]$. This normalization stabilizes gradient flow during training.

**Step 4. Channel dimension**

$$
X \in \mathbb{R}^{1 \times H \times W}
$$

A channel dimension is added to conform to the PyTorch tensor format (C, H, W).

### Implementation

```python
img = Image.open(path).convert("L")  # grayscale
img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
arr = np.asarray(img, dtype=np.float32) / 255.0
x = torch.from_numpy(arr).unsqueeze(0)  # shape: (1, H, W)
y = torch.tensor(label, dtype=torch.long)
```

The final tensor `x` has shape $(1, 64, 64)$ with values in $[0, 1]$, and `y` is a scalar class index.

---

## 5. Reproducibility

To guarantee reproducible experiments across runs, random seeds are fixed for all relevant libraries:

$$
\texttt{seed} = 42
$$

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This ensures:
- Identical data splits across experiments
- Reproducible weight initialization
- Deterministic training dynamics (within hardware limitations)

---

## 6. Evaluation Metrics

All evaluation metrics are implemented from scratch without external libraries and shared across all models.

### Confusion Matrix

For $C$ classes, the confusion matrix $M \in \mathbb{N}^{C \times C}$ counts predictions:

$$
M_{i,j} = \bigl|\{n : y_n = i \land \hat{y}_n = j\}\bigr|
$$

where rows represent true labels and columns represent predictions.

```python
idx = y_true * num_classes + y_pred
bc = torch.bincount(idx, minlength=num_classes * num_classes)
cm = bc.reshape(num_classes, num_classes)
```

### Precision, Recall, and F1-Score

For each class $c$, let $TP_c$, $FP_c$, $FN_c$ denote true positives, false positives, and false negatives respectively:

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c + \varepsilon}
$$

$$
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c + \varepsilon}
$$

$$
F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c + \varepsilon}
$$

where $\varepsilon$ is a small constant for numerical stability.

**Macro-averaged F1** treats all classes equally:

$$
\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c
$$

**Balanced Accuracy** is the mean per-class recall:

$$
\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=1}^{C} \text{Recall}_c
$$

### Top-$k$ Accuracy

Top-$k$ accuracy measures whether the true label appears among the $k$ highest-scoring predictions:

$$
\text{Top-}k = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\bigl[y_n \in \text{TopK}(\mathbf{z}_n, k)\bigr]
$$

where $\mathbf{z}_n$ is the logit vector for sample $n$. In this project, $k=2$ is used to capture near-miss predictions.

---

## 7. Loss Function

The cross-entropy loss is computed directly from logits using a numerically stable formulation based on the log-sum-exp trick.

For a batch of $B$ samples with logits $\mathbf{z}_n \in \mathbb{R}^C$ and true labels $y_n$:

$$
\log p_{n,c} = z_{n,c} - \log \sum_{j=1}^{C} e^{z_{n,j}}
$$

$$
\mathcal{L} = -\frac{1}{B} \sum_{n=1}^{B} \log p_{n, y_n}
$$

```python
def logsumexp(x, dim=-1, keepdim=False):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))

lse = logsumexp(logits, dim=1, keepdim=True)
log_probs = logits - lse
nll = -log_probs[torch.arange(B), y]
loss = nll.mean()
```

The `logsumexp` function subtracts the maximum value before exponentiation to prevent numerical overflow, ensuring stable computation even with large logit magnitudes.

---

## Summary

The data preparation pipeline produces:

- **Indexed dataset** with deterministic label encoding
- **Stratified splits** preserving class balance (70/15/15)
- **Normalized tensors** of shape $(1, 64, 64)$ with values in $[0, 1]$
- **Fixed random seeds** for full reproducibility
- **Unified evaluation interface** with metrics computed from scratch

This architecture-agnostic representation serves as the foundation for training and comparing all models in subsequent sections.

## Softmax Regression (Linear Baseline)

### Motivation and role in the project

As a first step, we implemented **Softmax Regression** as a **linear baseline model**.  
The purpose of this model is not to achieve high accuracy, but to establish a **reference point** for the task difficulty and to quantify how much performance gain is obtained when increasing model capacity (MLP and CNN).

Softmax Regression is the **simplest discriminative model** suitable for multi-class classification. It assumes that the classes are **linearly separable** in the input space and does not exploit any spatial structure of the images. For this reason, it provides a meaningful lower bound on performance for image-based tasks.

---

### Model definition

Each input image is first flattened into a vector:

$$
\mathbf{x} \in \mathbb{R}^{D}, \quad D = H \cdot W
$$

The model computes class scores (logits) using a single linear transformation:

$$
\mathbf{z} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{D \times C}, \ \mathbf{b} \in \mathbb{R}^{C}
$$

where $C = 16$ is the number of surface roughness classes.

The logits are converted into class probabilities using the **softmax function**:

$$
p(y=c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}
$$

This formulation ensures that the output is a valid categorical distribution over the classes.

---

### Loss function

We trained the model using the **categorical cross-entropy loss**, implemented from scratch in a numerically stable way:

$$
\mathcal{L} = -\frac{1}{B} \sum_{n=1}^{B} \log p(y_n \mid \mathbf{x}_n)
$$

To avoid numerical instability, the loss is computed directly from logits using the log-sum-exp trick:

$$
\log p_{n,c} = z_{n,c} - \log \sum_{j=1}^{C} e^{z_{n,j}}
$$

---

### Optimization and regularization

All components of the training procedure were implemented **from scratch using PyTorch tensors**, without relying on high-level modules such as `nn.Linear` or `nn.CrossEntropyLoss`.

Training was performed using **mini-batch gradient descent**, with the following gradient expressions:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top (\mathbf{P} - \mathbf{Y}), \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{n} (\mathbf{P}_n - \mathbf{Y}_n)
$$

where:
- $\mathbf{P}$ is the matrix of predicted probabilities,
- $\mathbf{Y}$ is the one-hot encoding of the ground-truth labels.

We optionally applied **L2 weight decay**:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta (\nabla \mathbf{W} + \lambda \mathbf{W})
$$

---

### Hyperparameter search (cross-validation)

To comply with the course requirements, we performed a **systematic grid search** over the main hyperparameters:

- Number of epochs: $\{5, 10, 20, 35\}$
- Learning rate: $\{10^{-2}, 3\cdot10^{-3}, 10^{-3}\}$
- Weight decay: $\{0, 10^{-4}, 10^{-3}\}$

For each configuration, the model was trained from scratch and evaluated on the **validation set**.  
The selection criterion was the **best validation Macro-F1 score across epochs**, which is more informative than accuracy for multi-class problems.

**Figure:** *Grid search results — Best validation Macro-F1 across all configurations*  
**Figure:** *Heatmaps of validation Macro-F1 for different learning rates and weight decay values*

---

### Learning dynamics

For the best configuration, we monitored training and validation metrics across epochs:

- Training loss
- Validation loss
- Validation accuracy
- Validation Macro-F1
- Validation Top-2 accuracy

**Figure:** *Learning curves for Softmax Regression (loss, accuracy, Macro-F1)*

These curves show fast convergence and limited capacity to further improve performance, as expected from a linear model.

---

### Final performance

The best hyperparameter configuration was:
- Epochs: 35  
- Learning rate: 0.01  
- Weight decay: $10^{-4}$

Final metrics:

- **Validation set**  
  - Accuracy: ≈ 0.11  
  - Macro-F1: ≈ 0.08  

- **Test set**  
  - Accuracy: ≈ 0.13  
  - Macro-F1: ≈ 0.10  

**Figure:** *Confusion matrix on the test set*  
**Figure:** *Per-class F1 scores on the test set*

The confusion matrix shows that the model struggles to separate most classes, confirming that linear decision boundaries are insufficient for this texture-based classification task.

---

### Computational efficiency

Despite low predictive performance, Softmax Regression is extremely efficient:

- Number of parameters: ~65k  
- Model size: ~0.25 MB  
- Inference latency: < 1 ms per image  

**Figure:** *Quality vs performance (Macro-F1 vs inference latency)*

This highlights the trade-off between model simplicity and representational power.

---

### Summary

Softmax Regression serves as a **transparent, fully controlled baseline**, implemented entirely from scratch.  
Its poor performance relative to deeper models demonstrates that surface roughness classification from microscopy images requires **non-linear feature extraction** and **spatial inductive biases**, motivating the use of MLPs and convolutional neural networks in the subsequent sections.
