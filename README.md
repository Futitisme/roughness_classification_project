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
