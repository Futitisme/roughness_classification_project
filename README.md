# What problem I'm solving?

The objective of this project is to solve a **multi-class image classification problem**, where microscopic grayscale images of material surfaces must be assigned to one of several discrete surface roughness categories. The task is approached by training and comparing neural network models with progressively increasing representational capacity, starting from a **linear baseline**, moving to fully connected **feed-forward network**, and finally employing **convolutional neural network** designed to capture spatial and textural patterns. Each model is trained and evaluated under a unified experimental protocol, with systematic hyperparameter tuning based on validation performance, in order to analyze the trade-off between model complexity, classification accuracy, and computational cost.


# Input Data and Preprocessing

This section describes the input data used in the project and the full preprocessing pipeline applied before training the models. The goal of the pipeline is to obtain a **clean, reproducible, and architecture-agnostic representation** of the data, suitable for linear models, fully connected networks, and convolutional neural networks alike.

---

## 1. Raw Input Data

The dataset is provided as an **image classification dataset organized by folders**, where:

- each folder corresponds to **one surface roughness class**,
- the folder name (e.g., `"00"`, `"03"`, …, `"45"`) represents the class label,
- each folder contains multiple **JPEG microscopy images** acquired at fixed magnification.

The problem is a **multi-class classification task** with:
- **16 mutually exclusive classes**,
- each image belonging to **exactly one class**.

This directory-based structure allows labels to be inferred directly from file paths.

---

## 2. Dataset Indexing and Label Encoding

To ensure reproducibility and simplify downstream processing, the entire dataset is indexed into a single Pandas DataFrame. Each row corresponds to one image and contains:

- `path`: absolute path to the image file,
- `class_name`: class identifier as a string (folder name),
- `label`: integer-encoded class index in the range $[0, 15]$.

### Label mapping

Class names are sorted alphabetically and mapped to integer indices:

$$
\texttt{class\_to\_idx} : \texttt{class\_name} \rightarrow \{0, \dots, 15\}
$$

$$
\texttt{label}_i = \texttt{class\_to\_idx}(\texttt{class\_name}_i)
$$

Core implementation:

```python
class_names = sorted(df["class_name"].unique())
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
df["label"] = df["class_name"].map(class_to_idx)
```

**Rationale.**
Using a centralized DataFrame ensures that:

* data splits are reproducible,
* statistics are easy to compute,
* all models operate on the same indexed dataset.

---

## 3. Stratified Train / Validation / Test Split

The dataset is split into three disjoint subsets:

* **Training**: 70%
* **Validation**: 15%
* **Test**: 15%

The split is **stratified by class label**, meaning that each subset preserves the original per-class distribution.

### Split procedure

1. Extract the test set:

$$
\mathcal{D}_{\text{test}} \sim 0.15 \cdot \mathcal{D}
$$

2. Split the remaining data into training and validation:

$$
\frac{|\mathcal{D}_{\text{val}}|}{|\mathcal{D}_{\text{trainval}}|} = \frac{0.15}{0.85}
$$

Core code:

```python
df_trainval, df_test = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=SEED
)
val_ratio = 0.15 / 0.85
df_train, df_val = train_test_split(
    df_trainval, test_size=val_ratio, stratify=df_trainval["label"], random_state=SEED
)
```

**Rationale.**
Stratification prevents biased splits and allows fair comparison between models by ensuring that validation and test sets reflect the same class distribution as the training data.

---

## 4. Image Preprocessing and Torch Dataset

All images are transformed into a **uniform tensor representation**, independent of the model architecture.

### Preprocessing steps

For each image $I$:

1. **Grayscale conversion**

$$
I \leftarrow \texttt{to\_grayscale}(I)
$$

2. **Resize to fixed resolution**

$$
I \leftarrow \text{resize}(I,\ (H, W)), \quad H = W = \texttt{IMAGE\_SIZE}
$$

3. **Intensity normalization**

$$
X = \frac{I}{255}, \quad X \in [0,1]^{H \times W}
$$

4. **Channel dimension addition**

$$
X \in \mathbb{R}^{1 \times H \times W}
$$

### Core implementation

```python
img = Image.open(path).convert("L")
img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
arr = np.asarray(img, dtype=np.float32) / 255.0
x = torch.from_numpy(arr).unsqueeze(0)
y = torch.tensor(label, dtype=torch.long)
```

**Rationale.**

* A fixed spatial resolution is required for batching and model comparison.
* Grayscale images reduce input dimensionality while preserving texture information.
* Normalization stabilizes optimization and gradient flow.

---

## 5. Reproducibility

To guarantee reproducible experiments, random seeds are fixed for all relevant libraries:

$$
\texttt{seed} = 42
$$

Applied to:

* Python random module,
* NumPy,
* PyTorch (CPU and CUDA),
* cuDNN deterministic mode.

Core code:

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Rationale.**
This ensures identical data splits and comparable training dynamics across runs.

---

## 6. Evaluation Metrics (Implemented from Scratch)

All evaluation metrics are implemented manually, without relying on external libraries, and are shared across all models.

### Confusion matrix

For $C$ classes, the confusion matrix $M \in \mathbb{N}^{C \times C}$ is defined as:

$$
M_{i,j} = \#\{n \mid y_n = i,\ \hat{y}_n = j\}
$$

Efficient computation using a bincount-based approach:

```python
idx = y_true * num_classes + y_pred
bc = torch.bincount(idx, minlength=num_classes*num_classes)
cm = bc.reshape(num_classes, num_classes)
```

---

### Precision, Recall, and F1-score

For each class $c$:

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c + \varepsilon}
$$

$$
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c + \varepsilon}
$$

$$
F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c + \varepsilon}
$$

Aggregate metrics:

$$
\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c
$$

$$
\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=1}^{C} \text{Recall}_c
$$

---

### Top-$k$ Accuracy ($k = 2$)

$$
\text{Top-}k = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\{y_n \in \text{TopK}(\mathbf{z}_n)\}
$$

Core code:

```python
topk = torch.topk(logits, k=k, dim=1).indices
correct = (topk == y_true.view(-1,1)).any(dim=1).float().mean().item()
```

**Rationale.**
Macro-F1 and balanced accuracy provide class-balanced performance estimates, while Top-2 accuracy reflects near-miss predictions in ambiguous cases.

---

## 7. Stable Cross-Entropy Loss from Logits

The cross-entropy loss is computed directly from logits using a numerically stable formulation:

$$
\log p_{n,c} = z_{n,c} - \log \sum_{j=1}^{C} e^{z_{n,j}}
$$

$$
\mathcal{L} = -\frac{1}{B} \sum_{n=1}^{B} \log p_{n,y_n}
$$

Core implementation:

```python
lse = logsumexp(logits, dim=1, keepdim=True)
log_probs = logits - lse
nll = -log_probs[torch.arange(logits.size(0)), y]
loss = nll.mean()
```

**Rationale.**
Using `logsumexp` avoids numerical instability that can arise from explicitly computing softmax probabilities.

---

## 8. Unified Evaluation Interface

All models—both scratch implementations and PyTorch-based CNNs—are evaluated using a single unified function:

```python
evaluate_model(loader, num_classes, device, model=cnn_model)
evaluate_model(loader, num_classes, device, forward_fn=scratch_forward)
```

This function:

* computes loss and predictions,
* accumulates confusion matrices,
* returns accuracy, top-2 accuracy, Macro-F1, Balanced Accuracy, and per-class metrics.

**Rationale.**
A unified evaluation protocol guarantees a fair and consistent comparison across all architectures.
