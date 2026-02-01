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

## MLP (From Scratch) — Nonlinear Baseline

### Motivation and role in the project

After the linear Softmax Regression baseline, we implemented a **Multi-Layer Perceptron (MLP)** to test whether adding **nonlinear capacity** improves performance on microscopy texture classification.  
Unlike the linear model, an MLP can learn **nonlinear decision boundaries** and compose multiple transformations of the input. However, it still treats the image as a **flat vector** and therefore does **not explicitly model spatial locality** (unlike a CNN). This makes the MLP a meaningful intermediate baseline between Softmax Regression and CNN.

---

### Architecture

We used a fully-connected feed-forward network operating on flattened images:

- Input: grayscale image $x \in \mathbb{R}^{1 \times H \times W}$, flattened to $\mathbf{x} \in \mathbb{R}^{D}$ where $D = H\cdot W$.
- Hidden stack: multiple **Linear → Sigmoid** blocks, optionally followed by dropout during training.
- Output: final Linear layer producing logits $\mathbf{z} \in \mathbb{R}^{C}$ for $C=16$ classes.
- Output activation: **Softmax** (multi-class, mutually exclusive labels).

Best-performing configuration in our grid search:
- Hidden sizes: **(1024, 512)**  (two hidden layers)
- Dropout: **0.0**
- Epochs: **60**
- Learning rate: **0.001**
- Weight decay: **0.0**
- Input preprocessing: **$[0,1] \to [-1,1]$** via $x' = 2x - 1$

**Figure to insert:** *Deep MLP — architecture diagram (Flatten → Linear(4096→1024) → Sigmoid → Linear(1024→512) → Sigmoid → Linear(512→16))*

> Note on naming: even with two hidden layers, it is a **Multi-Layer** Perceptron because it contains **multiple trainable linear transformations** (more than one linear layer).

---

### Forward pass mathematics

#### Linear layer
For an input minibatch $\mathbf{X} \in \mathbb{R}^{B \times d_{in}}$:

$$
\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

where $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$, $\mathbf{b} \in \mathbb{R}^{d_{out}}$.

#### Sigmoid activation

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### Dropout (training only, inverted dropout)
Given dropout probability $p$, keep probability $q=1-p$:

$$
\mathbf{M} \sim \text{Bernoulli}(q), \quad \tilde{\mathbf{H}} = \frac{\mathbf{H}\odot \mathbf{M}}{q}
$$

At inference, dropout is identity (no masking).

#### Softmax output
For logits $\mathbf{z} \in \mathbb{R}^{C}$:

$$
p_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}
$$

We implemented a numerically stable version:

$$
p_c = \frac{e^{z_c - m}}{\sum_{j=1}^{C} e^{z_j - m}}, \quad m=\max_j z_j
$$

---

### Loss function (multi-class cross-entropy)

Classes are **mutually exclusive**. Therefore we use softmax + cross-entropy:

$$
\mathcal{L} = -\frac{1}{B}\sum_{n=1}^{B}\log p_{n,y_n}
$$

---

### Backpropagation (implemented from scratch)

All gradients were computed manually, without `torch.nn` layers and without autograd.

#### Softmax + cross-entropy gradient (key simplification)
Let $\mathbf{P} \in \mathbb{R}^{B\times C}$ be predicted probabilities and $\mathbf{Y}$ the one-hot labels. Then:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} = \frac{\mathbf{P} - \mathbf{Y}}{B}
$$

This gives the gradient w.r.t. logits of the last layer.

#### Linear layer gradients
Given $\mathbf{Z} = \mathbf{XW} + \mathbf{b}$ and upstream gradient $\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{n=1}^{B} \mathbf{G}_n, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \mathbf{G}\mathbf{W}^\top
$$

#### Sigmoid backward
If $\mathbf{S}=\sigma(\mathbf{Z})$, then:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} =
\frac{\partial \mathcal{L}}{\partial \mathbf{S}} \odot \mathbf{S}(1-\mathbf{S})
$$

#### Weight decay (L2 regularization)
We applied L2 penalty via the gradient update:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}} + \lambda \mathbf{W}\right)
$$

#### Parameter update (SGD)
For each layer:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}}, \quad
\mathbf{b} \leftarrow \mathbf{b} - \eta \nabla_{\mathbf{b}}
$$

---

### Implementation details (what was done "from scratch")

The MLP training pipeline uses PyTorch only as:
- a tensor library (`matmul`, `exp`, etc.)
- mini-batching via `DataLoader`
- device acceleration (CPU/GPU)

All core neural network parts were implemented manually:
- Linear forward/backward
- Sigmoid forward/backward
- Dropout forward (train-time) and identity at inference
- Softmax + Cross-Entropy forward and backward
- SGD updates with optional L2 weight decay
- Metrics computed from scratch (confusion matrix, accuracy, macro-F1, balanced accuracy, top-2)

**Figure to insert:** *Deep MLP — learning curves (train loss, val loss, val acc, val macro-F1, val top-2)*

---

### Hyperparameter search (validation-based tuning)

To satisfy the course requirement ("evaluate different values for hyperparameters to find the optimal ones"), we ran a grid search over:

- Hidden layer layouts:
  - (512, 256)
  - (512, 512, 256)
  - (1024, 512)
  - (1024, 512, 256)

- Dropout:
  - 0.0, 0.3, 0.5

- Learning rate:
  - 0.001, 0.0003

- Weight decay:
  - 0.0, 1e-4

- Epochs:
  - 40, 60

Total runs: **96 configurations**.

Selection criterion: **best validation Macro-F1** during training (not only last epoch).

**Figure to insert:** *Deep MLP — grid search ranking (best validation Macro-F1 across all runs)*  
**Figure to insert:** *Deep MLP — hyperparameter heatmap (lr × weight decay) for fixed epochs/dropout and best architecture*

---

### Final results (best configuration)

Best configuration:
- Hidden sizes: **(1024, 512)**
- Dropout: **0.0**
- Epochs: **60**
- Learning rate: **0.001**
- Weight decay: **0.0**
- Input preprocessing: $x' = 2x-1$

Metrics:

**Validation**
- Loss: **2.6807**
- Accuracy: **0.1597**
- Macro-F1: **0.1267**
- Balanced accuracy: **0.1597**
- Top-2 accuracy: **0.2795**

**Test**
- Loss: **2.6829**
- Accuracy: **0.1302**
- Macro-F1: **0.1042**
- Balanced accuracy: **0.1302**
- Top-2 accuracy: **0.2674**

**Figure to insert:** *Deep MLP — confusion matrix (validation)*  
**Figure to insert:** *Deep MLP — confusion matrix (test)*  
**Figure to insert:** *Deep MLP — per-class F1 (test)*

---

### Discussion: what works and what does not

- Compared to Softmax Regression, the MLP increases representational power via nonlinear layers, but improvements are **limited**.  
- The main limitation is that the MLP flattens the image into a vector and cannot exploit **local texture patterns** and **translation invariance**.
- The low macro-F1 indicates that the model is not learning sufficiently discriminative features for many classes, even after increasing depth/width and testing dropout and regularization.
- This motivates CNNs, which are designed to learn local texture features through convolution and pooling.

---

### Performance considerations

We also benchmarked inference speed and model size for the selected deep MLP.

**Figure to insert:** *Deep MLP — quality vs performance (Macro-F1 vs latency)*

This allows a direct comparison of "cost vs quality" with the baseline linear model and the CNN.

---

### Summary

The MLP section demonstrates a fully manual implementation of a multi-layer neural network (forward + backward + SGD), and a proper validation-based hyperparameter search.  
Despite increasing depth, the MLP still underperforms strongly compared to CNNs, supporting the conclusion that spatial inductive biases (convolution/pooling) are essential for microscopy texture-based classification.


## Model 3 — Convolutional Neural Network (CNN, PyTorch layers allowed)

### Goal and motivation
The first two models (softmax regression and MLP trained "from scratch") operate on **flattened pixels** and therefore struggle to capture **local texture patterns** (edges, repeated micro-structures, spatial frequency) that are crucial for microscopy-based surface roughness recognition. A CNN is specifically designed to exploit **spatial locality** and **translation invariance** through convolutional filters and pooling, so it is expected to outperform both linear and fully-connected baselines on texture classification.

---

## CNN architecture (lecture-style CNN)
We implemented a standard CNN similar to the one discussed in the course:

**Feature extractor (3 convolutional blocks):**
1. `Conv2d(1 → 32, 3×3, padding=1)` → `ReLU` → `MaxPool2d(2)`
2. `Conv2d(32 → 64, 3×3, padding=1)` → `ReLU` → `MaxPool2d(2)`
3. `Conv2d(64 → 128, 3×3, padding=1)` → `ReLU` → `MaxPool2d(2)`

**Classification head:**
- `Flatten`
- `Linear(128·8·8 → 256)` → `ReLU`
- `Dropout(p=0.5)`
- `Linear(256 → 16)`

Because the input resolution is fixed at `64×64`, three pooling operations halve the spatial size each time:

$$64 \times 64 \xrightarrow{\text{pool}/2} 32 \times 32 \xrightarrow{\text{pool}/2} 16 \times 16 \xrightarrow{\text{pool}/2} 8 \times 8$$

Therefore the flattened feature vector has size:

$$128 \cdot 8 \cdot 8 = 8192$$

---

## Mathematical formulation

### Convolution layer
A convolution layer applies a set of learnable kernels (filters). For an input tensor $x$ and a kernel $w$, each output feature map is computed as:

$$z_{k}(i,j) = \sum_{c}\sum_{u,v} w_{k,c}(u,v)\,x_{c}(i+u, j+v) + b_k$$

where:
- $k$ is the output channel index,
- $c$ is the input channel index,
- $u,v$ iterate over the kernel spatial positions (here $3\times3$),
- $b_k$ is the bias term.

### ReLU activation

$$\text{ReLU}(z)=\max(0,z)$$

ReLU avoids saturation issues typical of sigmoid and improves gradient flow.

### Max pooling
Max pooling selects the maximum value in each spatial window (here $2\times2$):

$$y(i,j)=\max_{(u,v)\in \text{window}} x(2i+u,2j+v)$$

Pooling reduces resolution and provides local invariance, which is useful for texture variations and small shifts.

### Final classifier + softmax cross-entropy
The CNN produces **logits** $\mathbf{z} \in \mathbb{R}^{16}$. We train with multi-class softmax cross-entropy:

$$p_c = \frac{e^{z_c}}{\sum_{j=1}^{16} e^{z_j}},\qquad \mathcal{L} = -\log p_{y}$$

where $y$ is the true class label.

---

## Training setup and hyperparameter choices

### Allowed PyTorch components (as requested)
For the CNN, we used standard PyTorch modules and autograd (this was explicitly allowed):
- `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Linear`, `nn.Dropout`
- `nn.CrossEntropyLoss`
- `torch.optim.Adam`
- `loss.backward()` for gradients

### Optimization
We used Adam because it typically converges faster and more stably than vanilla SGD in CNN training:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

with bias-corrected first/second moment estimates $\hat{m}_t,\hat{v}_t$.

### Regularization: dropout and weight decay
- **Dropout** in the fully-connected head with $p=0.5$ reduces overfitting by randomly zeroing activations during training:

$$h' = m \odot h,\quad m_i \sim \text{Bernoulli}(1-p)$$

- **Weight decay** (L2 regularization) in Adam:

$$\mathcal{L}' = \mathcal{L} + \lambda \|\theta\|_2^2$$

helps prevent large weights and improves generalization.

### Early stopping (selection criterion)
To follow the course requirement of selecting hyperparameters based on validation performance, we used:
- **early stopping** with **patience = 8**
- validation metric: **Macro-F1**

Macro-F1 is appropriate because it gives equal importance to all classes:

$$\text{MacroF1}=\frac{1}{C}\sum_{c=1}^{C}F1_c$$

---

## Best CNN run (selected model)

**Configuration**
- Channels: `(32, 64, 128)`
- Dropout: `0.5`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Max epochs: `60`
- Early stopping patience: `8`

**Early stopping result**
- Best training epoch (highest val Macro-F1 during training): **epoch 31**
- Training stopped at epoch 39 due to no improvement for 8 epochs.

---

## Final performance (from your run)

### Validation set
- **Accuracy:** 0.5972  
- **Macro-F1:** 0.5944  
- **Balanced Accuracy:** 0.5972  
- **Top-2 Accuracy:** 0.8368  
- **Loss:** 1.3328  

### Test set
- **Accuracy:** 0.5920  
- **Macro-F1:** 0.5898  
- **Balanced Accuracy:** 0.5920  
- **Top-2 Accuracy:** 0.8247  
- **Loss:** 1.3352  

**Interpretation:** CNN dramatically improves over both pixel-based models because it learns discriminative local texture filters. The remaining generalization gap (train loss decreasing while val loss increases after the best epoch) is mitigated using dropout + weight decay + early stopping, but can be further improved with augmentation (see "possible improvements").

---

## Efficiency / performance metrics

### Model size
- **Parameters:** 2,194,192  
- **Approx. size:** 8.37 MB (float32)

### Training time
- **Avg sec/epoch:** 21.70 s  
- **Total training time:** 846.24 s (39 epochs executed)

### Inference benchmark (test loader)
- **Latency:** 4.05 ms / image  
- **Throughput:** 246.78 images/sec  

**Trade-off:** CNN is slower than shallow baselines, but the gain in Macro-F1 is very large, making it the best quality/cost trade-off for this task.

---

## What to include as figures in the report (placeholders)

1. **CNN Learning Curves (train loss, val loss, val accuracy, val macro-F1, val top-2)**  
   *Filename suggestion:* `cnn_learning_curves.png`

2. **CNN Confusion Matrix (Validation)**  
   *Filename suggestion:* `cnn_cm_val.png`

3. **CNN Confusion Matrix (Test)**  
   *Filename suggestion:* `cnn_cm_test.png`

4. **CNN Per-class F1 (Test)**  
   *Filename suggestion:* `cnn_f1_per_class_test.png`

5. **Quality vs Performance plot** (e.g., Macro-F1 vs inference latency across models)  
   *Filename suggestion:* `quality_vs_latency.png`

---

## Notes for professor-oriented discussion (why CNN works)
- Softmax regression assumes a **linear decision boundary in pixel space** and cannot represent texture patterns.
- MLP can represent non-linear boundaries, but flattening destroys spatial structure and the model lacks inductive bias for locality.
- CNN encodes the right inductive biases:
  - local receptive fields → learns micro-structures,
  - shared weights → detects patterns anywhere in the image,
  - pooling → robustness to small spatial shifts,
  - deeper hierarchy → edges → motifs → texture signatures.

---

## Possible improvements (optional extensions for discussion)
If required, we can additionally test:
- **Data augmentation** (small rotations, random crops, flips, brightness/contrast jitter) to improve generalization.
- **Batch normalization** after conv layers to stabilize optimization.
- Larger grids: alternative channel widths, dropout values, Adam vs SGD+momentum.
- Higher input resolution (e.g., 96×96) to preserve more texture detail (at higher compute cost).
