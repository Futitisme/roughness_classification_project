# Surface Roughness Classification using Neural Networks

## Project Overview

This project solves a **multi-class image classification problem**: classifying microscopic grayscale images of material surfaces into 16 discrete roughness categories. We compare three neural network architectures with increasing representational power:

1. **Softmax Regression** (linear baseline, implemented from scratch)
2. **Multi-Layer Perceptron** (MLP with sigmoid activation, implemented from scratch)
3. **Convolutional Neural Network** (CNN, using PyTorch layers)

Each model is trained and evaluated under a unified experimental protocol with systematic hyperparameter tuning based on validation performance.

---

# Data Preparation

## Raw Data

The dataset is the **Surface Roughness Classification** dataset from Kaggle containing microscopy images of machined surfaces.

| Property | Value |
|----------|-------|
| Total images | 3840 |
| Number of classes | 16 |
| Images per class | 240 (balanced) |
| Image format | JPEG |
| Class labels | `00`, `03`, `06`, ..., `45` |

## Label Encoding

Class names are sorted and mapped to integers 0-15:

```python
class_names = sorted(df["class_name"].unique())
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
df["label"] = df["class_name"].map(class_to_idx)
```

## Train / Validation / Test Split

The dataset is split with **stratified sampling** to preserve class distribution:

| Subset | Proportion | Images | Per class |
|--------|------------|--------|-----------|
| Train | 70% | 2688 | 168 |
| Validation | 15% | 576 | 36 |
| Test | 15% | 576 | 36 |

```python
df_trainval, df_test = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=SEED
)

val_ratio = 0.15 / 0.85  # ≈ 0.176
df_train, df_val = train_test_split(
    df_trainval, test_size=val_ratio, stratify=df_trainval["label"], random_state=SEED
)
```

## Image Preprocessing

All images undergo uniform preprocessing:

1. **Grayscale conversion** — color information is discarded since surface texture is the primary discriminative feature
2. **Resize to 64×64** — fixed resolution for batching
3. **Normalize to [0, 1]** — pixel values scaled from [0, 255] to floating-point [0, 1]
4. **Add channel dimension** — shape becomes (1, 64, 64) for PyTorch

```python
img = Image.open(path).convert("L")  # grayscale
img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
arr = np.asarray(img, dtype=np.float32) / 255.0
x = torch.from_numpy(arr).unsqueeze(0)  # shape: (1, H, W)
y = torch.tensor(label, dtype=torch.long)
```

---

# Model 1: Softmax Regression (Linear Baseline)

## Motivation

Softmax Regression is the **simplest discriminative model** for multi-class classification. It assumes classes are **linearly separable** in pixel space and does not exploit spatial structure. This establishes a **lower bound** on performance.

## Mathematical Formulation

### Forward Pass

Each image is flattened into a vector $\mathbf{x} \in \mathbb{R}^{D}$ where $D = 64 \times 64 = 4096$.

The model computes **logits** (raw scores) via a linear transformation:

$$\mathbf{z} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{D \times C}$, $\mathbf{b} \in \mathbb{R}^{C}$, and $C = 16$ classes.

### Softmax Function

The logits are converted to probabilities using the **softmax function**:

$$p(y=c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

For numerical stability, we use the log-sum-exp trick:

$$\log p_c = z_c - \log \sum_{j=1}^{C} e^{z_j}$$

### Cross-Entropy Loss

The model is trained to minimize **categorical cross-entropy**:

$$\mathcal{L} = -\frac{1}{B} \sum_{n=1}^{B} \log p(y_n \mid \mathbf{x}_n)$$

where $B$ is the batch size.

### Gradient Computation (Backpropagation)

The key insight is that the gradient of softmax + cross-entropy has a simple form. Let $\mathbf{P}$ be predicted probabilities and $\mathbf{Y}$ the one-hot encoded labels:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top (\mathbf{P} - \mathbf{Y}), \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{n} (\mathbf{P}_n - \mathbf{Y}_n)$$

### L2 Regularization (Weight Decay)

To prevent overfitting, we apply L2 penalty:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \left(\nabla_\mathbf{W} \mathcal{L} + \lambda \mathbf{W}\right)$$

## Implementation (From Scratch)

The entire model was implemented **without using `nn.Linear` or `nn.CrossEntropyLoss`**:

```python
def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax: subtract max for stability."""
    m = logits.max(dim=1, keepdim=True).values
    exps = torch.exp(logits - m)
    return exps / exps.sum(dim=1, keepdim=True)

def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class indices to one-hot vectors."""
    oh = torch.zeros((y.size(0), num_classes), device=y.device, dtype=torch.float32)
    oh[torch.arange(y.size(0), device=y.device), y] = 1.0
    return oh

class SoftmaxRegressionScratch:
    def __init__(self, input_dim: int, num_classes: int, seed: int, device: torch.device):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self.W = (torch.randn(input_dim, num_classes, generator=g) * 0.01).to(device)
        self.b = torch.zeros(num_classes, device=device)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        X = x.view(x.size(0), -1)  # flatten: (B, 1, H, W) -> (B, D)
        return X @ self.W + self.b

    def train_step(self, x: torch.Tensor, y: torch.Tensor, lr: float, weight_decay: float):
        B = x.size(0)
        X = x.view(B, -1)
        
        # Forward pass
        logits = X @ self.W + self.b
        probs = stable_softmax(logits)
        loss = -torch.log(probs[torch.arange(B), y] + 1e-9).mean()
        
        # Backward pass (manual gradients)
        dlogits = (probs - one_hot(y, NUM_CLASSES)) / B
        dW = X.t() @ dlogits
        db = dlogits.sum(dim=0)
        
        # L2 regularization
        if weight_decay > 0:
            dW = dW + weight_decay * self.W
        
        # SGD update
        self.W = self.W - lr * dW
        self.b = self.b - lr * db
        
        return float(loss.item())
```

## Hyperparameter Search

We performed grid search over:

- **Epochs**: {5, 10, 20, 35}
- **Learning rate**: {0.01, 0.003, 0.001}
- **Weight decay**: {0, 1e-4, 1e-3}

**Total: 36 configurations**. Selection criterion: **best validation Macro-F1**.

<img width="1083" alt="Softmax grid search results" src="https://github.com/user-attachments/assets/1909efff-13ed-437b-90c9-aec92b6c3b06" />

### Validation Macro-F1 by Learning Rate

<img width="591" alt="Validation F1 lr=0.01" src="https://github.com/user-attachments/assets/bbefd806-d191-442e-a991-a6d500f0d4d5" />
<img width="602" alt="Validation F1 lr=0.003" src="https://github.com/user-attachments/assets/52ee7f34-8681-4ce7-914f-34aebdf4e1a0" />
<img width="592" alt="Validation F1 lr=0.001" src="https://github.com/user-attachments/assets/4977315a-4a46-44b0-8506-57b84b4fbc85" />
<img width="566" alt="Validation F1 summary" src="https://github.com/user-attachments/assets/b9efc0ff-367a-4ad2-88a3-490319aff335" />

## Learning Dynamics (Best Configuration)

Best configuration: **epochs=35, lr=0.01, weight_decay=1e-4**

<img width="1131" alt="Softmax learning curves" src="https://github.com/user-attachments/assets/7883a14b-91e5-4c5c-8571-9bdff6d8bbc3" />

The curves show fast convergence but limited capacity to improve further — expected for a linear model.

## Final Performance

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | ~0.11 | ~0.13 |
| Macro-F1 | ~0.08 | ~0.10 |

<img width="455" alt="Softmax confusion matrix" src="https://github.com/user-attachments/assets/2c985f07-3f8a-4ec9-ada5-6a9eca631dd8" />

<img width="669" alt="Softmax test metrics" src="https://github.com/user-attachments/assets/d4a15c98-e365-4e57-9655-db9cc04d2dc7" />

The confusion matrix shows the model struggles to separate classes, confirming that **linear decision boundaries are insufficient** for texture classification.

## Computational Efficiency

- **Parameters**: ~65k
- **Model size**: ~0.25 MB
- **Inference latency**: < 1 ms/image

Despite low accuracy, Softmax Regression is extremely efficient — highlighting the trade-off between simplicity and representational power.

---

# Model 2: Multi-Layer Perceptron (MLP)

## Motivation

The MLP adds **non-linear capacity** through hidden layers with activation functions. Unlike the linear baseline, an MLP can learn **non-linear decision boundaries**. However, it still treats the image as a **flat vector** and does not model spatial locality.

## Architecture

The MLP has the following structure:

1. **Input**: Flatten image from (1, 64, 64) to vector of size 4096
2. **Hidden layers**: Multiple (Linear → Sigmoid → Dropout) blocks
3. **Output**: Linear layer producing 16 logits
4. **Output activation**: Softmax (mutually exclusive classes)

Best configuration from grid search:
- Hidden sizes: **(1024, 512)** — two hidden layers
- Dropout: **0.0**
- Input preprocessing: $x' = 2x - 1$ (maps [0,1] to [-1,1] for better gradient flow)

## Mathematical Formulation

### Linear Layer

For input $\mathbf{X} \in \mathbb{R}^{B \times d_{in}}$:

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$, $\mathbf{b} \in \mathbb{R}^{d_{out}}$.

### Sigmoid Activation

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid squashes values to (0, 1), introducing non-linearity. We use sigmoid (as discussed with the professor) because:
- It's a classic activation for educational purposes
- It demonstrates the vanishing gradient problem that ReLU solves

### Dropout (Inverted)

During training, randomly zero out neurons with probability $p$:

$$\tilde{\mathbf{H}} = \frac{\mathbf{H} \odot \mathbf{M}}{1-p}, \quad \mathbf{M}_i \sim \text{Bernoulli}(1-p)$$

The scaling by $1/(1-p)$ ensures expected values match at test time (when dropout is disabled).

### Softmax Cross-Entropy Loss

Same as Softmax Regression:

$$\mathcal{L} = -\frac{1}{B}\sum_{n=1}^{B}\log p_{n,y_n}$$

## Backpropagation (From Scratch)

All gradients were computed manually, without `torch.nn` layers or autograd.

### Softmax + Cross-Entropy Gradient

The gradient w.r.t. logits simplifies to:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} = \frac{\mathbf{P} - \mathbf{Y}}{B}$$

where $\mathbf{P}$ = predicted probabilities, $\mathbf{Y}$ = one-hot labels.

### Linear Layer Gradients

Given upstream gradient $\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^\top \mathbf{G}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{n} \mathbf{G}_n, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \mathbf{G}\mathbf{W}^\top$$

### Sigmoid Gradient

If $\mathbf{S} = \sigma(\mathbf{Z})$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \odot \mathbf{S}(1 - \mathbf{S})$$

This is the famous $\sigma'(z) = \sigma(z)(1-\sigma(z))$ property.

## Implementation (From Scratch)

The MLP uses PyTorch only as a **tensor library** (`matmul`, `exp`, etc.) and for data loading. All neural network components are manual:

```python
class LinearScratch:
    """Fully-connected layer with manual forward/backward."""
    
    def __init__(self, in_dim: int, out_dim: int, seed: int, device: torch.device):
        # Xavier initialization for better gradient flow
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self.W = torch.empty(in_dim, out_dim).uniform_(-limit, limit, generator=g).to(device)
        self.b = torch.zeros(out_dim, device=device)
        self.x = None  # cache for backward
        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x  # save for backward pass
        return x @ self.W + self.b

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        self.dW = self.x.t() @ dout      # gradient w.r.t. weights
        self.db = dout.sum(dim=0)         # gradient w.r.t. bias
        dx = dout @ self.W.t()            # gradient to propagate back
        return dx

    def sgd_step(self, lr: float, weight_decay: float):
        if weight_decay > 0:
            self.dW = self.dW + weight_decay * self.W
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db


class SigmoidScratch:
    """Sigmoid activation with manual backward."""
    
    def __init__(self):
        self.s = None  # cache sigmoid output
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = 1.0 / (1.0 + torch.exp(-z))
        self.s = s
        return s
    
    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        # σ'(z) = σ(z) * (1 - σ(z))
        return dout * self.s * (1.0 - self.s)


class DropoutScratch:
    """Inverted dropout (from scratch)."""
    
    def __init__(self, p: float):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            self.mask = None
            return x
        keep_prob = 1.0 - self.p
        self.mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        return x * self.mask / keep_prob  # scale to maintain expected value
    
    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return dout
        keep_prob = 1.0 - self.p
        return dout * self.mask / keep_prob


class MLPScratch:
    """Multi-layer perceptron: Flatten -> [Linear -> Sigmoid -> Dropout]×K -> Linear"""
    
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout_p, seed, device):
        self.layers = []
        cur = input_dim
        s = seed
        
        for h in hidden_sizes:
            self.layers.append(LinearScratch(cur, h, s, device))
            self.layers.append(SigmoidScratch())
            if dropout_p > 0:
                self.layers.append(DropoutScratch(dropout_p))
            cur = h
            s += 1
        
        # Output layer (no activation - logits)
        self.layers.append(LinearScratch(cur, num_classes, s, device))
    
    def forward_logits(self, x_img: torch.Tensor) -> torch.Tensor:
        # Center input: [0,1] -> [-1,1] for better sigmoid behavior
        x = (x_img * 2.0 - 1.0).view(x_img.size(0), -1)
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dlogits: torch.Tensor):
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)
    
    def sgd_step(self, lr: float, weight_decay: float):
        for layer in self.layers:
            if isinstance(layer, LinearScratch):
                layer.sgd_step(lr, weight_decay)
```

## Hyperparameter Search

We performed extensive grid search over **96 configurations**:

- **Hidden layouts**: (512, 256), (512, 512, 256), (1024, 512), (1024, 512, 256)
- **Dropout**: 0.0, 0.3, 0.5
- **Learning rate**: 0.001, 0.0003
- **Weight decay**: 0.0, 1e-4
- **Epochs**: 40, 60

Selection criterion: **best validation Macro-F1** during training.

<img width="699" alt="MLP grid search" src="https://github.com/user-attachments/assets/b263aec2-cab6-4e82-b76c-8490850d5e7d" />

## Final Results (Best Configuration)

Best configuration: **hidden=(1024, 512), dropout=0.0, lr=0.001, epochs=60**

| Metric | Validation | Test |
|--------|------------|------|
| Loss | 2.68 | 2.68 |
| Accuracy | 0.160 | 0.130 |
| Macro-F1 | 0.127 | 0.104 |
| Top-2 Accuracy | 0.280 | 0.267 |

### Learning Curves

<img width="1068" alt="MLP learning curves" src="https://github.com/user-attachments/assets/f1da18aa-282e-4ea3-84e6-73300642c96c" />

### Confusion Matrices

<img width="489" alt="MLP validation confusion matrix" src="https://github.com/user-attachments/assets/393d7a24-1676-4923-bd81-535215af530b" />

<img width="517" alt="MLP test confusion matrix" src="https://github.com/user-attachments/assets/49c7ec88-32f7-4f57-917e-ae3e24b659bb" />

## Analysis: Why MLP Struggles

The MLP improves over Softmax Regression due to non-linear capacity, but gains are **limited**:

1. **Flattening destroys spatial structure** — the model cannot exploit that nearby pixels are related
2. **No translation invariance** — a texture pattern in one location is treated differently than the same pattern elsewhere
3. **Sigmoid saturation** — gradients vanish when activations are near 0 or 1, making deep networks hard to train

This motivates CNNs, which are designed for image data.

## Computational Efficiency

- **Inference latency**: 1.16 ms/image
- **Throughput**: 859 images/sec

---

# Model 3: Convolutional Neural Network (CNN)

## Motivation

The first two models operate on **flattened pixels** and cannot capture **local texture patterns** (edges, micro-structures, spatial frequency). A CNN is designed to exploit:

- **Spatial locality** — nearby pixels are related
- **Translation invariance** — patterns should be detected regardless of position
- **Hierarchical features** — edges → textures → complex patterns

## Architecture

We implemented a standard CNN following the course structure:

**Feature Extractor (3 convolutional blocks):**

```
Conv2d(1 → 32, 3×3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, 3×3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(64 → 128, 3×3, padding=1) → ReLU → MaxPool2d(2)
```

**Classifier Head:**

```
Flatten → Linear(128·8·8 → 256) → ReLU → Dropout(0.5) → Linear(256 → 16)
```

**Spatial dimension evolution:**

$$64 \times 64 \xrightarrow{\text{pool}} 32 \times 32 \xrightarrow{\text{pool}} 16 \times 16 \xrightarrow{\text{pool}} 8 \times 8$$

Flattened feature size: $128 \times 8 \times 8 = 8192$

## Mathematical Formulation

### Convolution Operation

A convolution applies learnable filters to detect local patterns. For input $x$ and kernel $w$:

$$z_k(i,j) = \sum_{c}\sum_{u,v} w_{k,c}(u,v) \cdot x_c(i+u, j+v) + b_k$$

where:
- $k$ = output channel (filter index)
- $c$ = input channel
- $(u, v)$ = kernel position (3×3 in our case)
- $b_k$ = bias term

### ReLU Activation

$$\text{ReLU}(z) = \max(0, z)$$

ReLU avoids the vanishing gradient problem of sigmoid and allows deeper networks.

### Max Pooling

Selects the maximum value in each 2×2 window:

$$y(i,j) = \max_{(u,v) \in \text{window}} x(2i+u, 2j+v)$$

Benefits:
- Reduces spatial dimensions (computational efficiency)
- Provides local translation invariance
- Extracts dominant features

### Softmax Cross-Entropy

Same loss function as previous models:

$$\mathcal{L} = -\log \frac{e^{z_y}}{\sum_{j=1}^{16} e^{z_j}}$$

## Implementation (PyTorch)

For the CNN, we used standard PyTorch modules (as allowed by the professor):

```python
class CNNModel(nn.Module):
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
        
        # After 3 pools: 64→32→16→8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## Training Setup

**Optimizer**: Adam with adaptive learning rates

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam typically converges faster than vanilla SGD for CNNs.

**Regularization**:
- Dropout (p=0.5) in fully-connected layer
- Weight decay (L2) in optimizer

**Early stopping**: patience=8 epochs based on validation Macro-F1

## Best Configuration

- Channels: (32, 64, 128)
- Dropout: 0.5
- Learning rate: 1e-3
- Weight decay: 1e-4
- Max epochs: 60
- Early stopping patience: 8

Training stopped at epoch 39 (best epoch: 31).

## Final Performance

| Metric | Validation | Test |
|--------|------------|------|
| Loss | 1.33 | 1.34 |
| Accuracy | **0.597** | **0.592** |
| Macro-F1 | **0.594** | **0.590** |
| Top-2 Accuracy | 0.837 | 0.825 |

### Learning Curves

<img width="1065" alt="CNN learning curves" src="https://github.com/user-attachments/assets/c79e2fef-5e06-4286-bd5b-d41cd791d275" />

### Confusion Matrices

<img width="463" alt="CNN validation confusion matrix" src="https://github.com/user-attachments/assets/73e73d67-3169-46b8-b3af-b17cfe523482" />

<img width="479" alt="CNN test confusion matrix" src="https://github.com/user-attachments/assets/527aeada-bb08-4ac6-9112-9ee48f4f2e78" />

### Per-Class F1 Scores

<img width="1063" alt="CNN per-class F1" src="https://github.com/user-attachments/assets/603f0d10-7faa-482c-99f7-790a0bcbe049" />

## Computational Efficiency

| Metric | Value |
|--------|-------|
| Parameters | 2,194,192 |
| Model size | 8.37 MB |
| Training time | 846 sec (39 epochs) |
| Inference latency | 4.05 ms/image |
| Throughput | 247 images/sec |

CNN is slower than shallow baselines, but the **massive gain in Macro-F1** (0.59 vs 0.10) makes it the best quality/cost trade-off.

---

# Conclusion

## Model Comparison

We systematically compared three architectures with increasing complexity on the same classification task.

### Performance Comparison

<img width="1074" alt="All models comparison" src="https://github.com/user-attachments/assets/c63971c8-b1db-4bcf-8747-d0346ec5e9e0" />

### Per-Class F1 Across Models

<img width="1066" alt="Per-class F1 all models" src="https://github.com/user-attachments/assets/f7c126e7-40db-4c22-af5c-6c9b6146312b" />

### Quality vs Performance Trade-off

<img width="626" alt="Quality vs performance" src="https://github.com/user-attachments/assets/7b2da291-e7f1-4017-a291-be3b62e9913a" />

## Key Findings

| Model | Test Accuracy | Test Macro-F1 | Parameters | Latency |
|-------|---------------|---------------|------------|---------|
| Softmax Regression | 0.13 | 0.10 | ~65K | <1 ms |
| MLP (1024, 512) | 0.16 | 0.12 | ~5M | 1.2 ms |
| CNN (32-64-128) | **0.59** | **0.59** | ~2.2M | 4 ms |

## Analysis

### Softmax Regression

Implemented **fully from scratch** including forward pass, softmax normalization, cross-entropy loss, and manual gradient updates. The lowest performance is expected: softmax regression operates on flattened pixels and can only learn **linear decision boundaries**, making it fundamentally limited for texture-rich image data.

### Multi-Layer Perceptron

Also implemented **fully from scratch** with manual backpropagation. Despite non-linear capacity, improvements are limited because:
- The image is **flattened** before processing, destroying spatial structure
- No **translation invariance** — patterns are position-dependent
- Sigmoid activation suffers from **vanishing gradients**

### Convolutional Neural Network

Achieved the best performance by exploiting:
- **Local receptive fields** — learns texture micro-structures
- **Shared weights** — detects patterns anywhere in the image
- **Pooling** — robustness to small spatial shifts
- **Hierarchical features** — edges → motifs → texture signatures

## Theoretical Takeaways

1. **Linear models are insufficient** for complex visual tasks requiring texture discrimination
2. **Fully connected networks** benefit from non-linearity but struggle without spatial inductive bias
3. **Convolutional architectures** provide the best balance between representation power and generalization for image data

The CNN represents the **most appropriate model** for surface roughness classification, while Softmax Regression and MLP serve as valuable baselines demonstrating the importance of architectural inductive biases in deep learning.
