# Concept: Transfer Learning and Fine-Tuning (EfficientNetB0)

- **Concept File:** `docs/learning/transfer-learning-and-finetuning-efficientnetb0.md`
- **Related Task:** T5-022
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-29
- **Status:** Completed

---

## 1. Intuition

Training a deep convolutional neural network from randomly initialized weights requires tens or hundreds of thousands of diverse training images to converge onto robust visual features. In agricultural robotics and hydroponics, labeled leaf deficiency datasets typically contain hundreds to low thousands of images.

1. **Pretrained Feature Hierarchies:**
   Pretraining on large-scale datasets (such as ImageNet-1k with 1.28 million natural images) equips a neural network with a general-purpose hierarchical visual representation:
   - Early layers learn Gabor-like filters, color edges, and spatial gradients.
   - Middle layers compose textures, geometric contours, and surface patterns.
   - Deep layers synthesize complex object parts and class-specific shapes.
2. **Transfer Learning via Fine-Tuning:**
   Instead of discarding this visual knowledge, we transfer the pretrained backbone to our hydroponic domain:
   - Replace the original 1000-class ImageNet output head with a custom classification head sized for the specific crop's deficiency classes (e.g., Nitrogen, Phosphorus, Potassium, Iron, Calcium, Healthy).
   - Use differential learning rates: train the new classification head with a standard learning rate ($\eta_{\text{head}} \approx 10^{-3}$) while keeping the pretrained backbone frozen or updating it with a much smaller learning rate ($\eta_{\text{backbone}} \approx 10^{-4}$ or $10^{-5}$) to prevent catastrophic forgetting.
3. **EfficientNetB0:**
   EfficientNetB0 balances high accuracy with low computational footprint (5.3M parameters, 390M FLOPs), making it ideal for edge deployment on greenhouse gateway devices.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Compound Scaling Principle

Standard ConvNets arbitrarily scale network depth ($d$), width ($w$), or input resolution ($r$). Tan & Le (2019) demonstrated that balancing these three dimensions yields optimal parameter efficiency:

$$\text{Depth: } d = \alpha^\phi, \quad \text{Width: } w = \beta^\phi, \quad \text{Resolution: } r = \gamma^\phi$$

$$\text{subject to: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \ \beta \ge 1, \ \gamma \ge 1$$

where $\phi$ is the user-defined compound coefficient. For EfficientNetB0, $\phi = 1$ with baseline resolution $224 \times 224$.

### 2.2 Mobile Inverted Bottleneck Convolution (MBConv)

EfficientNet utilizes MBConv blocks with depthwise separable convolutions and Squeeze-and-Excitation (SE) channel attention:

1. **Expansion Phase:** Expands channel dimension from $C_{\text{in}}$ to $e \cdot C_{\text{in}}$ (expansion ratio $e \in \{1, 6\}$) using $1 \times 1$ pointwise convolution.
2. **Depthwise Phase:** Applies $k \times k$ spatial convolution independently per channel ($k \in \{3, 5\}$).
3. **Squeeze-and-Excitation (SE) Optimization:**
   - **Squeeze:** Global average pooling generates channel descriptor $\mathbf{z} \in \mathbb{R}^C$:
     $$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i, j)$$
   - **Excitation:** Two FC layers with reduction ratio $r=4$ and Swish / Sigmoid non-linearities compute channel scale weights $\mathbf{s}$:
     $$\mathbf{s} = \sigma\left(W_2 \cdot \text{SiLU}(W_1 \cdot \mathbf{z})\right)$$
   - Rescales the depthwise feature map: $\tilde{\mathbf{x}}_c = s_c \cdot \mathbf{x}_c$.
4. **Projection Phase:** Projects channels back down to $C_{\text{out}}$ with a linear $1 \times 1$ convolution.
5. **Residual Skip Connection:** Added if $C_{\text{in}} = C_{\text{out}}$ and stride is 1.

### 2.3 Head Replacement and Loss Function

For $K$ crop deficiency classes, the classification head projects the 1280-dimensional global pooled feature vector $\mathbf{v} \in \mathbb{R}^{1280}$:

$$\hat{\mathbf{y}} = \text{Softmax}\left(W_{\text{head}} \cdot \text{Dropout}(\mathbf{v}, p=0.2) + \mathbf{b}_{\text{head}}\right), \quad W_{\text{head}} \in \mathbb{R}^{K \times 1280}$$

Training minimizes categorical cross-entropy with optional class-weighting $\mathbf{w}$:

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} w_k \cdot y_{i, k} \log \hat{y}_{i, k}$$

---

## 3. From-Scratch Implementation

```python
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_efficientnet_model(
    num_classes: int,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Instantiate EfficientNetB0 transfer learning model with custom classification head.

    Args:
        num_classes: Number of target deficiency classes.
        freeze_backbone: If True, freezes all pretrained feature extractor layers.
        dropout_rate: Dropout probability in the classification head.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace final linear layer: EfficientNet-B0 classifier has [Dropout, Linear(1280, 1000)]
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model
```

---

## 4. Production & Library Usage

In `ml/image_models/efficientnet.py`, this architecture is wrapped with parameter groups for fine-tuning:

```python
def configure_fine_tuning_optimizer(
    model: nn.Module,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Set up AdamW optimizer with differential learning rates for head and backbone."""
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
```

In the Phase 5 benchmarks, EfficientNetB0 outperformed custom lightweight CNNs on validation macro-F1 across lettuce, spinach, and tomato, earning promotion to the project model registry (`ml/registry/`).

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Catastrophic Forgetting via High Backbone Learning Rate:**
   - *Failure:* Fine-tuning all layers with a high learning rate ($\eta = 10^{-2}$).
   - *Symptom:* Pretrained weights are drastically disrupted in the first epoch by noisy gradients from the uninitialized head, losing all ImageNet edge/texture representations.
   - *Fix:* Warm up the classification head with the backbone frozen for 2–3 epochs before unfreezing upper MBConv layers at $\eta \le 10^{-4}$.

2. **Batch Normalisation Running Stats with Small Batches:**
   - *Failure:* Unfreezing BatchNorm layers when training with mini-batch size $B \le 4$.
   - *Symptom:* Tiny batches cause noisy estimates of running mean $\hat{\mu}$ and running variance $\hat{\sigma}^2$, destabilizing inference.
   - *Fix:* Freeze BatchNorm layers (`model.eval()` for BN layers or `affine=True, track_running_stats=False`) when fine-tuning with tiny batch sizes.

3. **Input Resolution Mismatch:**
   - *Failure:* Passing arbitrary rectangular crops directly into EfficientNetB0 without padding or square resizing.
   - *Symptom:* Significant spatial aspect ratio distortion, flattening leaf geometry and altering texture frequency.
   - *Fix:* Pad crops to square before resizing to $224 \times 224$ in `ml/image_models/data.py`.

---

## 6. Where It Appears in This Project

- `ml/image_models/efficientnet.py`: Model definition, head replacement, and checkpoint loader.
- `ml/image_models/train.py`: Training pipeline supporting warm-up head training and fine-tuning.
- `scripts/verify_t5_022.py`: Verification suite checking weights, forward pass shapes, and parameter gradient flags.
- `tests/unit/test_efficientnet.py`: Unit tests verifying forward propagation and head dimensions.

---

## 7. Self-Check Questions

1. **Why does EfficientNetB0 outperform deeper networks like ResNet-101 on our small hydroponic dataset?**
   - *Answer:* EfficientNetB0 uses compound scaling and MBConv blocks to achieve high representational capacity with only 5.3 million parameters (compared to ResNet-101's 44.5M). On small training sets, over-parameterized models suffer severe empirical risk overfitting, whereas EfficientNetB0's compact parameter count provides stronger inductive regularization.

2. **What is the mathematical role of Squeeze-and-Excitation (SE) blocks in identifying foliar deficiencies?**
   - *Answer:* SE blocks explicitly model inter-channel dependencies. By computing a global average pooling vector and passing it through a bottleneck gating mechanism, the network adaptively recalibrates channel weights. For example, when detecting chlorosis, the SE block dynamically scales up feature maps corresponding to yellow/green color contrasts while suppressing background lighting noise.

3. **Why do we use differential learning rates ($\eta_{\text{head}} > \eta_{\text{backbone}}$) during fine-tuning?**
   - *Answer:* The classification head is initialized with random weights and requires large gradient steps ($\eta \approx 10^{-3}$) to converge toward the target class boundaries. The backbone, however, already contains delicate, pre-trained edge and texture representations; a large learning rate would destroy these weights (catastrophic forgetting). A small learning rate ($\eta \approx 10^{-4}$) allows subtle domain adaptation without loss of generic visual features.
