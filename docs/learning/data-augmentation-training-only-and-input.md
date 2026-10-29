# Concept: Data Augmentation (Training Only) and Input Normalisation

- **Concept File:** `docs/learning/data-augmentation-training-only-and-input.md`
- **Related Task:** T5-018
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-29
- **Status:** Completed

---

## 1. Intuition

Deep convolutional neural networks (CNNs) have millions of parameters and can readily memorize training images if data volume is limited. In agricultural vision, datasets often contain repeated views of plants taken from similar camera angles or fixed lighting rigs.

1. **Simulating Domain Variability:**
   Data augmentation generates stochastic, label-preserving transformations of training images on-the-fly. By introducing random horizontal flips, mild rotations, scaling, and photometric jitters during training, we force the network to learn invariant representations (e.g. recognizing interveinal chlorosis regardless of whether the leaf is tilted $15^\circ$ left or right).

2. **The "Training-Only" Iron Rule:**
   Augmentation must be applied **exclusively during the training phase**. Applying stochastic rotations or random color jitters during validation or testing corrupts evaluation benchmark consistency and introduces artificial evaluation noise. Validation and test sets must undergo strictly deterministic transformations: deterministic center crop / resize, followed by standard z-score normalization.

3. **Input Normalisation:**
   Neural network weights are initialized under the assumption that input activations have zero mean and unit variance. Normalizing inputs with standard ImageNet channel statistics ($\mu, \sigma$) prevents vanishing/exploding gradients during backpropagation and ensures smooth gradient descent across all feature channels.

---

## 2. Mathematical Formulation & Transformations

### 2.1 Geometric Transformations

Let $I(x, y)$ be an input image tensor:

1. **Random Horizontal Flip:**
   $$I_{\text{flip}}(x, y) = I(W - 1 - x, y) \quad \text{with probability } p = 0.5$$

2. **Affine Rotation by angle $\theta \sim \mathcal{U}(-\theta_{\max}, +\theta_{\max})$:**
   $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} x - x_c \\ y - y_c \end{bmatrix} + \begin{bmatrix} x_c \\ y_c \end{bmatrix}$$
   where $(x_c, y_c)$ is the image center, and pixel values at non-integer coordinates are computed via bilinear interpolation:
   $$I(x', y') = \sum_{i, j} I(i, j) \max(0, 1 - |x' - i|) \max(0, 1 - |y' - j|)$$

### 2.2 Photometric Transformations (Color Jitter)

Color jitter alters brightness, contrast, and saturation within safe agronomic bounds:

1. **Brightness scaling:**
   $$I_{\text{bright}}(x, y) = \alpha \cdot I(x, y), \quad \alpha \sim \mathcal{U}(1 - \delta_b, 1 + \delta_b)$$

2. **Contrast scaling:**
   $$I_{\text{contrast}}(x, y) = \beta \cdot I(x, y) + (1 - \beta) \cdot \mu_{\text{gray}}, \quad \beta \sim \mathcal{U}(1 - \delta_c, 1 + \delta_c)$$

> [!WARNING]
> **Agronomic Hue Limitation:** Extreme hue shifts ($\Delta H > 0.05$) must be strictly avoided. A large shift in hue will turn a healthy green leaf yellow (mimicking nitrogen deficiency) or purple (mimicking phosphorus deficiency), actively destroying the biological ground truth.

### 2.3 Channel Standardization (Input Normalisation)

Given an RGB image rescaled to float values $x \in [0.0, 1.0]^3$, channel-wise standard normalisation applies:

$$z_c(x, y) = \frac{x_c(x, y) - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}$$

For models pretrained on ImageNet (such as EfficientNetB0):
$$\boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$

---

## 3. From-Scratch Implementation

```python
import torch
from torchvision import transforms


def get_image_transforms(
    image_size: tuple[int, int] = (224, 224),
    is_training: bool = True,
) -> transforms.Compose:
    """Build PyTorch transformation pipelines adhering to the training-only augmentation rule.

    Args:
        image_size: Target (height, width) tuple.
        is_training: If True, applies stochastic augmentations; if False, applies deterministic resize only.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Controlled color jitter: mild brightness/contrast, negligible hue perturbation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
```

---

## 4. Production & Library Usage

In `ml/image_models/data.py`, this pattern is implemented in `HydroponicLeafDataset`:

```python
class HydroponicLeafDataset(Dataset):
    """PyTorch Dataset loading leaf crops from manifest with train/val transform separation."""

    def __init__(self, manifest_df, image_dir, transform=None):
        self.df = manifest_df
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["crop_relpath"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_to_idx[row["deficiency_class"]]
        return image, label
```

The split loader factory ensures that the training DataLoader receives the stochastic training transform, while the validation and test DataLoaders receive the deterministic evaluation transform.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Evaluation Leakage via Shared Transforms:**
   - *Failure:* Using a single global transform object with `RandomHorizontalFlip` for both train and validation datasets.
   - *Symptom:* Validation loss and macro-F1 fluctuate erratically across epochs because the validation images are constantly mutating underneath the metric calculator.
   - *Fix:* Instantiate distinct `train_transform` and `eval_transform` pipelines in dataset factory functions.

2. **Double Normalisation:**
   - *Failure:* Applying `ToTensor()` (which rescales $[0, 255] \to [0.0, 1.0]$) after manual division by $255.0$.
   - *Symptom:* Input pixels fall into $[0.0, 0.0039]$, causing severe underexposure and zeroed activations.
   - *Fix:* Rely on `transforms.ToTensor()` to handle $[0, 255] \to [0.0, 1.0]$ conversion directly from PIL `Image`.

3. **Loss of Diagnostic Pathology via Hue Augmentation:**
   - *Failure:* Setting `transforms.ColorJitter(hue=0.5)`.
   - *Symptom:* A healthy green leaf is rotated $180^\circ$ in color space into magenta/red. The model receives a healthy label for a red leaf, inducing catastrophic label noise and degraded validation F1.
   - *Fix:* Bound `hue` to $\le 0.02$ or disable hue alteration entirely in agronomic pipelines.

---

## 6. Where It Appears in This Project

- `ml/image_models/data.py`: `HydroponicLeafDataset` and augmentation pipeline definitions.
- `scripts/verify_t5_018.py`: Verification script testing data loaders and transform consistency.
- `tests/unit/test_image_datasets.py`: Unit tests validating deterministic behavior on validation sets and correct normalization tensors.

---

## 7. Self-Check Questions

1. **Why is random data augmentation strictly prohibited on the validation and test splits?**
   - *Answer:* The validation and test sets must provide an objective, fixed benchmark to compare model checkpoints and detect overfitting. If stochastic transformations are applied to validation samples, metric fluctuations will reflect random transform noise rather than actual learning progress, obscuring early stopping decisions.

2. **What happens to backpropagation gradients if input images are passed unnormalized in $[0, 255]$?**
   - *Answer:* Inputs with values up to 255 cause first-layer pre-activations ($z = Wx + b$) to become extremely large in magnitude. In saturated activation regions or deep layers, this leads to exploding gradients or extreme gradient instability, requiring impractically tiny learning rates and slowing down training convergence.

3. **Why must hue jitter be constrained far more tightly in crop disease classification than in generic ImageNet classification?**
   - *Answer:* In generic object recognition (e.g., cars, dogs), color is often secondary to structural shape. In plant pathology, however, color is the primary diagnostic symptom: chlorosis (yellowing), necrosis (browning), and anthocyanin accumulation (purpling) are the exact biochemical markers separating nutrient deficiencies. Large hue shifts corrupt these biological signals.
