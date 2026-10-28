# Concept: Texture Features: GLCM Contrast and Entropy

- **Concept File:** `docs/learning/texture-features-glcm-contrast-and-entropy.md`
- **Related Task:** T5-003
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-28
- **Status:** Completed

---

## 1. Intuition

Gray-Level Co-occurrence Matrices (GLCM), originally introduced by Robert Haralick et al. (1973), are second-order statistical texture descriptors. While first-order statistics (such as histograms, mean, or variance) summarize how often individual gray levels occur regardless of spatial arrangement, second-order statistics capture spatial relationships: *how often does a pixel with gray level $i$ appear at a specific spatial displacement $(\Delta r, \Delta c)$ relative to a pixel with gray level $j$?*

In hydroponic leaf monitoring:
- A smooth, healthy leaf surface exhibits minimal high-frequency intensity shifts between neighboring pixels. Most neighboring pixels have nearly identical intensity ($i \approx j$), concentrating probability mass along the principal diagonal of the co-occurrence matrix.
- Conversely, nutrient deficiencies (e.g., magnesium or iron chlorosis, tip burn, necrosis) generate mottled, chlorotic spots or necrotic lesions, introducing sharp transitions between dark veins and pale interveinal areas ($i \neq j$).
- **Contrast** measures local spatial intensity variation (the spread away from the diagonal). High contrast signifies jagged, mottled, or rough texture.
- **Entropy** measures the disorder, randomness, or complexity of the gray-level transitions. A leaf with uniform coloration has low entropy (approaching zero), whereas a diseased leaf displaying patchy chlorosis exhibits high entropy because transitions are widely distributed across gray levels.

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 The Co-occurrence Matrix $P(i, j; d, \theta)$

Let $I(r, c)$ be a quantized 2D grayscale image with discrete intensity values $i, j \in \{0, 1, \dots, N-1\}$. For a displacement distance $d$ and angle $\theta$, the co-occurrence count matrix $C(i, j; d, \theta)$ tallies transitions:

$$
C(i, j; d, \theta) = \sum_{r} \sum_{c} \mathbf{1}\big(I(r, c) = i \ \land \ I(r + \Delta r, c + \Delta c) = j\big)
$$

where $(\Delta r, \Delta c)$ is defined by distance $d$ and direction $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$:
- $\theta = 0^\circ$: $(\Delta r, \Delta c) = (0, d)$ (horizontal)
- $\theta = 45^\circ$: $(\Delta r, \Delta c) = (-d, d)$ (anti-diagonal / up-right)
- $\theta = 90^\circ$: $(\Delta r, \Delta c) = (-d, 0)$ (vertical)
- $\theta = 135^\circ$: $(\Delta r, \Delta c) = (-d, -d)$ (main-diagonal / up-left)

### 2.2 Flags: `symmetric=True` and `normed=True`

1. **`symmetric=True`**:
   The matrix counts transitions symmetrically: for every pair $(i \to j)$, the reverse transition $(j \to i)$ is also accumulated:
   $$
   C_{\text{sym}}(i, j) = C(i, j) + C(j, i)
   $$
   This ensures $P(i, j) = P(j, i)$, rendering the texture metric invariant to the forward vs. backward raster scanning direction.

2. **`normed=True`**:
   The matrix is normalized by the total count of all pairs:
   $$
   P(i, j) = \frac{C_{\text{sym}}(i, j)}{\sum_{i=0}^{N-1} \sum_{j=0}^{N-1} C_{\text{sym}}(i, j)}, \quad \text{such that} \quad \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} P(i, j) = 1.0
   $$
   $P(i, j)$ represents the empirical joint probability mass function of adjacent pixel values.

### 2.3 Feature Metrics

#### GLCM Contrast
Contrast calculates the expected squared difference between neighbor pixel intensities:

$$
\text{Contrast}_\theta = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} |i - j|^2 P(i, j; d, \theta)
$$

The final reported feature is the arithmetic mean across the four canonical angles:
$$
\text{glcm\_contrast} = \frac{1}{4} \sum_{\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}} \text{Contrast}_\theta
$$

#### GLCM Entropy
Entropy computes the Shannon information disorder of the joint probability distribution using logarithm base 2 ($\log_2$), measuring entropy in **bits (shannons)**:

$$
\text{Entropy}_\theta = - \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} P(i, j; d, \theta) \log_2\big(P(i, j; d, \theta)\big) \quad \text{for } P(i, j) > 0
$$

By continuity, $\lim_{p \to 0^+} p \log_2(p) = 0$, so cells with zero probability are excluded. The final reported feature is the mean across the four angles:
$$
\text{glcm\_entropy} = \frac{1}{4} \sum_{\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}} \text{Entropy}_\theta
$$

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

```python
import numpy as np


def compute_glcm_from_scratch(
    image: np.ndarray,
    levels: int = 32,
    distance: int = 1,
    angles: tuple = (0, 45, 90, 135),
    symmetric: bool = True,
    normed: bool = True,
):
    """Compute GLCM, contrast, and base-2 entropy from scratch without external libraries."""
    h, w = image.shape
    # Mapping angles to (dr, dc)
    angle_offsets = {
        0: (0, distance),
        45: (-distance, distance),
        90: (-distance, 0),
        135: (-distance, -distance),
    }

    n_angles = len(angles)
    glcm = np.zeros((levels, levels, n_angles), dtype=np.float64)

    for a_idx, deg in enumerate(angles):
        dr, dc = angle_offsets[deg]
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    i = int(image[r, c])
                    j = int(image[nr, nc])
                    glcm[i, j, a_idx] += 1.0
                    if symmetric:
                        glcm[j, i, a_idx] += 1.0

        if normed and glcm[:, :, a_idx].sum() > 0:
            glcm[:, :, a_idx] /= glcm[:, :, a_idx].sum()

    # Contrast & Entropy
    contrasts = []
    entropies = []
    for a_idx in range(n_angles):
        p = glcm[:, :, a_idx]
        # Contrast = sum |i - j|^2 * p(i, j)
        grid_i, grid_j = np.indices((levels, levels))
        c_val = np.sum(((grid_i - grid_j) ** 2) * p)
        contrasts.append(c_val)

        # Entropy = - sum p * log2(p) for p > 0
        nz = p[p > 0]
        h_val = -np.sum(nz * np.log2(nz)) if len(nz) > 0 else 0.0
        entropies.append(max(0.0, float(h_val)))

    return {
        "glcm_contrast": float(np.mean(contrasts)),
        "glcm_entropy": float(np.mean(entropies)),
    }


if __name__ == "__main__":
    # Hand-computed 4x4 block example:
    toy_4x4 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3]
    ], dtype=np.uint8)

    res = compute_glcm_from_scratch(toy_4x4, levels=32)
    print(f"Scratch GLCM Contrast: {res['glcm_contrast']:.6f}")  # 1.250000
    print(f"Scratch GLCM Entropy:  {res['glcm_entropy']:.6f}")   # 3.321888
    assert abs(res["glcm_contrast"] - 1.25) < 1e-5
    assert abs(res["glcm_entropy"] - 3.321888) < 1e-5
    print("All scratch assertions passed!")
```

---

## 4. Production & Library Usage

In production (`ml/features/glcm.py`), we combine `skimage.feature.graycomatrix` and `graycoprops` with a robust base-2 entropy accumulator:

```python
from ml.features.glcm import extract_glcm_features

# Given an RGB leaf image (H, W, 3) and optional segmentation mask (H, W):
features = extract_glcm_features(
    image=rgb_image,
    mask=leaf_mask,
    levels=32,
    distances=(1,),
    angles=(0.0, 0.785398, 1.570796, 2.356194),
    log_base=2.0,
)
# Returns: {'glcm_contrast': 0.8412, 'glcm_entropy': 2.9150}
```

**Key Hyperparameters & Parameters:**
- `levels = 32`: Quantizing from 256 to 32 levels compresses the co-occurrence matrix from $256 \times 256 = 65,536$ cells down to $32 \times 32 = 1,024$ cells. This prevents extreme matrix sparsity on small leaf patches while retaining essential textural gradients.
- `distances = (1,)`: A 1-pixel neighborhood captures fine-grained leaf vein and stippling texture without skipping across micro-lesions.
- `angles = (0, pi/4, pi/2, 3*pi/4)`: Four directions averaged together yield rotational invariance, preventing camera angle or leaf orientation in the hydroponic trough from skewing the model.
- `log_base = 2.0`: Standard Shannon entropy in bits. Ensures direct comparability across datasets and adheres to Overview §16.4.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Constant-Colour Image (Zero Texture):**
   - *Failure Mode:* If a synthetic or heavily saturated crop contains identical pixels across the ROI, all transitions land on $(v, v)$ with probability $1.0$. Computing $\log(0)$ on empty cells causes an unhandled floating point exception or `NaN`.
   - *Mitigation:* Strict slicing `p[p > 0]` ensures $\log$ is only evaluated on non-zero entries. For constant color, $1.0 \times \log_2(1.0) = 0.0$, producing `contrast = 0.0` and `entropy = 0.0` without NaN or crash.
2. **Empty Mask:**
   - *Failure Mode:* When a background segmenter produces zero foreground pixels, GLCM computation receives an empty array, leading to dimension crashes or zero-division during normalization.
   - *Mitigation:* `extract_glcm_features` validates that $\sum \text{mask} > 0$ and raises typed `EmptyMaskError`.
3. **Sparsity from Over-Quantization:**
   - *Failure Mode:* Using `levels=256` on a small ROI ($64 \times 64 = 4096$ pixels) spreads samples over 65,536 cells, resulting in >95% zero probability and noisy entropy estimates.
   - *Mitigation:* Fixed `levels=32` guarantees substantial sample density per active cell.

---

## 6. Where It Appears in This Project

- **Feature Implementation:** [`ml/features/glcm.py`](file:///d:/Hydroponic-Nutrition-AI/ml/features/glcm.py)
- **Shared Module Export:** [`ml/features/__init__.py`](file:///d:/Hydroponic-Nutrition-AI/ml/features/__init__.py)
- **Spinach Feature Extractor (T5-005):** Required feature `glcm_contrast` in the 6-feature spinach vector.
- **Verification Script:** [`scripts/verify_glcm.py`](file:///d:/Hydroponic-Nutrition-AI/scripts/verify_glcm.py)
- **Unit Tests:** [`tests/unit/test_glcm.py`](file:///d:/Hydroponic-Nutrition-AI/tests/unit/test_glcm.py)
- **Design Specifications:** Overview §16.4 defaults (quantise 32, distance 1, 4 angles averaged, symmetric/normed True, log base 2).

---

## 7. Self-Check Questions (3–5 Questions)

1. **Q1: Why is `symmetric=True` preferred over non-symmetric co-occurrence matrices for leaf texture analysis?**
   *Answer:* It guarantees $P(i, j) = P(j, i)$, making the texture representation invariant to whether the image is scanned left-to-right or right-to-left. A horizontal vein transition from dark to light is treated symmetrically to light to dark.
2. **Q2: Why do we choose `levels=32` instead of native 256 grayscale levels?**
   *Answer:* A 256-level GLCM requires a $256 \times 256 = 65,536$-element matrix, which becomes excessively sparse and noisy on typical leaf crops ($100 \times 100$ to $500 \times 500$). Quantizing to 32 levels reduces the state space to $1,024$ cells, providing robust probability estimates while preserving contrast and entropy fidelity.
3. **Q3: What are the theoretical minimum and maximum values of GLCM contrast?**
   *Answer:* The minimum is $0.0$ (when all pixels are identical). The theoretical maximum on 32 levels is $(31 - 0)^2 = 961.0$ (occurring only in a checkerboard of pure 0 and pure 31).
4. **Q4: Why does a completely uniform (constant) image have an entropy of 0.0 bits?**
   *Answer:* Because the distribution is deterministic: exactly one cell has probability $P(v, v) = 1.0$ and all others are $0$. $H = - 1.0 \times \log_2(1.0) = 0.0$ bits. Zero uncertainty corresponds to zero entropy.
5. **Q5: What happens if an image feature extractor returns `NaN` in this codebase?**
   *Answer:* Per T5-002 and project rules, `assert_finite_features` raises a typed `NonFiniteFeatureError`. Features are strictly never replaced by 0 or silent fallbacks.
