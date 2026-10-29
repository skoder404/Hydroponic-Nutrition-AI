# Concept: Colour Spaces (BGR/RGB/HSV) and Vegetation Indices (ExG, ExR)

- **Concept File:** `docs/learning/colour-spaces-bgrrgbhsv-and-vegetation-indices.md`
- **Related Task:** T5-001
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-29
- **Status:** Completed

---

## 1. Intuition

In computer vision for precision agriculture and hydroponics, raw pixel intensities carry diagnostic biological signals, but their representation heavily depends on the chosen color space:

1. **RGB vs BGR:** Standard digital imaging libraries represent images in Red-Green-Blue (RGB) channel order, whereas OpenCV historically reads images from disk into Blue-Green-Red (BGR) channel arrays. A naive handoff between OpenCV and PIL or Matplotlib without explicit color space conversion transposes the blue and red channels, causing green foliage to appear purplish-blue and completely invalidating downstream feature extractors.
2. **Decoupling Luminance from Chrominance (HSV):** In the Cartesian RGB space, illumination intensity is entangled across all three channels: a change in sunlight or LED grow-lamp intensity scales $R$, $G$, and $B$ simultaneously. In the cylindrical Hue-Saturation-Value (HSV) space, chrominance (Hue) and purity (Saturation) are separated from brightness (Value). This allows robust green-canopy segmentation across varying greenhouse light levels.
3. **Vegetation Indices (ExG, ExR):** To quantify foliar health and separate green biomass from non-plant background (e.g., perlite, rockwool, nutrient solution, PVC channels), normalized vegetation indices compare green channel dominance relative to red and blue.
   - **Excess Green ($\text{ExG}$)** highlights chlorophyll-rich leaf tissue.
   - **Excess Red ($\text{ExR}$)** isolates senescent, necrotic, or iron/magnesium-deficient reddish/brownish leaf regions.

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 RGB to HSV Conversion

Given normalized RGB coordinates $R, G, B \in [0, 1]$:
Let $V = \max(R, G, B)$, $M = \min(R, G, B)$, and $\Delta = V - M$.

1. **Value ($V$):**
   $$V = \max(R, G, B)$$

2. **Saturation ($S$):**
   $$S = \begin{cases} 0, & \text{if } V = 0 \\ \frac{\Delta}{V}, & \text{if } V > 0 \end{cases}$$

3. **Hue ($H \in [0^\circ, 360^\circ)$):**
   $$H = \begin{cases} 0^\circ, & \text{if } \Delta = 0 \\ 60^\circ \times \left(\frac{G - B}{\Delta} \bmod 6\right), & \text{if } V = R \\ 60^\circ \times \left(\frac{B - R}{\Delta} + 2\right), & \text{if } V = G \\ 60^\circ \times \left(\frac{R - G}{\Delta} + 4\right), & \text{if } V = B \end{cases}$$

In 8-bit OpenCV (`cv2.COLOR_BGR2HSV`), $H$ is scaled to $[0, 180]$ to fit into an 8-bit unsigned integer (`uint8`), while $S, V \in [0, 255]$. Green vegetation predominantly clusters around $H \in [35^\circ, 85^\circ]$ ($[18, 43]$ in OpenCV units).

### 2.2 Normalized Chromatic Coordinates

To decouple pixel color from total irradiance, normalized chromatic coordinates $(r, g, b)$ normalize each channel by total intensity:

$$r = \frac{R}{R + G + B + \epsilon}, \quad g = \frac{G}{R + G + B + \epsilon}, \quad b = \frac{B}{R + G + B + \epsilon}$$

where $\epsilon = 10^{-6}$ prevents division by zero in shadow or completely dark regions ($R=G=B=0$). Note that $r + g + b = 1$.

### 2.3 Vegetation Indices

1. **Excess Green Index ($\text{ExG}$):**
   Originally formulated by Woebbecke et al. (1995), $\text{ExG}$ measures green reflectance surplus over red and blue:
   $$\text{ExG} = 2g - r - b = \frac{2G - R - B}{R + G + B + \epsilon}$$
   For pure chlorophyll-rich tissue ($G \gg R, B$), $\text{ExG} > 0$. For soil, PVC piping, and water, $\text{ExG} \le 0$.

2. **Excess Red Index ($\text{ExR}$):**
   Quantifies red dominance associated with anthocyanin accumulation, tissue necrosis, or chlorotic yellowing:
   $$\text{ExR} = 1.4r - g = \frac{1.4R - G}{R + G + B + \epsilon}$$

3. **Excess Green minus Excess Red ($\text{ExG} - \text{ExR}$):**
   $$\text{ExG} - \text{ExR} = 2g - r - b - (1.4r - g) = 3g - 2.4r - b$$
   Provides high contrast segmentation between living green vegetation and brown/reddish decaying debris.

---

## 3. From-Scratch Implementation

```python
import numpy as np


def compute_vegetation_indices(rgb_image: np.ndarray) -> dict[str, np.ndarray]:
    """Compute normalized chromatic coordinates and vegetation indices from an RGB image.

    Args:
        rgb_image: uint8 numpy array of shape (H, W, 3) in RGB channel order.

    Returns:
        dict containing 'ExG', 'ExR', 'r', 'g', 'b' float32 arrays.
    """
    img_float = rgb_image.astype(np.float32)
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    total = R + G + B
    # Guard against zero-division in black pixels
    eps = 1e-6
    denom = np.where(total == 0, eps, total)

    r = R / denom
    g = G / denom
    b = B / denom

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g

    return {
        "ExG": exg,
        "ExR": exr,
        "ExG_minus_ExR": exg - exr,
        "r": r,
        "g": g,
        "b": b,
    }
```

---

## 4. Production & Library Usage

In production, OpenCV is utilized for fast vectorized matrix manipulation, while strictly guarding channel order conventions:

```python
import cv2
import numpy as np

# Load image from disk via OpenCV (native BGR)
bgr = cv2.imread("leaf_sample.jpg")

# 1. Mandatory conversion to RGB for color index extraction
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# 2. Conversion to HSV for illumination-invariant mask generation
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower_green = np.array([25, 40, 40], dtype=np.uint8)
upper_green = np.array([85, 255, 255], dtype=np.uint8)
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# 3. Mask refinement using Otsu threshold on ExG
indices = compute_vegetation_indices(rgb)
exg_norm = cv2.normalize(indices["ExG"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, otsu_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

In the project codebase, these routines are consolidated in `ml/features/common.py` and exercised across `ml/features/lettuce.py`, `ml/features/spinach.py`, and `ml/features/tomato.py`.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **The BGR vs RGB Swapping Bug:**
   - *Failure:* Calling `compute_vegetation_indices(bgr)` without converting BGR to RGB.
   - *Symptom:* Channel 0 ($B$) is treated as $R$, and Channel 2 ($R$) is treated as $B$. While $G$ remains channel 1, $\text{ExR} = 1.4B - G$. On leaves under blue-heavy LED grow lights, $\text{ExR}$ explodes and $\text{ExG}$ collapses, wrongly classifying vibrant green leaves as necrotic.
   - *Fix:* Enforce input assertions or use typed image containers verifying RGB channel alignment.

2. **Shadows and Deep Underexposure:**
   - *Failure:* In heavily shaded leaf understories, $R+G+B \approx 0$.
   - *Symptom:* Normalization creates extreme noise amplification: tiny digital sensor noise ($\pm 1$ LSB) becomes massive fluctuations in $r, g, b$.
   - *Fix:* Apply a brightness gate: if $V < 20$ in HSV (or $R+G+B < 30$), mask out the pixel as non-informative background.

3. **Specular Glare:**
   - *Failure:* Waxy leaf cuticles under direct grow LEDs create white specular reflections ($R \approx G \approx B \approx 255$).
   - *Symptom:* $r \approx g \approx b \approx 0.333$, yielding $\text{ExG} \approx 0.0$, causing holes in the segmented leaf mask.
   - *Fix:* Combine $\text{ExG}$ with morphological hole-filling or use SAM 2.1 boundary prompts.

---

## 6. Where It Appears in This Project

- `notebooks/04_color_spaces.ipynb`: Interactive demonstration of the BGR vs RGB bug and index distributions across deficiency classes.
- `ml/features/common.py`: Core `extract_color_features` and vegetation index utilities.
- `ml/features/lettuce.py`, `ml/features/spinach.py`, `ml/features/tomato.py`: Inclusion of mean $\text{ExG}$, mean $\text{ExR}$, and HSV channel statistics in the tabular feature vector for baseline classifiers.

---

## 7. Self-Check Questions

1. **Why does feeding a BGR image directly to an Excess Green function distort deficiency detection?**
   - *Answer:* In a BGR image, channel 0 is Blue and channel 2 is Red. While Green remains channel 1, any index relying on channel asymmetry (e.g. $\text{ExR} = 1.4r - g$ or channel-specific standard deviations) confuses red reflectance (necrosis/anthocyanin) with blue reflectance (water/glare). Under LED lighting with high blue spectral content, the leaf is erroneously flagged with severe necrotic excess red.

2. **What mathematical property makes normalized coordinates $(r, g, b)$ robust against uniform illumination changes?**
   - *Answer:* If ambient illumination scales pixel intensities by factor $\alpha > 0$ ($R' = \alpha R, G' = \alpha G, B' = \alpha B$), the scalar $\alpha$ cancels out in the ratio:
     $$g' = \frac{\alpha G}{\alpha R + \alpha G + \alpha B} = \frac{G}{R + G + B} = g$$
     Thus, normalized chromaticity is invariant to uniform scalar intensity scaling.

3. **How does Excess Green ($\text{ExG}$) distinguish healthy leaves from algae on wet rockwool?**
   - *Answer:* While both algae and leaves contain chlorophyll, algae on saturated rockwool exhibits different water-absorption characteristics in the near-infrared and red bands, and typically has a lower $\text{ExG} / \text{ExR}$ ratio and lower saturation $S$ compared to vascular plant foliage. When combined with HSV saturation thresholds, plant tissue separates cleanly from dilute surface algae.
