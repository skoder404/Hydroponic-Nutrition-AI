# Concept: Shape Descriptors: Compactness, Solidity, Aspect Ratio

- **Concept File:** `docs/learning/shape-descriptors-compactness-solidity-aspect-ratio.md`
- **Related Task:** T5-007
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-29
- **Status:** Completed

---

## 1. Intuition

Leaf morphology and geometric boundary contours reflect plant turgor pressure, structural integrity, and nutrient health:

1. **Hydraulic Turgor and Wilting:** Adequate potassium and water maintain guard cell turgor and petiole rigidity. Severe potassium deficiency or water stress leads to flaccid lamina, edge curling, and wilting, distorting the 2D projected boundary of the leaf.
2. **Necrotic Notching and Tip Burn:** Calcium deficiency causes localized cell wall collapse at rapidly expanding margins (tip burn in lettuce, blossom end rot / apical die-back in tomato), creating serrated, ragged notches that increase perimeter relative to enclosed area.
3. **Geometric Descriptors:** To capture these morphological changes without relying on absolute pixel dimensions (which depend on camera distance), scale-invariant shape descriptors are computed from the segmented binary mask contour:
   - **Compactness (Isoperimetric Ratio / Form Factor):** Measures perimeter roughness and deviation from a smooth circle or oval.
   - **Solidity:** Measures the ratio of contour area to its convex hull, detecting marginal indentations and leaf lobe deep notches.
   - **Aspect Ratio:** Measures elongation versus roundness via the minimum enclosing bounding box.
   - **Droop Angle (Evaluation and Rejection):** Assessed in T5-006 to determine whether 2D petiole droop could reliably quantify wilting without calibrated 3D depth; rejected due to camera perspective distortion.

---

## 2. Mathematical Formulation & Formal Definition

Let $C = \{(x_k, y_k)\}_{k=1}^{M}$ be the ordered sequence of 2D boundary coordinates defining the closed outer contour of a segmented leaf mask.

### 2.1 Contour Area ($A$) and Perimeter ($P$)

By Green's Theorem, the enclosed area $A$ is given by the discrete shoelace formula:

$$A = \frac{1}{2} \left| \sum_{k=1}^{M} (x_k y_{k+1} - x_{k+1} y_k) \right|$$

with $(x_{M+1}, y_{M+1}) = (x_1, y_1)$.

The perimeter $P$ is the cumulative Euclidean arc length along adjacent contour vertices:

$$P = \sum_{k=1}^{M} \sqrt{(x_{k+1} - x_k)^2 + (y_{k+1} - y_k)^2}$$

### 2.2 Compactness (Form Factor / Circularity)

The isoperimetric quotient compares the contour's area-to-perimeter ratio to that of a perfect circle (which attains maximal area for a given perimeter):

$$\text{Compactness} = \frac{4\pi A}{P^2}$$

- For a circle: $\text{Compactness} = 1.0$.
- For an elongated or ragged contour: $P$ increases significantly while $A$ stays constant or decreases, causing $\text{Compactness} \to 0$.
- In some literature, compactness is defined as $P^2 / A$ (non-normalized, minimum $4\pi \approx 12.57$). In our project (`ml/features/common.py`), we standardize on the normalized $[0, 1]$ metric $\frac{4\pi A}{P^2}$.

### 2.3 Convex Hull and Solidity

The convex hull $\text{Conv}(C)$ is the smallest convex polygon containing all points in $C$.
Let $A_{\text{convex}}$ be the area of $\text{Conv}(C)$.

$$\text{Solidity} = \frac{A}{A_{\text{convex}}}$$

- Since $A \le A_{\text{convex}}$ for any shape, $\text{Solidity} \in (0, 1.0]$.
- A convex leaf (e.g., broad oval spinach leaf) has $\text{Solidity} \approx 0.95 - 0.99$.
- A compound tomato leaf or a wilting leaf with curled, folded margins exhibits deep concavities, lowering $\text{Solidity}$ to $0.60 - 0.80$.

### 2.4 Aspect Ratio and Extent

Let $(w, h)$ be the width and height of the minimum-area rotated bounding rectangle enclosing the contour:

$$\text{Aspect Ratio} = \frac{\max(w, h)}{\min(w, h)}$$

$$\text{Extent} = \frac{A}{w \times h}$$

$\text{Aspect Ratio} \ge 1.0$ describes the leaf's elongation along its primary venation axis.

### 2.5 The Droop Angle Evaluation (T5-006 Decision)

In T5-006, the engineering team evaluated whether `droop_angle` (the angle of the principal inertia axis relative to the horizontal plane) should be retained as an agronomic feature:
- *Finding:* In unconstrained top-down greenhouse cameras, the 2D angle of petiole tilt is severely confounded by the plant's azimuth angle relative to the optical center. A leaf hanging downward at $45^\circ$ directly toward the camera appears identical in 2D projection to a rigid horizontal leaf.
- *Decision:* Droop angle was removed as an uncalibrated 2D feature to prevent severe perspective artifacts from corrupting classifier training.

---

## 3. From-Scratch Implementation

```python
import cv2
import numpy as np


def compute_shape_descriptors(mask: np.ndarray) -> dict[str, float]:
    """Compute scale-invariant shape descriptors from a binary segmentation mask.

    Args:
        mask: uint8 binary mask of shape (H, W) where 255 indicates leaf tissue.

    Returns:
        dict containing 'area', 'perimeter', 'compactness', 'solidity', 'aspect_ratio'.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "solidity": 0.0,
            "aspect_ratio": 1.0,
        }

    # Take the dominant leaf contour
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))

    if perimeter == 0.0 or area == 0.0:
        return {
            "area": area,
            "perimeter": perimeter,
            "compactness": 0.0,
            "solidity": 0.0,
            "aspect_ratio": 1.0,
        }

    # 1. Compactness (normalized isoperimetric quotient)
    compactness = (4.0 * np.pi * area) / (perimeter ** 2)
    compactness = float(np.clip(compactness, 0.0, 1.0))

    # 2. Solidity
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    solidity = float(np.clip(solidity, 0.0, 1.0))

    # 3. Aspect Ratio from minimum area bounding box
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if min(w, h) > 0:
        aspect_ratio = float(max(w, h) / min(w, h))
    else:
        aspect_ratio = 1.0

    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
    }
```

---

## 4. Production & Library Usage

In `ml/features/tomato.py`, shape features are extracted as part of the 7-feature descriptor suite:

```python
from ml.features.common import extract_contour_descriptors, extract_color_features

def extract_tomato_features(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract 7-dimensional feature vector for tomato deficiency classification."""
    color_feats = extract_color_features(image_rgb, mask)  # ExG, ExR, H, S
    shape_feats = extract_contour_descriptors(mask)        # compactness, solidity, aspect_ratio

    # Vector: [ExG, ExR, H_mean, S_mean, compactness, solidity, aspect_ratio]
    return np.hstack([color_feats, shape_feats])
```

Unit tests in `tests/unit/test_tomato_features.py` verify that artificial circular masks yield `compactness > 0.9` and `solidity > 0.95`, while synthetic jagged star masks yield `compactness < 0.5`.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Sub-Pixel Discretization on Tiny Leaves:**
   - *Failure:* On very small segmented leaf crops ($< 30 \times 30$ pixels), raster grid quantization causes discrete contour corners to artificially inflate perimeter $P$.
   - *Symptom:* Compactness on small circular masks can drop to $0.70$ due to pixel staircasing.
   - *Fix:* Enforce minimum leaf size filtering in `ml/segmentation/` (`min_area >= 500` pixels) or use Freeman chain code smoothing (`cv2.approxPolyDP`).

2. **Occlusion and Boundary Truncation:**
   - *Failure:* If a leaf touches the image frame edge, its contour is clipped by the camera boundary.
   - *Symptom:* The artificial flat edge dramatically distorts convex hull calculation, yielding false low solidity.
   - *Fix:* Filter out border-touching leaf masks during multi-leaf plant instance extraction.

3. **Multiple Fragmented Mask Islands:**
   - *Failure:* Severe necrotic holes can cause SAM 2.1 to split a single leaf into disjoint binary blobs.
   - *Symptom:* Calculating contour area on `contours[0]` only captures a fraction of the leaf.
   - *Fix:* Use morphological closing (`cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)`) before extracting outer contours, or select the maximal contour by area.

---

## 6. Where It Appears in This Project

- `ml/features/common.py`: Shared morphological descriptor extractors.
- `ml/features/tomato.py`: Tomato feature extraction module (implementing the T5-006 droop angle decision).
- `scripts/verify_tomato_features.py`: Automated validation of contour invariants and numerical stability.
- `tests/unit/test_tomato_features.py`: Unit test coverage ensuring scale invariance.

---

## 7. Self-Check Questions

1. **Why is compactness defined as $\frac{4\pi A}{P^2}$ rather than simply $P / A$?**
   - *Answer:* $P / A$ has units of $\text{length}^{-1}$ and therefore scales inversely with the size of the object: doubling an image's resolution or moving the camera closer halves $P / A$. The formulation $\frac{4\pi A}{P^2}$ has units of $\text{length}^2 / \text{length}^2 = 1$ (dimensionless), making it strictly scale-invariant across camera distances.

2. **Why was droop angle removed from the tomato feature extractor in T5-006?**
   - *Answer:* In a single 2D camera view without calibrated stereo depth, 3D leaf orientation is collapsed onto the 2D image plane. A drooping leaf facing the camera appears foreshortened but aligned with the vertical axis, while a healthy leaf growing laterally appears angled. Without depth sensor calibration, 2D droop angle reflects perspective geometry rather than physiological plant turgor.

3. **How does calcium tip burn in lettuce affect contour solidity?**
   - *Answer:* Tip burn causes death and collapse of marginal tissue along the leaf perimeter, forming deep necrotic notches and jagged indentations. While the convex hull bridges across the outer tips, the actual leaf contour dips into the dead hollows, reducing the ratio $A / A_{\text{convex}}$ (solidity).
