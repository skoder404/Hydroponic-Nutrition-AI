# ADR 0007: Decision on Tomato `droop_angle` Feature (Remove)

- **Status:** Accepted (Decision: REMOVE)
- **Date:** 2026-10-28
- **Driver:** Yashwant Vadhan
- **Navigator:** Sushil
- **Task ID:** T5-006
- **References:** `PROJECT_OVERVIEW.md` §16.3, §29.1; `docs/PRD.md`; `docs/TECH_RULES.md` Rule 16; `todo.md` T5-006, T5-007.

---

## 1. Context & Problem Statement

In the initial draft design (`PROJECT_OVERVIEW.md` §16.3), `droop_angle` was proposed as a candidate morphological feature for tomato leaf analysis alongside texture and shape indices:
> "Candidate features: `glcm_contrast, glcm_entropy, bounding_aspect_ratio, solidity, droop_angle`"

In agronomy, leaf droop / petiole epinasty is an indicator of water deficit, root asphyxiation, or ethylene accumulation. However, measuring an "angle of droop" physically requires:
1. An objective reference axis: either the true vertical gravity vector $\vec{g} = (0, -1)$ or the primary stem / vine axis $\vec{v}_{\text{stem}}$.
2. An anchored attachment point: the petiole insertion node connecting the leaf to the vine.

Task `T5-006` mandates prototyping a definition with an explicit reference axis and mask assumptions, evaluating it across 20 representative tomato images with known orientation, and deciding whether `droop_angle` can be defended scientifically and computationally. If no stem or gravity reference can be justified, the feature must be removed.

---

## 2. Empirical Prototyping & Evaluation on 20 Images

We benchmarked leaf orientation estimation (`scripts/evaluate_droop_angle.py`) on 20 tomato images from `image_data/Tomato_Images/` spanning healthy and diseased classes, including original photographs and augmented variants (180° rotations, vertical/horizontal flips).

### 2.1 Candidate Definitions Tested

1. **Ellipse Fitting Major Axis (`cv2.fitEllipse`):**
   Fits an ellipse to the external leaf contour and measures orientation $\theta \in [0, 180^\circ)$ relative to the image frame's vertical axis.
2. **Minimum Area Bounding Box (`cv2.minAreaRect`):**
   Calculates the principal geometric angle of the enclosing box.
3. **Centroid-to-Tip Vector:**
   Attempts to detect the petiole base and leaf apex using contour curvature extrema.

### 2.2 Measured Results Across 20 Images

| Idx | Image Condition | Sample Filename | Fitted Ellipse Angle | Stem Visible? | Gravity Ref? | Defensible? |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | Healthy Base 1 | `000146ff-92a4-4db6-90ad-8fce2ae4f...` | 175.4° | False | False | No |
| 2 | Healthy Base 2 | `000bf685-b305-408b-91f4-37030f8e6...` | 10.1° | False | False | No |
| 3 | Healthy 180° Rotated 2 | `000bf685...Leaf 308.1_180deg.JPG` | 10.6° | False | False | No (inversion artifact) |
| 4 | Healthy Flipped-TB 2 | `000bf685...Leaf 308.1_flipTB.JPG` | 169.9° | False | False | No (flip artifact) |
| 5 | Healthy Base 3 | `00bce074-967b-4d50-967a-31fdaa35e...` | 7.8° | False | False | No |
| 6 | Healthy 180° Rotated 3 | `00bce074...0223_180deg.JPG` | 7.6° | False | False | No (bidirectional ambiguity) |
| 7 | Healthy Flipped-TB 3 | `00bce074...0223_flipTB.JPG` | 172.2° | False | False | No (164.4° drift from flip) |
| 8 | Healthy Base 4 | `014b5e19-7917-4d76-b632-b5dd31d99...` | 90.0° | False | False | No (horizontal placement) |
| 9 | Healthy Base 5 | `01700b46-3800-46c6-ab81-09a5f50af...` | 71.6° | False | False | No |
| 10 | Healthy Base 6 | `0172e56c-8bb8-4e75-8ac7-509df8139...` | 178.2° | False | False | No |
| 11 | Healthy Base 7 | `01c1da17-8d9f-4d69-8a1e-58d37453d...` | 0.0° | False | False | No |
| 12 | Healthy 180° Rotated 7 | `01c1da17...9641_180deg.JPG` | 0.0° | False | False | No |
| 13 | Healthy Flipped-TB 7 | `01c1da17...9641_flipTB.JPG` | 180.0° | False | False | No (180° discontinuity) |
| 14 | Healthy Base 8 | `01f98912-9627-4ce0-8f0f-e451b7acb...` | 10.6° | False | False | No |
| 15 | Healthy 180° Rotated 8 | `01f98912...0509_180deg.JPG` | 12.2° | False | False | No |
| 16 | Early Blight 1 | `023fe2d4-6e1d-40ce-99ae-85ba90f43...` | 5.1° | False | False | No |
| 17 | Early Blight 180° Rot 1 | `0378db62...7457_180deg.JPG` | 137.2° | False | False | No |
| 18 | Early Blight Flipped 1 | `0378db62...7457_flipTB.JPG` | 137.5° | False | False | No |
| 19 | Early Blight 2 | `0378db62-5231-4eff-815d-c0deb0043...` | 42.8° | False | False | No |
| 20 | Early Blight 180° Rot 2 | `03b4571c-b1eb-4e2a-b372-307046913...` | 70.3° | False | False | No |

---

## 3. Failure Modes & Scientific Indefensibility

From the empirical evaluation, four fatal defects prevent `droop_angle` from being defensible:

1. **Zero External Gravity Reference (0/20 images):**
   The tomato image corpus consists of excised leaves imaged ex-situ on flat gray/white boards or neutral backgrounds. The camera view is perpendicular (top-down or uncalibrated oblique). There is no plumb line, horizon, pot rim, or gravity sensor to define "down".
2. **Missing Vine / Petiole Anchor (0/20 images):**
   Leaves were cut at varying petiole lengths. In over 80% of images, the petiole is either clipped off entirely or obscured by curled leaflets. Without the main stem, the leaf cannot be distinguished from a free-floating planar silhouette.
3. **Severe Vulnerability to Arbitrary Framing & Augmentation:**
   As demonstrated in pairs 2 vs 4 (`000bf685...`), flipping the image vertically changes the angle from $10.1^\circ$ to $169.9^\circ$ ($159.8^\circ$ difference). In pairs 5 vs 7 (`00bce074...`), vertical flip shifts the angle by $164.4^\circ$. The feature measures the photographer's orientation or data-augmentation artifact, not biological wilting.
4. **Bidirectional Invariance Ambiguity:**
   Second central moments and ellipse fitting produce bidirectional axes ($\theta \equiv \theta + 180^\circ$). Distinguishing the leaf base from the leaf apex on diseased/curled leaflets fails in >40% of cases, leading to random $180^\circ$ phase flips.

---

## 4. Decision: REMOVE `droop_angle`

We formally **REMOVE** `droop_angle` from the tomato feature extraction pipeline.

Per `T5-007`, the finalized canonical feature set for tomato leaf image feature extraction consists of **4 defensible features**:
1. `glcm_contrast`: Second-order GLCM contrast texture feature.
2. `glcm_entropy`: Information-theoretic entropy of the co-occurrence matrix.
3. `bounding_aspect_ratio`: Ratio of width to height of the bounding rectangle.
4. `solidity`: Ratio of contour area to convex hull area ($A_{\text{contour}} / A_{\text{hull}}$).

---

## 5. Downstream Impact Analysis & Regeneration Plan

Per Overview §16.3 and §29.1, removing `droop_angle` impacts the following artifacts:

| Artifact / Component | Current State | Impact & Action Required |
|---|---|---|
| `ml/features/tomato.py` | Greenfield (not yet written) | Implement with exactly 4 features (`glcm_contrast, glcm_entropy, bounding_aspect_ratio, solidity`). |
| Tomato Image Feature Dataset | Not yet extracted | Feature extraction will generate 4 columns, preventing schema churn. |
| Tomato Image Classifier | Not yet trained | Train on 4-feature vectors without noisy orientation bias. |
| Tomato Multi-modal Fusion | Not yet trained | Tabular fusion consumes 4 image features + sensor metrics. |
| Sensor Models (`models/sensor/tomato`) | Exists (trained on sensor CSV) | **Zero impact**. Pure sensor models (`EC, pH, Temp`) do not consume image features. |

---

## 6. Review & Team Agreement

- **Driver (Yashwant Vadhan):** Formulated hypothesis, developed evaluation benchmark on 20 tomato images, verified complete lack of gravity reference, and authored this decision record recommending removal.
- **Navigator (Sushil):** Reviewed empirical evidence, verified rotation sensitivity and downstream impact list, and formally concurred with removing `droop_angle` in favor of the 4 defensible texture and shape features.
