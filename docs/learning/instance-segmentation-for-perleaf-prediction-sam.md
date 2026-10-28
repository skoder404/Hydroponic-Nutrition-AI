# Concept: Instance Segmentation for Per-Leaf Prediction (SAM 2.1)

- **Concept File:** `docs/learning/instance-segmentation-for-perleaf-prediction-sam.md`
- **Related Task:** T5-010
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-28
- **Status:** Completed

---

## 1. Intuition

In controlled-environment hydroponics, cameras capture overhead or oblique canopy photographs containing multiple overlapping leaves, background troughs, nutrient solution pipes, and peat plugs. Whole-plant image classification directly on raw canopy photos suffers from severe background confounding and spatial dilution:
1. **Background Noise:** Wet PVC pipes, algae buildup, and reflective water surfaces distract deep vision models.
2. **Deficiency Heterogeneity:** Nutrient deficiencies manifest unevenly across plant ontogeny. For instance, mobile nutrient deficiencies (Nitrogen, Magnesium, Potassium) show symptoms on older, lower leaves as nutrients are translocated, while immobile deficiencies (Calcium, Iron) first strike young, expanding inner leaves.
3. **Occlusion & Clutter:** A global canopy image blends necrotic spots on one leaf with healthy tissue on another, diluting localized deficiency signatures.

**Segment Anything Model 2.1 (SAM 2.1)** acts as a foundation instance segmenter that isolates individual leaf instances into clean, background-subtracted ROIs. Once isolated, each leaf instance can be independently analyzed for color metrics (ExG, ExR, chlorosis ratios), texture (GLCM), and morphological shape before aggregating plant-level diagnoses.

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 Mask Intersection over Union (IoU)

For two candidate binary masks $M_A, M_B \in \{0, 1\}^{H \times W}$, spatial overlap is quantified via the Jaccard index:

$$
\text{IoU}(M_A, M_B) = \frac{|M_A \cap M_B|}{|M_A \cup M_B|} = \frac{\sum_{r, c} (M_A[r, c] \land M_B[r, c])}{\sum_{r, c} (M_A[r, c] \lor M_B[r, c])}
$$

### 2.2 Deduplication Decision Rule

When multiple prompt points trigger redundant segmentations of the same physical leaf:
1. Candidate masks are ordered by predictor confidence score descending:
   $$
   S_1 \ge S_2 \ge \dots \ge S_K
   $$
2. For each candidate $M_k$, if:
   $$
   \max_{j \in \text{Accepted}} \text{IoU}(M_k, M_j) \ge \tau_{\text{overlap}} \quad (\text{default } \tau_{\text{overlap}} = 0.50)
   $$
   - **Suppression (Default):** Candidate $k$ is marked as a duplicate, assigned `MaskVerdict.REJECT`, and moved to `rejected[]` with reason citing primary leaf $j$.
   - **Merging Option (`merge_duplicates=True`):** $M_j \leftarrow M_j \lor M_k$, re-computing bounding boxes and topological metrics.

### 2.3 Quality Gate Constraints

Each mask must satisfy the geometric criteria from ADR 0003:
- **Area Ratio:** $\tau_{\text{min\_area}} \le \frac{|M|}{H \times W} \le \tau_{\text{max\_area}}$ ($0.01 \le \text{ratio} \le 0.85$).
- **Border Contact:** $\frac{\sum_{\text{border}} M[r, c]}{|M|} \le \tau_{\text{border}}$ ($\le 0.08$).
- **Topological Coherence:** $\frac{|M_{\text{largest\_cc}}|}{|M|} \ge \tau_{\text{component}}$ ($\ge 0.85$).

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

```python
import numpy as np


def compute_mask_iou_from_scratch(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute exact IoU between two boolean masks."""
    intersection = np.count_nonzero(mask_a & mask_b)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union)


def deduplicate_masks_from_scratch(
    masks: list[np.ndarray],
    scores: list[float],
    iou_threshold: float = 0.50,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Sort candidates by score and filter duplicates above IoU threshold.

    Returns:
        (accepted_indices, rejected_info)
    """
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    accepted = []
    rejected = []

    for idx in sorted_indices:
        cand_mask = masks[idx]
        is_dup = False
        for acc_idx in accepted:
            iou = compute_mask_iou_from_scratch(cand_mask, masks[acc_idx])
            if iou >= iou_threshold:
                is_dup = True
                rejected.append((idx, acc_idx, iou))
                break
        if not is_dup:
            accepted.append(idx)

    return accepted, rejected


if __name__ == "__main__":
    # Two heavily overlapping synthetic masks (IoU = 0.80)
    m1 = np.zeros((20, 20), dtype=bool)
    m1[2:12, 2:12] = True  # 10x10 = 100 px

    m2 = np.zeros((20, 20), dtype=bool)
    m2[2:12, 4:14] = True  # 10x10 = 100 px, overlap = 80 px, union = 120 px -> IoU = 0.667

    acc, rej = deduplicate_masks_from_scratch([m1, m2], [0.92, 0.78], iou_threshold=0.50)
    assert acc == [0], f"Expected mask 0 accepted, got {acc}"
    assert rej[0][0] == 1 and rej[0][1] == 0
    print(f"Scratch deduplication passed! Rejected mask 1 with IoU={rej[0][2]:.3f}")
```

---

## 4. Production & Library Usage

In production (`ml/segmentation/leaf_extraction.py`):

```python
from ml.segmentation.leaf_extraction import LeafExtractionConfig, extract_leaves

# Unified call for training and inference
result = extract_leaves(
    image="image_data/Lettuce_Images/Healthier/Healthier_0001.jpg",
    config=LeafExtractionConfig(
        single_leaf_mode=False,
        max_leaves=10,
        iou_overlap_threshold=0.50,
        merge_duplicates=False,
    ),
)

print(f"Extracted {len(result.leaves)} valid leaves:")
for leaf in result.leaves:
    print(f" - {leaf.leaf_id}: bbox={leaf.bounding_box}, conf={leaf.confidence:.3f}")

print(f"Rejected {len(result.rejected)} candidate masks with explicit reasons:")
for rej in result.rejected:
    print(f" - {rej.candidate_id}: {rej.reason}")
```

**Key Parameters & Trade-offs:**
- `single_leaf_mode`: When True, bypasses vegetation peak proposal generation. Essential for datasets containing pre-sliced leaf photos.
- `iou_overlap_threshold = 0.50`: Balances over-segmentation against false suppression. Lower values (e.g. 0.30) risk discarding distinct overlapping leaves; higher values (e.g. 0.70) risk admitting duplicate prompts of the same leaf.
- `max_leaves = 10`: Caps downstream feature extraction and inference latency to ensure deterministic latency bounds.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Extreme Over-segmentation on Dense Canopies:**
   - *Symptom:* SAM 2.1 produces 30+ tiny candidate masks along vein margins or overlapping lobes.
   - *Mitigation:* `min_area_ratio=0.01` in `validate_mask` instantly catches and rejects sub-1% leaf fragments.
2. **Flooding the Entire Frame:**
   - *Symptom:* Background drainage tray or bench gets grouped with canopy.
   - *Mitigation:* `max_area_ratio=0.85` marks whole-image floods as `MaskVerdict.REJECT`.
3. **Severe Edge Truncation:**
   - *Symptom:* Leaf partially out-of-frame has misleadingly low leaf area and high border contact.
   - *Mitigation:* `border_contact_ratio > 0.08` rejects truncated border instances.
4. **CUDA OOM on High-Resolution Multi-Leaf Images:**
   - *Mitigation:* Integrated device manager OOM handler in `SAM2Segmenter` catches CUDA OOM, frees memory via `torch.cuda.empty_cache()`, and seamlessly triggers CPU fallback.

---

## 6. Where It Appears in This Project

- **Pipeline Module:** [`ml/segmentation/leaf_extraction.py`](file:///d:/Hydroponic-Nutrition-AI/ml/segmentation/leaf_extraction.py)
- **Foundation Wrapper:** [`ml/segmentation/sam2_segmenter.py`](file:///d:/Hydroponic-Nutrition-AI/ml/segmentation/sam2_segmenter.py)
- **Quality Gate:** [`ml/segmentation/mask_validation.py`](file:///d:/Hydroponic-Nutrition-AI/ml/segmentation/mask_validation.py)
- **Training Crop Generation:** Slicing lettuce plant images into training datasets.
- **Backend API Endpoints:** Image inference endpoint for per-leaf anomaly scoring.

---

## 7. Self-Check Questions (3–5 Questions)

1. **Q1: Why is per-leaf instance segmentation preferred over whole-canopy bounding boxes in hydroponics?**
   *Answer:* Hydroponic nutrient deficiency symptoms are spatially and temporally localized across leaves (e.g., mobile deficiencies on older outer leaves, immobile deficiencies on inner heart leaves). Whole-canopy boxes blend healthy and diseased tissue and include distracting background hardware.
2. **Q2: Why do we generate candidate prompts from ExG vegetation centroids instead of a dense uniform grid?**
   *Answer:* Unsupervised dense grid generation (as evaluated in ADR 0003) takes 5–25 seconds per frame and frequently merges touching leaves. ExG centroid proposals target active plant tissue directly, reducing prompt count to 3–8 coordinates and executing in <200 ms.
3. **Q3: What happens to candidate masks that fail validation or are identified as duplicates?**
   *Answer:* They are collected into the structured `rejected[]` list with their typed verdict and a human-readable reason string. They are never silently discarded.
4. **Q4: How are background pixels handled in the extracted leaf crops?**
   *Answer:* For each accepted leaf, the RGB crop zeroes background pixels outside the mask to black `(0, 0, 0)`, and provides a corresponding 4-channel RGBA crop with alpha=0 on background pixels to prevent background leakage into color/texture feature extractors.
