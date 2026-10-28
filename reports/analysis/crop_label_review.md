# Lettuce Crop Label Review & Weak Label Analysis (T5-012)

- **Date:** 2026-10-28
- **Reviewer / Driver:** Yashwant
- **Navigator / Concept Note:** Sushil
- **Total Source Images Evaluated:** 580
- **Total Valid Leaf Crops Generated:** 987
- **Zero-Crop Source Images:** 90
- **Random Sample Size for Review:** 120 crops (30 per class across 4 classes)
- **Concept Note Reference:** `docs/learning/weak-labels-and-instancelevel-perleaf-prediction.md`

---

## 1. Executive Summary

In Task T5-012, whole-plant lettuce images (`image_data/Lettuce_Images/`) were segmented into individual leaf crops using Meta SAM 2.1 (`sam2.1_hiera_tiny.pt`) and the quality gate from `ml.segmentation.mask_validation`.

Each leaf crop inherits the **split and the class label of its source image**. Because plant-level nutrient deficiency manifests asymmetrically across the canopy (mobile nutrients like $N$, $P$, $K$ translocate from older basal leaves to younger apical foliage), individual leaves from deficient plants frequently appear physiologically normal or healthy.

A stratified random sample of 30 leaf crops per class (120 crops total) was systematically reviewed to quantify this **weak label noise rate**.

---

## 2. Quantitative Review Results (30 Crops Per Class)

| Class | Sampled Crops | Visibly Consistent | Visibly Inconsistent (Healthy Appearance on Deficient Plant) | Inconsistency Rate (%) | Primary Physiological Explanation |
| :--- | -: | -: | -: | -: | :--- |
| **Healthier** | 30 | 29 | 1 | 3.3% | Baseline healthy foliage; 1 crop showed minor mechanical handling mark. |
| **NitrogenDeficiency** | 30 | 23 | 7 | 23.3% | Highly mobile $N$ translocated from older foliage; young crown leaves remain green and fully pigmented. |
| **PhosphorusDeficiency** | 30 | 25 | 5 | 16.7% | Emerging leaves exhibit minimal anthocyanin dullness or purple hues compared to mature fan leaves. |
| **PotassiumDeficiency** | 30 | 24 | 6 | 20.0% | Marginal necrosis and tip burn affect mature outer wrapper leaves first; inner leaves retain smooth intact margins. |
| **Total / Aggregate** | **120** | **101** | **19** | **15.8%** | **Overall Weak Label Inconsistency Rate: 15.8%** |

---

## 3. Zero-Crop Source Images Analysis

Out of 580 whole-plant images in the split manifest, **90 images** produced zero valid leaf crops passing the segmentation and quality validation gate.

- **Class breakdown of zero-crop images:**
  - `Healthier`: 42 images
  - `NitrogenDeficiency`: 24 images
  - `PhosphorusDeficiency`: 14 images
  - `PotassiumDeficiency`: 10 images
- **Root Cause Categorization:**
  1. **Very Young Seedlings / Early Rosette (<14 days):** Emerging leaves were smaller than the 1% frame area threshold (`min_area_ratio=0.01`), causing rejection by the tiny artifact filter.
  2. **Severe Overlap / Tightly Clustered Rosette:** Overlapping canopy clusters failed the component compactness or multi-component ratio (`min_component_ratio=0.85`), preventing ambiguous merged leaves from entering the dataset.
  3. **High Border Contact:** Dense plants extending beyond all 4 frame boundaries exceeded `max_border_contact_ratio=0.08` and were appropriately gated.

All 90 zero-crop images are tracked with individual image identifiers and failure reasons in `processed_data/lettuce_leaf_crops/summary.json`.

---

## 4. Key Agronomic Implications for Downstream Models

1. **Theoretical Accuracy Ceiling:**
   Due to the 15.8% weak label noise rate, an instance-level classifier evaluated on raw leaf crops will encounter valid healthy-looking leaves carrying deficiency labels. A model achieving ~80–85% accuracy on raw crops is effectively performing near Bayes optimality for individual leaf instances.
2. **Inference Aggregation Requirement:**
   At deployment time, the system must not rely on a single leaf crop. Whole-plant diagnosis requires **Multiple Instance Learning (MIL) pooling** across all segmented leaves (as documented in `docs/learning/weak-labels-and-instancelevel-perleaf-prediction.md`), prioritizing high-confidence deficiency symptoms when detected on mature leaves.
3. **Data Split Invariant:**
   All 987 leaf crops strictly inherited the split assignment of their parent image (`train`: 703, `val`: 128, `test`: 156), guaranteeing zero data leakage across partitions.

