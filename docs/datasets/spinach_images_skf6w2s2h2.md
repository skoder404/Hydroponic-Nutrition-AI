# Dataset Provenance: Local Spinach Leaf Dataset (Mendeley skf6w2s2h2.3)

- **Document Version:** 1.0.0
- **Author:** Sushil
- **Related Tasks:** T2-003, T1-001
- **Target Crop:** Local Spinach Varieties (including *Spinacia oleracea*, Water Spinach, Malabar Spinach)
- **Modality:** Image (RGB Leaf Photographs)
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** Local Spinach Leaf Dataset
- **Source URL:** https://data.mendeley.com/datasets/skf6w2s2h2/3
- **DOI:** 10.17632/skf6w2s2h2.3
- **Authors:** Md Nahid Hasan, Sonjoy Prosad Shaha, Amatul Bushra Akhi
- **Source Version / Date Downloaded:** Version 3 (November 2024 / Downloaded October 2026)
- **Original Filename(s):** `Local Spinach Leaf Dataset.zip` (Mendeley repository)
- **Current Repository Location:** `image_data/Spinach_Images/` (categorized under `Spinach_Healthy/` and `Spinach_Unhealthy/`)

---

## 2. License, Terms & Redistribution Status

- **License Type:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Original Terms / Rights Holder:** Md Nahid Hasan, Sonjoy Prosad Shaha, Amatul Bushra Akhi
- **Redistribution Status:** Publicly available with citation; raw images remain local under `image_data/` and are excluded from git commits per repository governance rules.
- **Commercial / Non-Commercial Constraints:** Open for educational and research usage with proper attribution.

---

## 3. Sample Counts & Volume

| Metric Stage | Sample Count (Images) | Notes / Filtering Rationale |
|---|---|---|
| **Original Raw Count** | Multi-thousand | Sourced from Mendeley Data deposit |
| **Filtered / Reorganized Count** | 0 dropped | Curated into local class directories |
| **Prepared Local Count (Disk)** | Sourced within `image_data/Spinach_Images/` | Contributes to the 18,302 total images stored on disk |

---

## 4. Class Labels & Distribution

Provides local varieties and non-fresh / freshness degradation symptoms:
- `Spinach_Healthy`: Fresh, green, turgid leaf foliage
- `Spinach_Non fresh`: Wilting, senescent, yellowing, or post-harvest degraded leaves
- `Spinach_anthracnose`: Distinct necrotic spots
- `Spinach_bacterial spot`: Bacterial lesions

---

## 5. Preprocessing & Transformation Pipeline

- **Preprocessing Script & Version:** Segmentation and feature extraction pipelines (SAM 2.1 and OpenCV texture/colour extractors in Phase 5).
- **Cleaning & Imputation Steps:** Verification that images decode successfully via PIL/OpenCV.
- **Normalization / Scaling:** Resized to standardized input dimensions (224x224) with ImageNet normalization for neural models.
- **Augmentation Details:** Training-only geometric and colour jitter transforms.

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** Grouped partitioning with duplicate detection to prevent data leakage.
- **Split Rationale:** Near-duplicate check required prior to split assignment per Overview §9.3.
- **Manifest Location:** `processed_data/manifests/spinach_image_splits.csv` (to be created in Phase 2/3).

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

- **Synthetic Elements:** None. Real optical smartphone/camera captures.
- **Derived / Feature Fields:** None in raw images.
- **Rule-Generated Labels:** None in image dataset; classes provided by dataset contributors.

---

## 8. Column / Field Taxonomy Table

Raw image format; fields represent image metadata attributes:

| Field Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `image_file` | `binary` | `measured` | JPEG / PNG | Raw optical image capture |
| `width` | `integer` | `measured` | Pixels | Native image pixel width |
| `height` | `integer` | `measured` | Pixels | Native image pixel height |
| `channels` | `integer` | `measured` | 3 (RGB) | Colour channel depth |
| `label` | `string` | `measured` | Category | Visual condition label (e.g. `Spinach_Non fresh`) |

---

## 9. Known Limitations & Unsupported Interpretations

- **Cultivar Diversity:** Includes local leafy greens (e.g., Malabar spinach, water spinach) which have thicker succulent leaves than true European spinach (*Spinacia oleracea*).
- **Non-Fresh vs Disease:** `Spinach_Non fresh` represents post-harvest wilting and dehydration rather than nutrient deficiency or pathogenic infection.
- **No Sensor Alignment:** Not paired with greenhouse water-chemistry sensor data (Overview §19).
