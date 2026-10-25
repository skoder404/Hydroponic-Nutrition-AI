# Dataset Provenance: Kaggle Lettuce NPK Dataset

- **Document Version:** 1.0.0
- **Author:** Yashwant
- **Related Tasks:** T2-002, T1-001, T3-001
- **Target Crop:** Lettuce (*Lactuca sativa* L.)
- **Modality:** Image (RGB Foliage & Leaf Photographs)
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** Kaggle Lettuce NPK Dataset
- **Source URL:** https://www.kaggle.com/datasets/ (exact Kaggle dataset slug: `unverified`)
- **Source Version / Date Downloaded:** `unverified` (ingested during repository initial setup)
- **Original Filename(s):** Upstream archive containing class subdirectories: `-K` (potassium deficiency), `-N` (nitrogen deficiency), `-P` (phosphorus deficiency), and `FN` (full nutrition / healthier)
- **Current Repository Location:** `image_data/Lettuce_Images/`
  - Subdirectories: `Healthier/`, `NitrogenDeficiency/`, `PhosphorusDeficiency/`, `PotassiumDeficiency/`
  - Audited manifest: `image_data/Lettuce_Images/manifest.csv` (943 candidate records)
  - Class summary: `image_data/Lettuce_Images/class_counts.csv`

---

## 2. License, Terms & Redistribution Status

- **License Type:** `unverified` (standard Kaggle community upload; explicit license terms unrecorded in project overview)
- **Original Terms / Rights Holder:** `unverified`
- **Redistribution Status:** Local repository only. Raw image files reside locally under `image_data/` and are excluded from git commits per repository governance rules. Only audited metadata manifests are committed.
- **Commercial / Non-Commercial Constraints:** `unverified` (restricted to non-commercial academic research and coursework decision support).

---

## 3. Sample Counts & Volume (Audited from Disk)

Counts audited directly from disk inspection of `image_data/Lettuce_Images/` and `manifest.csv`:

| Metric Stage | Sample Count | Notes / Filtering Rationale |
|---|---|---|
| **Raw Candidate Count (Manifest)** | 943 files | Total files evaluated in the raw Kaggle source archive |
| **Skipped: Augmented Duplicates** | 266 files | Upstream pre-augmented copies (rotated/flipped) dropped to prevent data leakage across splits |
| **Skipped: Orphans Not Selected** | 90 files | Secondary unverified candidate images excluded during initial curation |
| **Skipped: Exact Duplicates** | 7 files | SHA-256 identical duplicate files removed |
| **Prepared Count on Disk** | **580 images** | Unique, verified RGB `.jpg` files organized in class directories |
| **Total Storage Size on Disk** | 91,528,839 bytes (~87.29 MB) | Sum of 580 image files and 2 tracking CSVs |

---

## 4. Class Labels & Distribution (Active Files on Disk)

Audited directly from files residing on disk in `image_data/Lettuce_Images/`:

| Class Label | Folder Name | Image Count | Prevalence (%) | Visual Diagnostic Definition |
|---|---|:---:|:---:|---|
| `Healthier` | `Healthier/` | 389 | 67.07% | Uniform green leaf pigmentation, normal turgor, no chlorosis or necrosis |
| `PotassiumDeficiency` | `PotassiumDeficiency/` | 68 | 11.72% | Marginal and tip chlorosis progressing to necrosis/scorching on older leaves |
| `PhosphorusDeficiency` | `PhosphorusDeficiency/` | 66 | 11.38% | Dark green foliage with purplish/bronze discoloration, stunted leaf expansion |
| `NitrogenDeficiency` | `NitrogenDeficiency/` | 57 | 9.83% | General chlorosis, pale yellowish-green foliage starting from older leaves |
| **Total** | | **580** | **100.00%** | |

---

## 5. Preprocessing & Transformation Pipeline

- **Image Standardisation:** Converted from raw formats (PNG/JPEG) to uniform RGB `.jpg`.
- **Resolution Range:**
  - Width: 183 px minimum, 2,048 px maximum
  - Height: 168 px minimum, 1,552 px maximum
- **Deduplication:** SHA-256 hash computed for every candidate image to eliminate exact duplicates (7 removed) and prevent leakage.
- **Downstream Pipeline (Phase 5):**
  - Leaf separation via Segment Anything Model (SAM 2.1) on leaf-level inputs.
  - Classical feature extraction: color histograms, HSV/LAB color statistics, and GLCM Haralick texture features (`graycomatrix`, `graycoprops`).
  - Standardized resizing to 224×224 with ImageNet normalization for deep transfer learning (EfficientNet-B0).

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** Stratified split across classes with strict holdout.
- **Split Rationale:** No duplicate or near-duplicate images permitted across train and test partitions.
- **Manifest Location:** `image_data/Lettuce_Images/manifest.csv` (contains raw-to-processed mappings, dimensions, and SHA-256 checksums).
- **Partitioning Target (Phase 2):**
  - Train: 70%
  - Validation: 15%
  - Test: 15% (strictly held out until final evaluation)

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

- **Synthetic Elements:** None. All 580 images are real photographic captures. Upstream synthetic augmentations (266 files) were identified and discarded.
- **Derived Features:** Image width, height, and SHA-256 hashes recorded in `manifest.csv`.
- **Rule-Generated Labels:** None in this dataset; labels were pre-assigned by the original Kaggle contributor.

---

## 8. Column / Field Taxonomy Table (`manifest.csv`)

| Column Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `new_filename` | `string` | `derived` | File basename | Standardized target filename (e.g. `PotassiumDeficiency_0001.jpg`) |
| `class` | `string` | `measured` | 4 categories | Categorical deficiency or healthy label |
| `original_filename` | `string` | `measured` | String | Source filename in raw upstream archive (e.g. `k_1.png`) |
| `original_folder` | `string` | `measured` | `-K`, `-N`, `-P`, `FN` | Original directory in archive |
| `was_augmented` | `boolean` | `derived` | `True` / `False` | Flag identifying if source image was an upstream augmented copy |
| `width` | `integer` | `measured` | 183 – 2048 px | Original image pixel width |
| `height` | `integer` | `measured` | 168 – 1552 px | Original image pixel height |
| `mode_before_conversion` | `string` | `measured` | `RGB`, `RGBA`, `L` | Original PIL image color mode before standardization |
| `sha256` | `string` | `derived` | 64-char hex | Cryptographic SHA-256 checksum of the image binary |
| `status` | `string` | `rule-generated` | Category | Processing status: `copied`, `skipped_augmented_duplicate`, `skipped_orphan_not_selected`, `skipped_exact_duplicate` |

---

## 9. Known Limitations & Unsupported Interpretations

- **Unpaired Modality:** This image dataset was collected independently from the HydroGrowNet sensor experiments. It has **no temporal, spatial, or plant-level pairing** with `sensor_data/Lettuce_Sensor-Readings/`. Cross-modal pairing models are explicitly forbidden (fusion is rule-based and decision-level only per TECH_RULES §15).
- **Class Imbalance:** Healthy samples (`Healthier`: 389) outnumber deficiencies (~57–68 per class) by ~6:1. Models must use class weighting or balanced sampling during training.
- **Macro-Nutrient Specificity:** Visual deficiency symptoms for N, P, and K can overlap in early stages; visual classification is decision-support only, never a definitive chemical tissue diagnosis.
- **Crop Transfer:** Models trained on this dataset apply to lettuce only and must not be used on spinach or tomato without retraining.
