# Dataset Provenance: Tomato Leaf Disease Image Dataset (Mendeley zfv4jj7855.1)

- **Document Version:** 1.0.0
- **Author:** Yashwant
- **Related Tasks:** T2-004, T1-001, T3-001
- **Target Crop:** Tomato (*Solanum lycopersicum*)
- **Modality:** Image (RGB Foliage & Leaf Photographs)
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** Mendeley Tomato Leaf Disease Dataset
- **Source URL:** https://data.mendeley.com/datasets/zfv4jj7855/1
- **DOI:** 10.17632/zfv4jj7855.1
- **Authors / Depositors:** Mendeley Data research contributors
- **Source Version / Date Downloaded:** Version 1 (published 2020) / Ingested October 2026
- **Original Filename(s):** Multi-class image archive deposit
- **Current Repository Location:** `image_data/Tomato_Images/` (curated alongside the Kaggle multiple-sources aggregation)

---

## 2. License, Terms & Redistribution Status

- **License Type:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original Terms / Rights Holder:** Original Mendeley dataset depositors
- **Redistribution Status:** Publicly accessible with attribution; raw image binaries reside locally under `image_data/Tomato_Images/` and are excluded from git commits per repository governance rules.
- **Commercial / Non-Commercial Constraints:** `unverified` (assumed educational/research use per CC BY 4.0 terms).

---

## 3. Sample Counts & Volume (Audited from Disk)

Counts audited directly from disk inspection of `image_data/Tomato_Images/`:

| Metric Stage | Sample Count (Images) | Notes / Filtering Rationale |
|---|:---:|---|
| **Raw Repository Count** | 33,564 images | Comprehensive local image pool across all tomato disease classes |
| **Filtered / Corrupted** | 0 images | All files verified to decode cleanly as RGB images |
| **Prepared Count on Disk** | **33,564 images** | Subdivided into `Tomato_Healthy/` (4,014) and 10 disease directories under `Tomato_Unhealthy/` (29,550) |
| **Total Storage Size on Disk** | 2,134,828,355 bytes (~1.99 GB) | Verified binary volume on disk |

---

## 4. Class Labels & Distribution (Files on Disk)

| Class Label | Folder Name | Image Count | Prevalence (%) | Visual Diagnostic Definition |
|---|---|:---:|:---:|---|
| `Tomato_Healthy` | `Tomato_Healthy/` | 4,014 | 11.96% | Healthy green foliage, regular leaf morphology, no lesions or mottling |
| `Tomato_Bacterial_spot` | `Tomato_Unhealthy/Tomato_Bacterial_spot/` | 3,590 | 10.70% | Small, water-soaked circular lesions turning brown/black with yellow halos (*Xanthomonas*) |
| `Tomato_Early_blight` | `Tomato_Unhealthy/Tomato_Early_blight/` | 3,393 | 10.11% | Concentric target-board rings, brown necrotic lesions surrounded by chlorotic tissue (*Alternaria solani*) |
| `Tomato_Late_blight` | `Tomato_Unhealthy/Tomato_Late_blight/` | 3,945 | 11.75% | Large, irregular pale-green water-soaked lesions turning dark brown/purplish (*Phytophthora infestans*) |
| `Tomato_Leaf_Mold` | `Tomato_Unhealthy/Tomato_Leaf_Mold/` | 3,893 | 11.60% | Pale greenish-yellow spots on upper leaf surface, olive-green velvety mold underneath (*Passalora fulva*) |
| `Tomato_Powdery_Mildew` | `Tomato_Unhealthy/Tomato_Powdery_Mildew/` | 1,256 | 3.74% | White powdery talcum-like fungal patches on leaf lamina (*Oidium neolycopersici*) |
| `Tomato_Septoria_leaf_spot` | `Tomato_Unhealthy/Tomato_Septoria_leaf_spot/` | 3,628 | 10.81% | Small circular spots with dark brown margins and sunken grayish-white centers (*Septoria lycopersici*) |
| `Tomato_Spider_mites_Two-spotted_spider_mite` | `Tomato_Unhealthy/Tomato_Spider_mites Two-spotted_spider_mite/` | 2,187 | 6.52% | Fine stippling, chlorotic flecking, visible webbing and foliar bronzing (*Tetranychus urticae*) |
| `Tomato_Target_Spot` | `Tomato_Unhealthy/Tomato_Target_Spot/` | 2,287 | 6.81% | Brown circular lesions with prominent concentric rings (*Corynespora cassiicola*) |
| `Tomato_Tomato_mosaic_virus` | `Tomato_Unhealthy/Tomato_Tomato_mosaic_virus/` | 2,737 | 8.15% | Mottled dark/light green mosaic patterning, leaf distortion, blister-like foliar rugosity (ToMV) |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Unhealthy/Tomato_Tomato_Yellow_Leaf_Curl_Virus/` | 2,634 | 7.85% | Upward curling of leaf margins, interveinal chlorosis, severe plant stunting (TYLCV) |
| **Total** | | **33,564** | **100.00%** | |

---

## 5. Preprocessing & Transformation Pipeline

- **Format Standardization:** Decoded and standardized to 3-channel RGB image tensors.
- **Input Policy (Overview §14.3):** Whole-plant tomato photographs require semantic segmentation and leaf separation (via SAM 2.1) before leaf-level feature extraction or classification.
- **Model Standard Dimensions:** Standardized resizing to 224×224 pixels with ImageNet standard normalization for deep feature backbones (EfficientNet-B0).
- **Classical Features:** Extracted GLCM Haralick texture metrics (`contrast`, `homogeneity`, `energy`, `correlation`), color channel statistics (mean, variance, skewness across RGB, HSV, LAB).

---

## 6. Partitioning & Split Protocol

- **Target Split Allocation:** 70% Train, 15% Validation, 15% Test.
- **Leakage Prevention:** Split manifests must partition at the original image capture level; leaf crops derived from the same photograph must inherit the parent image's partition assignment.
- **Deduplication:** Near-duplicate check enforced before final split manifest creation.

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

- **Synthetic Elements:** None. Real photographic leaf captures.
- **Derived Features:** Extracted visual feature descriptors (color histograms, GLCM texture matrices).
- **Rule-Generated Labels:** None. Diagnostic labels were assigned by research collectors.

---

## 8. Column / Field Taxonomy Table (Metadata Representation)

| Column Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `image_path` | `string` | `derived` | File path | Relative file location under `image_data/Tomato_Images/` |
| `disease_label` | `string` | `measured` | 11 categories | Expert-assigned diagnostic disease class or healthy |
| `display_label` | `string` | `rule-generated` | String | User-facing display mapping: `healthy` or `unhealthy/<class>` |
| `image_width` | `integer` | `measured` | pixels (`unverified`) | Raw image horizontal dimension |
| `image_height` | `integer` | `measured` | pixels (`unverified`) | Raw image vertical dimension |
| `color_channels` | `integer` | `measured` | 3 | Standardized RGB color channels |

---

## 9. Known Limitations & Unsupported Interpretations

- **Laboratory vs. Field Conditions:** Photographs include controlled-background plant disease captures that may exhibit visual domain shifts when presented with field/polyhouse campus camera lighting.
- **Unpaired with Sensors:** Sourced completely independently from the Kolar precision agriculture dataset. There is no biological or temporal correspondence with `sensor_data/Tomato_Sensor-Readings/`. Unpaired cross-modal training is forbidden.
- **Symptom Mimicry:** Early-stage viral and fungal symptoms can visually overlap; automated outputs must be reported as decision-support condition probabilities, never definitive pathology diagnosis.
