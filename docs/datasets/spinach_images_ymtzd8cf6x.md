# Dataset Provenance: Spinach Image Dataset (Mendeley ymtzd8cf6x.1)

- **Document Version:** 1.0.0
- **Author:** Sushil
- **Related Tasks:** T2-003, T1-001
- **Target Crop:** Spinach (*Spinacia oleracea*)
- **Modality:** Image (RGB Leaf Photographs)
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** Spinach Dataset
- **Source URL:** https://data.mendeley.com/datasets/ymtzd8cf6x/1
- **DOI:** 10.17632/ymtzd8cf6x.1
- **Authors:** Pradnya Vishram Kulkarni, Vinaya Gohokar
- **Source Version / Date Downloaded:** Version 1 (March 2026 / Downloaded October 2026)
- **Original Filename(s):** `Spinach Dataset.zip` (original Mendeley archive)
- **Current Repository Location:** `image_data/Spinach_Images/` (partitioned across `Spinach_Healthy/` and `Spinach_Unhealthy/` subfolders)

---

## 2. License, Terms & Redistribution Status

- **License Type:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Original Terms / Rights Holder:** Pradnya Vishram Kulkarni & Vinaya Gohokar (Mendeley Data deposit)
- **Redistribution Status:** Open access under CC BY 4.0 attribution; raw dataset binaries kept local in `image_data/` and excluded from git tracking to prevent repository bloat.
- **Commercial / Non-Commercial Constraints:** Open for academic and research use with attribution.

---

## 3. Sample Counts & Volume

| Metric Stage | Sample Count (Images) | Notes / Filtering Rationale |
|---|---|---|
| **Original Raw Count** | Variable by version | Sourced from Mendeley repository archive |
| **Filtered / Reorganized Count** | 0 dropped | Retained for healthy and unhealthy visual categories |
| **Prepared Local Count (Disk)** | Sourced within `image_data/Spinach_Images/` | Combined total with `skf6w2s2h2` across 6 target classes |

---

## 4. Class Labels & Distribution

Mendeley source `ymtzd8cf6x.1` contributes to the standardized spinach leaf disease taxonomy:
- `Spinach_Healthy`: Clean, turgid foliage without necrotic lesions
- `Spinach_anthracnose`: Folate spots with dark borders (*Colletotrichum dematium*)
- `Spinach_bacterial spot`: Water-soaked to necrotic irregular spots (*Pseudomonas syringae*)
- `Spinach_downy mildew`: Yellowish chlorotic lesions and purplish/gray sporulation (*Peronospora farinosa*)
- `Spinach_pest damage`: Insect leaf chewing, perforations, and leafminer trails

---

## 5. Preprocessing & Transformation Pipeline

- **Preprocessing Script & Version:** Leaf cropping / segmentation pipeline (to be executed by SAM 2.1 / OpenCV extractors in Phase 5).
- **Cleaning & Imputation Steps:** Read-only inspection; corrupt files checked via PIL image decoding.
- **Normalization / Scaling:** Image models require standard ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Augmentation Details:** Augmentation strictly training-only per TECH_RULES §4.

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** Grouped holdout split avoiding near-duplicate leakage across splits.
- **Split Rationale:** Near-duplicate check required prior to split assignment per Overview §9.3.
- **Manifest Location:** `processed_data/manifests/spinach_image_splits.csv` (to be generated in Phase 2/3).

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

- **Synthetic Elements:** None. Physical camera captures of foliage.
- **Derived / Feature Fields:** None in raw images. Handcrafted GLCM/ExG features generated in `ml/features/spinach.py`.
- **Rule-Generated Labels:** None in image dataset; labels were pathologist/source annotated.

---

## 8. Column / Field Taxonomy Table

Raw image format; fields represent image metadata attributes:

| Field Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `image_file` | `binary` | `measured` | JPEG / PNG | Raw optical image capture |
| `width` | `integer` | `measured` | Pixels | Native image pixel width |
| `height` | `integer` | `measured` | Pixels | Native image pixel height |
| `channels` | `integer` | `measured` | 3 (RGB) | Colour channel depth |
| `label` | `string` | `measured` | Category | Human expert diagnosed pathological label |

---

## 9. Known Limitations & Unsupported Interpretations

- **Hydroponic vs. Soil Artifacts:** Part of the source images originate from field and soil cultivation; leaf symptoms may differ subtly from water-culture deficiency symptoms.
- **Lighting and Background Variations:** Backgrounds include soil, greenhouse benches, and laboratory backdrops. Leaf-level instance segmentation (SAM 2.1) is mandatory before feeding to classifiers.
- **No Direct Sensor Pairing:** These images are completely unaligned with the Meng et al. sensor dataset (Overview §19).
