# Dataset Provenance: Kaggle Tomato Disease Multiple Sources Dataset

- **Document Version:** 1.0.0
- **Author:** Yashwant
- **Related Tasks:** T2-004, T1-001, T3-001
- **Target Crop:** Tomato (*Solanum lycopersicum*)
- **Modality:** Image (RGB Foliage & Leaf Photographs)
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** Tomato Disease Multiple Sources Dataset
- **Source URL:** https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources
- **Authors / Contributor:** Kaggle contributor `cookiefinder` (aggregating multi-source open plant pathology repositories including PlantVillage)
- **Source Version / Date Downloaded:** `unverified` (accessed and curated during initial repository setup)
- **Original Filename(s):** Multi-source zip archive
- **Current Repository Location:** `image_data/Tomato_Images/`
  - Healthy subset: `image_data/Tomato_Images/Tomato_Healthy/`
  - Unhealthy subsets: `image_data/Tomato_Images/Tomato_Unhealthy/<Disease_Name>/`

---

## 2. License, Terms & Redistribution Status

- **License Type:** `unverified` (community aggregation; original upstream datasets largely CC0 / CC BY, but composite distribution terms are unrecorded)
- **Original Terms / Rights Holder:** `unverified`
- **Redistribution Status:** Local repository only. Image files reside locally under `image_data/` and are git-ignored.
- **Commercial / Non-Commercial Constraints:** `unverified` (restricted to non-commercial academic research and coursework decision support).

---

## 3. Sample Counts & Volume (Audited from Disk)

Counts audited directly from disk inspection of `image_data/Tomato_Images/`:

| Metric Stage | Sample Count (Images) | Notes / Filtering Rationale |
|---|:---:|---|
| **Audited Count on Disk** | **33,564 images** | Comprehensive image directory across 11 target classes |
| **Healthy Images** | 4,014 images | `image_data/Tomato_Images/Tomato_Healthy/` |
| **Unhealthy Images** | 29,550 images | 10 subdirectories under `image_data/Tomato_Images/Tomato_Unhealthy/` |
| **Total Disk Footprint** | 2,134,828,355 bytes (~1.99 GB) | Verified total on disk |

---

## 4. Class Labels & Distribution (Active Files on Disk)

| Class Label | Folder Name | Image Count | Prevalence (%) | Visual Diagnostic Definition |
|---|---|:---:|:---:|---|
| `healthy` | `Tomato_Healthy/` | 4,014 | 11.96% | Healthy leaf foliage without visible chlorosis, lesions, or distortion |
| `Tomato_Bacterial_spot` | `Tomato_Unhealthy/Tomato_Bacterial_spot/` | 3,590 | 10.70% | Water-soaked circular lesions (*Xanthomonas spp.*) |
| `Tomato_Early_blight` | `Tomato_Unhealthy/Tomato_Early_blight/` | 3,393 | 10.11% | Concentric rings, dark brown necrotic spots (*Alternaria solani*) |
| `Tomato_Late_blight` | `Tomato_Unhealthy/Tomato_Late_blight/` | 3,945 | 11.75% | Dark water-soaked lesions, rapid foliar collapse (*Phytophthora infestans*) |
| `Tomato_Leaf_Mold` | `Tomato_Unhealthy/Tomato_Leaf_Mold/` | 3,893 | 11.60% | Yellow foliar chlorosis with olive-velvety underside mold (*Passalora fulva*) |
| `Tomato_Powdery_Mildew` | `Tomato_Unhealthy/Tomato_Powdery_Mildew/` | 1,256 | 3.74% | White powdery talcum-like superficial fungal growth (*Oidium neolycopersici*) |
| `Tomato_Septoria_leaf_spot` | `Tomato_Unhealthy/Tomato_Septoria_leaf_spot/` | 3,628 | 10.81% | Circular spots with dark margins and gray centers (*Septoria lycopersici*) |
| `Tomato_Spider_mites Two-spotted_spider_mite` | `Tomato_Unhealthy/Tomato_Spider_mites Two-spotted_spider_mite/` | 2,187 | 6.52% | Chlorotic stippling, webbing, foliar bronzing (*Tetranychus urticae*) |
| `Tomato_Target_Spot` | `Tomato_Unhealthy/Tomato_Target_Spot/` | 2,287 | 6.81% | Concentric zonate target-like brown lesions (*Corynespora cassiicola*) |
| `Tomato_Tomato_mosaic_virus` | `Tomato_Unhealthy/Tomato_Tomato_mosaic_virus/` | 2,737 | 8.15% | Mottled leaf mosaic pattern, foliar rugosity (ToMV) |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Unhealthy/Tomato_Tomato_Yellow_Leaf_Curl_Virus/` | 2,634 | 7.85% | Upward leaf curling, chlorotic margins, stunting (TYLCV) |
| **Total** | | **33,564** | **100.00%** | |

---

## 5. Preprocessing & Transformation Pipeline

- **Validation:** Every file verified to load cleanly via PIL/OpenCV image decoders without truncated header errors.
- **Input Policy (Overview §14.3):** Whole-plant tomato images require leaf separation (SAM 2.1) before leaf-oriented OpenCV extraction is executed.
- **Backbone Normalization:** Standardized 224×224 resolution with ImageNet channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

---

## 6. Partitioning & Split Protocol

- **Target Splits:** 70% Train, 15% Validation, 15% Test.
- **Split Governance:** Manifest files must ensure that leaf crops derived from the same source plant or photograph never cross train/validation/test partitions (sibling crops share partition assignment per Overview §10.3).

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

- **Synthetic Elements:** None. Real optical camera captures.
- **Derived Features:** Texture and color summary statistics extracted downstream.
- **Rule-Generated Labels:** None in source imagery.

---

## 8. Column / Field Taxonomy Table (Metadata Representation)

| Column Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `image_path` | `string` | `derived` | Relative path | Local file system path |
| `disease_label` | `string` | `measured` | 11 categories | Target class from source folder name |
| `display_label` | `string` | `rule-generated` | Category | Display mapping: `healthy` -> `healthy`, disease -> `unhealthy/<class>` |
| `width` | `integer` | `measured` | pixels (`unverified`) | Raw image horizontal pixel dimension |
| `height` | `integer` | `measured` | pixels (`unverified`) | Raw image vertical pixel dimension |
| `channels` | `integer` | `measured` | 3 | Color channels (RGB) |

---

## 9. Known Limitations & Unsupported Interpretations

- **Composite Provenance:** Aggregated from multiple heterogeneous sources (including PlantVillage laboratory captures and field photographs); differences in background and lighting could introduce spurious correlation if background segmentation is not applied.
- **Unpaired with Sensor Modality:** Completely unpaired with Kolar or greenhouse sensor data. Joint multimodal training cannot be performed on this data (rule-based decision fusion only).
- **Advisory Classification:** Outputs represent advisory condition likelihoods and must never be described as definitive diagnosis.
