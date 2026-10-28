# Model Evaluation Report: Tomato — tomato_feature_random_forest_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T08:58:51.701653+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.4346** | Primary optimization metric per §18.1 |
| Accuracy | 0.4519 | Overall sample accuracy |
| Macro Precision | 0.4519 | Unweighted class average |
| Macro Recall | 0.4811 | Unweighted class average |
| Evaluated Split Samples | 5034 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 0.4294 | 0.5762 | 0.4921 | 538 |
| `Tomato_Early_blight` | 0.3316 | 0.3752 | 0.3521 | 509 |
| `Tomato_Healthy` | 0.7504 | 0.7392 | 0.7448 | 602 |
| `Tomato_Late_blight` | 0.5212 | 0.2280 | 0.3173 | 592 |
| `Tomato_Leaf_Mold` | 0.3605 | 0.3253 | 0.3420 | 584 |
| `Tomato_Powdery_Mildew` | 0.2803 | 0.7021 | 0.4006 | 188 |
| `Tomato_Septoria_leaf_spot` | 0.3600 | 0.0331 | 0.0606 | 544 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0.5648 | 0.5579 | 0.5613 | 328 |
| `Tomato_Target_Spot` | 0.3375 | 0.6356 | 0.4408 | 343 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 0.4873 | 0.4380 | 0.4613 | 395 |
| `Tomato_Tomato_mosaic_virus` | 0.5479 | 0.6813 | 0.6074 | 411 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Tomato_Bacterial_spot` | `Tomato_Early_blight` | `Tomato_Healthy` | `Tomato_Late_blight` | `Tomato_Leaf_Mold` | `Tomato_Powdery_Mildew` | `Tomato_Septoria_leaf_spot` | `Tomato_Spider_mites Two-spotted_spider_mite` | `Tomato_Target_Spot` | `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Tomato_mosaic_virus` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 310 | 105 | 14 | 18 | 5 | 59 | 2 | 0 | 0 | 25 | 0 |
| `Tomato_Early_blight` | 73 | 191 | 14 | 15 | 77 | 25 | 2 | 10 | 46 | 19 | 37 |
| `Tomato_Healthy` | 16 | 11 | 445 | 15 | 24 | 11 | 1 | 7 | 36 | 4 | 32 |
| `Tomato_Late_blight` | 66 | 77 | 15 | 135 | 75 | 102 | 13 | 12 | 39 | 30 | 28 |
| `Tomato_Leaf_Mold` | 126 | 68 | 14 | 33 | 190 | 20 | 3 | 12 | 37 | 43 | 38 |
| `Tomato_Powdery_Mildew` | 14 | 7 | 14 | 3 | 2 | 132 | 0 | 0 | 0 | 16 | 0 |
| `Tomato_Septoria_leaf_spot` | 75 | 43 | 47 | 22 | 69 | 71 | 18 | 14 | 111 | 14 | 60 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0 | 7 | 1 | 1 | 10 | 1 | 2 | 183 | 98 | 14 | 11 |
| `Tomato_Target_Spot` | 0 | 11 | 8 | 1 | 22 | 0 | 0 | 53 | 218 | 12 | 18 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 41 | 42 | 4 | 6 | 30 | 39 | 2 | 20 | 31 | 173 | 7 |
| `Tomato_Tomato_mosaic_virus` | 1 | 14 | 17 | 10 | 23 | 11 | 7 | 13 | 30 | 5 | 280 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/features/tomato`
- **Split Protocol:** Stratified split across 11 disease and health conditions (PlantVillage tomato corpus); train/val strictly isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Pathological/phytopathological expert visual diagnosis per leaf image

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 23495 |
| Validation | 5034 |
| Test (Held-Out) | 0 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `RandomForest`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "n_estimators": 150,
  "max_depth": 8,
  "min_samples_split": 3,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "random_state": 42
}
```

## 6. Software & Runtime Versions

| Package / Component | Version |
|---|---|
| numpy | `2.5.3` |
| platform | `Windows-11-10.0.26300-SP0` |
| python | `3.13.5` |
| scikit-learn | `1.9.1` |
| torch | `2.7.1+cu118` |

## 7. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> The following known caveats and constraints must remain visible in documentation and UI:

- Features are extracted according to ADR 0007 (schema_version 1.0.0) with droop_angle formally removed, retaining 4 defensible texture and shape features (glcm_contrast, glcm_entropy, bounding_aspect_ratio, solidity).
- Out of dataset manifests, exactly 1 unreadable source image (HL_(336)) was excluded during feature extraction.
- Model performance is evaluated exclusively on the validation holdout split (val.csv, 5,034 instances); test.csv remains strictly untouched.
