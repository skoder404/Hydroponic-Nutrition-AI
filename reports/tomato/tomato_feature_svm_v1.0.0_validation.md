# Model Evaluation Report: Tomato — tomato_feature_svm_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T09:00:14.620468+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.4271** | Primary optimization metric per §18.1 |
| Accuracy | 0.4384 | Overall sample accuracy |
| Macro Precision | 0.4431 | Unweighted class average |
| Macro Recall | 0.4666 | Unweighted class average |
| Evaluated Split Samples | 5034 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 0.4331 | 0.5595 | 0.4882 | 538 |
| `Tomato_Early_blight` | 0.2938 | 0.3792 | 0.3310 | 509 |
| `Tomato_Healthy` | 0.7566 | 0.7126 | 0.7340 | 602 |
| `Tomato_Late_blight` | 0.4921 | 0.1571 | 0.2382 | 592 |
| `Tomato_Leaf_Mold` | 0.3300 | 0.2808 | 0.3034 | 584 |
| `Tomato_Powdery_Mildew` | 0.2222 | 0.6170 | 0.3268 | 188 |
| `Tomato_Septoria_leaf_spot` | 0.3540 | 0.1048 | 0.1617 | 544 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0.5482 | 0.6067 | 0.5760 | 328 |
| `Tomato_Target_Spot` | 0.3798 | 0.6356 | 0.4755 | 343 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 0.4893 | 0.4051 | 0.4432 | 395 |
| `Tomato_Tomato_mosaic_virus` | 0.5747 | 0.6740 | 0.6204 | 411 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Tomato_Bacterial_spot` | `Tomato_Early_blight` | `Tomato_Healthy` | `Tomato_Late_blight` | `Tomato_Leaf_Mold` | `Tomato_Powdery_Mildew` | `Tomato_Septoria_leaf_spot` | `Tomato_Spider_mites Two-spotted_spider_mite` | `Tomato_Target_Spot` | `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Tomato_mosaic_virus` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 301 | 105 | 11 | 14 | 18 | 63 | 3 | 0 | 0 | 22 | 1 |
| `Tomato_Early_blight` | 76 | 193 | 14 | 6 | 71 | 30 | 17 | 12 | 45 | 17 | 28 |
| `Tomato_Healthy` | 27 | 17 | 429 | 5 | 26 | 11 | 14 | 6 | 30 | 7 | 30 |
| `Tomato_Late_blight` | 56 | 110 | 27 | 93 | 69 | 115 | 18 | 12 | 30 | 32 | 30 |
| `Tomato_Leaf_Mold` | 115 | 85 | 10 | 40 | 164 | 28 | 19 | 14 | 37 | 40 | 32 |
| `Tomato_Powdery_Mildew` | 17 | 14 | 15 | 2 | 4 | 116 | 1 | 0 | 0 | 19 | 0 |
| `Tomato_Septoria_leaf_spot` | 66 | 52 | 32 | 20 | 70 | 81 | 57 | 18 | 87 | 10 | 51 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0 | 5 | 3 | 0 | 9 | 1 | 6 | 199 | 82 | 13 | 10 |
| `Tomato_Target_Spot` | 2 | 12 | 7 | 0 | 21 | 0 | 5 | 56 | 218 | 5 | 17 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 33 | 53 | 6 | 2 | 21 | 67 | 2 | 29 | 16 | 160 | 6 |
| `Tomato_Tomato_mosaic_virus` | 2 | 11 | 13 | 7 | 24 | 10 | 19 | 17 | 29 | 2 | 277 |

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

- **Model Family:** `SVM`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "C": 2.0,
  "gamma": "scale",
  "kernel": "rbf",
  "class_weight": "balanced",
  "probability": true,
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
