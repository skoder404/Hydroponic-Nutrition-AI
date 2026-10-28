# Model Evaluation Report: Spinach — spinach_feature_random_forest_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T10:47:55.548990+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.5243** | Primary optimization metric per §18.1 |
| Accuracy | 0.5100 | Overall sample accuracy |
| Macro Precision | 0.5004 | Unweighted class average |
| Macro Recall | 0.6302 | Unweighted class average |
| Evaluated Split Samples | 2745 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Spinach_Healthy` | 0.7420 | 0.4912 | 0.5911 | 1083 |
| `Spinach_Non fresh` | 0.6479 | 1.0000 | 0.7863 | 92 |
| `Spinach_anthracnose` | 0.2156 | 0.7094 | 0.3307 | 117 |
| `Spinach_bacterial spot` | 0.4461 | 0.4715 | 0.4585 | 685 |
| `Spinach_downy mildew` | 0.4827 | 0.7887 | 0.5989 | 265 |
| `Spinach_pest damage` | 0.4680 | 0.3201 | 0.3802 | 503 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Spinach_Healthy` | `Spinach_Non fresh` | `Spinach_anthracnose` | `Spinach_bacterial spot` | `Spinach_downy mildew` | `Spinach_pest damage` |
|---|---:|---:|---:|---:|---:|---:|
| `Spinach_Healthy` | 532 | 4 | 112 | 242 | 86 | 107 |
| `Spinach_Non fresh` | 0 | 92 | 0 | 0 | 0 | 0 |
| `Spinach_anthracnose` | 6 | 0 | 83 | 19 | 3 | 6 |
| `Spinach_bacterial spot` | 92 | 30 | 106 | 323 | 75 | 59 |
| `Spinach_downy mildew` | 8 | 2 | 13 | 22 | 209 | 11 |
| `Spinach_pest damage` | 79 | 14 | 71 | 118 | 60 | 161 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/features/spinach`
- **Split Protocol:** Stratified split across 6 spinach pathological and health conditions; train/val strictly isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Phytopathological disease and health classification per spinach leaf image

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 12812 |
| Validation | 2745 |
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
| python | `3.14.0` |
| scikit-learn | `1.9.1` |
| torch | `2.14.1+cpu` |

## 7. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> The following known caveats and constraints must remain visible in documentation and UI:

- Features are extracted according to canonical spinach 6-feature schema (schema_version 1.0.0): exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, relative_leaf_area, glcm_contrast.
- Segmentation/masking was performed via Excess Green thresholding (mask_config: {'algorithm': 'excess_green_thresholding', 'exg_threshold': 5.0, 'min_intensity': 30.0}). Exactly 0 leaves failed masking and were excluded from training/evaluation.
- Model performance is evaluated exclusively on the validation holdout split (val.csv, 2745 instances); test.csv remains strictly untouched.
