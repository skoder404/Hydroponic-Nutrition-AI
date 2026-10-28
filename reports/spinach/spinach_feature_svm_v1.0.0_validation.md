# Model Evaluation Report: Spinach — spinach_feature_svm_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T10:48:44.443232+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.4718** | Primary optimization metric per §18.1 |
| Accuracy | 0.4732 | Overall sample accuracy |
| Macro Precision | 0.4488 | Unweighted class average |
| Macro Recall | 0.5928 | Unweighted class average |
| Evaluated Split Samples | 2745 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Spinach_Healthy` | 0.6806 | 0.4958 | 0.5737 | 1083 |
| `Spinach_Non fresh` | 0.5170 | 0.9891 | 0.6791 | 92 |
| `Spinach_anthracnose` | 0.1716 | 0.6496 | 0.2714 | 117 |
| `Spinach_bacterial spot` | 0.4678 | 0.3606 | 0.4073 | 685 |
| `Spinach_downy mildew` | 0.4433 | 0.7811 | 0.5656 | 265 |
| `Spinach_pest damage` | 0.4123 | 0.2803 | 0.3337 | 503 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Spinach_Healthy` | `Spinach_Non fresh` | `Spinach_anthracnose` | `Spinach_bacterial spot` | `Spinach_downy mildew` | `Spinach_pest damage` |
|---|---:|---:|---:|---:|---:|---:|
| `Spinach_Healthy` | 537 | 20 | 154 | 169 | 97 | 106 |
| `Spinach_Non fresh` | 1 | 91 | 0 | 0 | 0 | 0 |
| `Spinach_anthracnose` | 12 | 1 | 76 | 12 | 6 | 10 |
| `Spinach_bacterial spot` | 139 | 35 | 110 | 247 | 78 | 76 |
| `Spinach_downy mildew` | 6 | 3 | 19 | 21 | 207 | 9 |
| `Spinach_pest damage` | 94 | 26 | 84 | 79 | 79 | 141 |

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
| python | `3.14.0` |
| scikit-learn | `1.9.1` |
| torch | `2.14.1+cpu` |

## 7. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> The following known caveats and constraints must remain visible in documentation and UI:

- Features are extracted according to canonical spinach 6-feature schema (schema_version 1.0.0): exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, relative_leaf_area, glcm_contrast.
- Segmentation/masking was performed via Excess Green thresholding (mask_config: {'algorithm': 'excess_green_thresholding', 'exg_threshold': 5.0, 'min_intensity': 30.0}). Exactly 0 leaves failed masking and were excluded from training/evaluation.
- Model performance is evaluated exclusively on the validation holdout split (val.csv, 2745 instances); test.csv remains strictly untouched.
