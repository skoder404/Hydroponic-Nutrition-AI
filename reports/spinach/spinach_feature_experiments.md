# Feature Classification Comparison Report: Spinach

> **Split Evaluated:** `validation` (strictly isolated from test holdout)
> **Generated:** `2026-10-28T10:48:44.445519+00:00`
> **Feature Schema Version:** `1.0.0`
> **Excluded Images Count:** `0`
> **Selected Champion Model:** `RandomForest` (Val Macro-F1: `0.5243`)

## 1. Model Family Comparison Table

| Model Family | Val Macro-F1 | Val Accuracy | Macro Precision | Macro Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RandomForest** | **0.5243** | 0.5100 | 0.5004 | 0.6302 | **CHAMPION** |
| **SVM** | **0.4718** | 0.4732 | 0.4488 | 0.5928 | Candidate |

## 2. Per-Class Metrics & Support (Validation Partition)

### RandomForest Per-Class Performance

| Class | Precision | Recall | F1-Score | Support (Validation Instances) |
| :--- | :---: | :---: | :---: | :---: |
| `Spinach_Healthy` | 0.7420 | 0.4912 | 0.5911 | 1083 |
| `Spinach_Non fresh` | 0.6479 | 1.0000 | 0.7863 | 92 |
| `Spinach_anthracnose` | 0.2156 | 0.7094 | 0.3307 | 117 |
| `Spinach_bacterial spot` | 0.4461 | 0.4715 | 0.4585 | 685 |
| `Spinach_downy mildew` | 0.4827 | 0.7887 | 0.5989 | 265 |
| `Spinach_pest damage` | 0.4680 | 0.3201 | 0.3802 | 503 |

### SVM Per-Class Performance

| Class | Precision | Recall | F1-Score | Support (Validation Instances) |
| :--- | :---: | :---: | :---: | :---: |
| `Spinach_Healthy` | 0.6806 | 0.4958 | 0.5737 | 1083 |
| `Spinach_Non fresh` | 0.5170 | 0.9891 | 0.6791 | 92 |
| `Spinach_anthracnose` | 0.1716 | 0.6496 | 0.2714 | 117 |
| `Spinach_bacterial spot` | 0.4678 | 0.3606 | 0.4073 | 685 |
| `Spinach_downy mildew` | 0.4433 | 0.7811 | 0.5656 | 265 |
| `Spinach_pest damage` | 0.4123 | 0.2803 | 0.3337 | 503 |

## 3. Confusion Matrices

#### RandomForest Confusion Matrix
```json
[
  [
    532,
    4,
    112,
    242,
    86,
    107
  ],
  [
    0,
    92,
    0,
    0,
    0,
    0
  ],
  [
    6,
    0,
    83,
    19,
    3,
    6
  ],
  [
    92,
    30,
    106,
    323,
    75,
    59
  ],
  [
    8,
    2,
    13,
    22,
    209,
    11
  ],
  [
    79,
    14,
    71,
    118,
    60,
    161
  ]
]
```

#### SVM Confusion Matrix
```json
[
  [
    537,
    20,
    154,
    169,
    97,
    106
  ],
  [
    1,
    91,
    0,
    0,
    0,
    0
  ],
  [
    12,
    1,
    76,
    12,
    6,
    10
  ],
  [
    139,
    35,
    110,
    247,
    78,
    76
  ],
  [
    6,
    3,
    19,
    21,
    207,
    9
  ],
  [
    94,
    26,
    84,
    79,
    79,
    141
  ]
]
```

## 4. Limitations & Segmentation Mask Note

- **Feature Schema Version:** Canonical 6-feature schema (`schema_version: 1.0.0`) comprising `exg_mean`, `exr_mean`, `yellow_pixel_ratio`, `brown_pixel_ratio`, `relative_leaf_area`, `glcm_contrast`.
- **Leaf Mask / Segmentation:** Foreground vegetation segmented using Excess Green thresholding with configuration: `{"algorithm": "excess_green_thresholding", "exg_threshold": 5.0, "min_intensity": 30.0}`.
- **Excluded Images:** Exactly 0 leaf image(s) failed segmentation masking and were excluded from dataset and model training (`excluded_images_count: 0`).
- **Validation Holdout Only:** Model selection and metrics are reported exclusively on the validation partition (`val.csv`, 2,745 instances). The test holdout partition (`test.csv`, 2,745 instances) was strictly never evaluated.
