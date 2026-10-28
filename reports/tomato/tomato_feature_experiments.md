# Feature Classification Comparison Report: Tomato

> **Split Evaluated:** `validation` (strictly isolated from test holdout)  
> **Generated:** `2026-10-28T09:00:14.623696+00:00`  
> **Feature Schema Version:** `1.0.0`  
> **Excluded Images Count:** `1`  
> **Selected Champion Model:** `RandomForest` (Val Macro-F1: `0.4346`)  

## 1. Model Family Comparison Table

| Model Family | Val Macro-F1 | Val Accuracy | Macro Precision | Macro Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RandomForest** | **0.4346** | 0.4519 | 0.4519 | 0.4811 | **CHAMPION** |
| **SVM** | **0.4271** | 0.4384 | 0.4431 | 0.4666 | Candidate |

## 2. Per-Class Metrics & Support (Validation Partition)

### RandomForest Per-Class Performance

| Class | Precision | Recall | F1-Score | Support (Validation Instances) |
| :--- | :---: | :---: | :---: | :---: |
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

### SVM Per-Class Performance

| Class | Precision | Recall | F1-Score | Support (Validation Instances) |
| :--- | :---: | :---: | :---: | :---: |
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

## 3. Confusion Matrices

#### RandomForest Confusion Matrix
```json
[
  [
    310,
    105,
    14,
    18,
    5,
    59,
    2,
    0,
    0,
    25,
    0
  ],
  [
    73,
    191,
    14,
    15,
    77,
    25,
    2,
    10,
    46,
    19,
    37
  ],
  [
    16,
    11,
    445,
    15,
    24,
    11,
    1,
    7,
    36,
    4,
    32
  ],
  [
    66,
    77,
    15,
    135,
    75,
    102,
    13,
    12,
    39,
    30,
    28
  ],
  [
    126,
    68,
    14,
    33,
    190,
    20,
    3,
    12,
    37,
    43,
    38
  ],
  [
    14,
    7,
    14,
    3,
    2,
    132,
    0,
    0,
    0,
    16,
    0
  ],
  [
    75,
    43,
    47,
    22,
    69,
    71,
    18,
    14,
    111,
    14,
    60
  ],
  [
    0,
    7,
    1,
    1,
    10,
    1,
    2,
    183,
    98,
    14,
    11
  ],
  [
    0,
    11,
    8,
    1,
    22,
    0,
    0,
    53,
    218,
    12,
    18
  ],
  [
    41,
    42,
    4,
    6,
    30,
    39,
    2,
    20,
    31,
    173,
    7
  ],
  [
    1,
    14,
    17,
    10,
    23,
    11,
    7,
    13,
    30,
    5,
    280
  ]
]
```

#### SVM Confusion Matrix
```json
[
  [
    301,
    105,
    11,
    14,
    18,
    63,
    3,
    0,
    0,
    22,
    1
  ],
  [
    76,
    193,
    14,
    6,
    71,
    30,
    17,
    12,
    45,
    17,
    28
  ],
  [
    27,
    17,
    429,
    5,
    26,
    11,
    14,
    6,
    30,
    7,
    30
  ],
  [
    56,
    110,
    27,
    93,
    69,
    115,
    18,
    12,
    30,
    32,
    30
  ],
  [
    115,
    85,
    10,
    40,
    164,
    28,
    19,
    14,
    37,
    40,
    32
  ],
  [
    17,
    14,
    15,
    2,
    4,
    116,
    1,
    0,
    0,
    19,
    0
  ],
  [
    66,
    52,
    32,
    20,
    70,
    81,
    57,
    18,
    87,
    10,
    51
  ],
  [
    0,
    5,
    3,
    0,
    9,
    1,
    6,
    199,
    82,
    13,
    10
  ],
  [
    2,
    12,
    7,
    0,
    21,
    0,
    5,
    56,
    218,
    5,
    17
  ],
  [
    33,
    53,
    6,
    2,
    21,
    67,
    2,
    29,
    16,
    160,
    6
  ],
  [
    2,
    11,
    13,
    7,
    24,
    10,
    19,
    17,
    29,
    2,
    277
  ]
]
```

## 4. Limitations & ADR 0007 Schema Compliance

- **Feature Schema Version:** Conforms to ADR 0007 (`schema_version: 1.0.0`) with 4 features (`glcm_contrast`, `glcm_entropy`, `bounding_aspect_ratio`, `solidity`). The candidate `droop_angle` was formally removed due to lack of gravity/stem reference in detached leaf imagery.
- **Excluded Images:** Exactly 1 corrupt/unreadable source image (HL_(336)) was excluded during feature extraction (`failed_images_count: 1`).
- **Validation Holdout Only:** Model selection and metrics are reported exclusively on the validation partition (`val.csv`, 5,034 instances). The test holdout partition (`test.csv`, 5,034 instances) was strictly never evaluated.