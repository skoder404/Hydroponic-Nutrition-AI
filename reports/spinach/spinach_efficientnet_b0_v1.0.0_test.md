# Model Evaluation Report: Spinach — spinach_efficientnet_b0_v1.0.0

> **Evaluated Split:** `test` | **Generated:** `2026-10-28T18:12:34.572545+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`test`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.7212** | Primary optimization metric per §18.1 |
| Accuracy | 0.6962 | Overall sample accuracy |
| Macro Precision | 0.7605 | Unweighted class average |
| Macro Recall | 0.6952 | Unweighted class average |
| Evaluated Split Samples | 2745 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Spinach_Healthy` | 0.7484 | 0.8688 | 0.8041 | 1082 |
| `Spinach_Non fresh` | 1.0000 | 0.9677 | 0.9836 | 93 |
| `Spinach_anthracnose` | 0.9091 | 0.6780 | 0.7767 | 118 |
| `Spinach_bacterial spot` | 0.5789 | 0.6111 | 0.5946 | 684 |
| `Spinach_downy mildew` | 0.7162 | 0.6000 | 0.6530 | 265 |
| `Spinach_pest damage` | 0.6104 | 0.4453 | 0.5149 | 503 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Spinach_Healthy` | `Spinach_Non fresh` | `Spinach_anthracnose` | `Spinach_bacterial spot` | `Spinach_downy mildew` | `Spinach_pest damage` |
|---|---:|---:|---:|---:|---:|---:|
| `Spinach_Healthy` | 940 | 0 | 5 | 96 | 9 | 32 |
| `Spinach_Non fresh` | 1 | 90 | 0 | 2 | 0 | 0 |
| `Spinach_anthracnose` | 12 | 0 | 80 | 13 | 6 | 7 |
| `Spinach_bacterial spot` | 163 | 0 | 2 | 418 | 27 | 74 |
| `Spinach_downy mildew` | 26 | 0 | 1 | 49 | 159 | 30 |
| `Spinach_pest damage` | 114 | 0 | 0 | 144 | 21 | 224 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/splits/image_splits_spinach.csv`
- **Split Protocol:** Stratified split; train, validation, and test partitions isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Individual leaf photography diagnosis

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 12812 |
| Validation | 2745 |
| Test (Held-Out) | 2745 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `EfficientNetB0`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "batch_size": 64,
  "initial_lr": 0.001,
  "finetune_lr": 0.0001,
  "unfreeze_at_epoch": 2,
  "unfreeze_mode": "top_blocks",
  "weight_decay": 0.0001,
  "optimizer": "adamw",
  "early_stopping_patience": 5,
  "seed": 42
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

- Final Test Evaluation: Evaluated strictly on the preserved held-out test partition after all model selection decisions were finalized.
- Strict Zero Leakage: Test partition remained unread during transfer learning fine-tuning, hyperparameter exploration, and temperature calibration.
- Inherited Plant-Level Target Labels: Ground-truth target annotations represent whole-plant or specimen pathology; individual leaf crops inherit these labels which can introduce label noise for unaffected leaves on stressed plants.
- Un-oversampled Test Evaluation: The test split contains the untouched natural class distribution without synthetic oversampling or artificial balancing.
- Leaf Photography Dataset: Evaluated on 2,745 leaf images across 6 target pathology and freshness classes.
