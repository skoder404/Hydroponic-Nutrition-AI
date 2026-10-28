# Model Evaluation Report: Lettuce — lettuce_efficientnet_b0_v1.0.0

> **Evaluated Split:** `test` | **Generated:** `2026-10-28T18:10:51.068525+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`test`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.8356** | Primary optimization metric per §18.1 |
| Accuracy | 0.8974 | Overall sample accuracy |
| Macro Precision | 0.8608 | Unweighted class average |
| Macro Recall | 0.8382 | Unweighted class average |
| Evaluated Split Samples | 156 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Healthier` | 1.0000 | 0.9875 | 0.9937 | 80 |
| `NitrogenDeficiency` | 0.8889 | 0.6154 | 0.7273 | 13 |
| `PhosphorusDeficiency` | 0.6970 | 1.0000 | 0.8214 | 23 |
| `PotassiumDeficiency` | 0.8571 | 0.7500 | 0.8000 | 40 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Healthier` | `NitrogenDeficiency` | `PhosphorusDeficiency` | `PotassiumDeficiency` |
|---|---:|---:|---:|---:|
| `Healthier` | 79 | 1 | 0 | 0 |
| `NitrogenDeficiency` | 0 | 8 | 0 | 5 |
| `PhosphorusDeficiency` | 0 | 0 | 23 | 0 |
| `PotassiumDeficiency` | 0 | 0 | 10 | 30 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/splits/image_splits_lettuce.csv`
- **Split Protocol:** Stratified split; train, validation, and test partitions isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Leaf crop instance diagnosis; crop labels inherited from whole-plant labels

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 703 |
| Validation | 128 |
| Test (Held-Out) | 156 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `EfficientNetB0`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "batch_size": 32,
  "initial_lr": 0.001,
  "finetune_lr": 0.0001,
  "unfreeze_at_epoch": 3,
  "unfreeze_mode": "all",
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
- Leaf Crops Instance Diagnosis: Evaluated on 156 segmented individual leaf crops isolated from whole-plant canopy photographs.
