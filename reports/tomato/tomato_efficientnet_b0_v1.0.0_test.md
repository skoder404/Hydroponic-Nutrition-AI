# Model Evaluation Report: Tomato — tomato_efficientnet_b0_v1.0.0

> **Evaluated Split:** `test` | **Generated:** `2026-10-28T18:15:31.507824+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`test`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.8696** | Primary optimization metric per §18.1 |
| Accuracy | 0.8681 | Overall sample accuracy |
| Macro Precision | 0.8715 | Unweighted class average |
| Macro Recall | 0.8715 | Unweighted class average |
| Evaluated Split Samples | 5034 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 0.9066 | 0.7922 | 0.8455 | 539 |
| `Tomato_Early_blight` | 0.8035 | 0.8114 | 0.8074 | 509 |
| `Tomato_Healthy` | 0.8788 | 0.9635 | 0.9192 | 602 |
| `Tomato_Late_blight` | 0.8730 | 0.8376 | 0.8549 | 591 |
| `Tomato_Leaf_Mold` | 0.7936 | 0.8887 | 0.8384 | 584 |
| `Tomato_Powdery_Mildew` | 0.7342 | 0.8624 | 0.7932 | 189 |
| `Tomato_Septoria_leaf_spot` | 0.8104 | 0.7702 | 0.7898 | 544 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0.9094 | 0.9482 | 0.9284 | 328 |
| `Tomato_Target_Spot` | 0.9522 | 0.8134 | 0.8774 | 343 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 0.9673 | 0.9722 | 0.9697 | 395 |
| `Tomato_Tomato_mosaic_virus` | 0.9572 | 0.9268 | 0.9418 | 410 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Tomato_Bacterial_spot` | `Tomato_Early_blight` | `Tomato_Healthy` | `Tomato_Late_blight` | `Tomato_Leaf_Mold` | `Tomato_Powdery_Mildew` | `Tomato_Septoria_leaf_spot` | `Tomato_Spider_mites Two-spotted_spider_mite` | `Tomato_Target_Spot` | `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Tomato_mosaic_virus` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 427 | 18 | 3 | 5 | 31 | 14 | 40 | 0 | 1 | 0 | 0 |
| `Tomato_Early_blight` | 5 | 413 | 8 | 17 | 25 | 0 | 29 | 7 | 5 | 0 | 0 |
| `Tomato_Healthy` | 0 | 1 | 580 | 2 | 10 | 4 | 0 | 1 | 2 | 0 | 2 |
| `Tomato_Late_blight` | 11 | 28 | 8 | 495 | 24 | 16 | 5 | 1 | 1 | 0 | 2 |
| `Tomato_Leaf_Mold` | 2 | 17 | 19 | 13 | 519 | 4 | 8 | 1 | 0 | 0 | 1 |
| `Tomato_Powdery_Mildew` | 5 | 1 | 3 | 6 | 3 | 163 | 8 | 0 | 0 | 0 | 0 |
| `Tomato_Septoria_leaf_spot` | 14 | 27 | 6 | 21 | 28 | 18 | 419 | 2 | 1 | 2 | 6 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 1 | 1 | 3 | 0 | 1 | 0 | 0 | 311 | 4 | 6 | 1 |
| `Tomato_Target_Spot` | 3 | 7 | 21 | 0 | 6 | 1 | 5 | 19 | 279 | 0 | 2 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 3 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 384 | 3 |
| `Tomato_Tomato_mosaic_virus` | 0 | 0 | 8 | 7 | 6 | 1 | 3 | 0 | 0 | 5 | 380 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/splits/image_splits_tomato.csv`
- **Split Protocol:** Stratified split; train, validation, and test partitions isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Individual leaf photography diagnosis

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 23496 |
| Validation | 5034 |
| Test (Held-Out) | 5034 |

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
- Extensive Multi-Class Foliar Diagnosis: Evaluated on 5,034 leaf images across 11 target pathology classes.
