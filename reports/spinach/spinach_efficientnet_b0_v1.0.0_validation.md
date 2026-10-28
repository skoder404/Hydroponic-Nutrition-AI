# Model Evaluation Report: Spinach — spinach_efficientnet_b0_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T16:31:58.870384+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.7256** | Primary optimization metric per §18.1 |
| Accuracy | 0.7089 | Overall sample accuracy |
| Macro Precision | 0.7723 | Unweighted class average |
| Macro Recall | 0.6965 | Unweighted class average |
| Evaluated Split Samples | 2745 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Spinach_Healthy` | 0.7522 | 0.8633 | 0.8040 | 1083 |
| `Spinach_Non fresh` | 1.0000 | 0.9674 | 0.9834 | 92 |
| `Spinach_anthracnose` | 0.8846 | 0.5897 | 0.7077 | 117 |
| `Spinach_bacterial spot` | 0.6097 | 0.6613 | 0.6345 | 685 |
| `Spinach_downy mildew` | 0.7647 | 0.6377 | 0.6955 | 265 |
| `Spinach_pest damage` | 0.6226 | 0.4592 | 0.5286 | 503 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Spinach_Healthy` | `Spinach_Non fresh` | `Spinach_anthracnose` | `Spinach_bacterial spot` | `Spinach_downy mildew` | `Spinach_pest damage` |
|---|---:|---:|---:|---:|---:|---:|
| `Spinach_Healthy` | 935 | 0 | 6 | 95 | 14 | 33 |
| `Spinach_Non fresh` | 0 | 89 | 0 | 3 | 0 | 0 |
| `Spinach_anthracnose` | 17 | 0 | 69 | 20 | 4 | 7 |
| `Spinach_bacterial spot` | 144 | 0 | 2 | 453 | 16 | 70 |
| `Spinach_downy mildew` | 29 | 0 | 0 | 37 | 169 | 30 |
| `Spinach_pest damage` | 118 | 0 | 1 | 135 | 18 | 231 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/features/spinach`
- **Split Protocol:** Stratified split; train and validation isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Leaf images as provided with team diagnosis labels

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 12812 |
| Validation | 2745 |
| Test (Held-Out) | 0 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `EfficientNetB0`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "crop": "spinach",
  "model_family": "efficientnet_b0",
  "model_name": null,
  "batch_size": 64,
  "epochs": 3,
  "lr": 0.001,
  "finetune_lr": 0.0001,
  "unfreeze_at_epoch": 2,
  "unfreeze_mode": "top_blocks",
  "num_unfrozen_blocks": 2,
  "weights": "DEFAULT",
  "dropout_rate": 0.2,
  "fc_dim": null,
  "weight_decay": 0.0001,
  "optimizer_name": "adamw",
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.0001,
  "seed": 42,
  "num_workers": 0,
  "force_cpu": false,
  "simulate_oom": false,
  "output_dir": null,
  "report_dir": null
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

- Model trained and evaluated on isolated spinach leaf images using holdout validation split.
- Input type: leaf image as provided for spinach.
- Verified compute hardware: target=cpu, actually_used=cpu.
- Split manifest SHA256: 4c271d42de9b12de59b32ac5bb2ba4cb47821d7fc8dc9eebbc50cc41ad494f68.
- Evaluated strictly on the validation partition; test holdout remains untouched.
