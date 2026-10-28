# Model Evaluation Report: Lettuce — lettuce_efficientnet_b0_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T15:44:47.400434+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.9119** | Primary optimization metric per §18.1 |
| Accuracy | 0.9453 | Overall sample accuracy |
| Macro Precision | 0.8958 | Unweighted class average |
| Macro Recall | 0.9438 | Unweighted class average |
| Evaluated Split Samples | 128 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Healthier` | 0.9831 | 0.9206 | 0.9508 | 63 |
| `NitrogenDeficiency` | 0.6000 | 0.9000 | 0.7200 | 10 |
| `PhosphorusDeficiency` | 1.0000 | 1.0000 | 1.0000 | 33 |
| `PotassiumDeficiency` | 1.0000 | 0.9545 | 0.9767 | 22 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Healthier` | `NitrogenDeficiency` | `PhosphorusDeficiency` | `PotassiumDeficiency` |
|---|---:|---:|---:|---:|
| `Healthier` | 58 | 5 | 0 | 0 |
| `NitrogenDeficiency` | 1 | 9 | 0 | 0 |
| `PhosphorusDeficiency` | 0 | 0 | 33 | 0 |
| `PotassiumDeficiency` | 0 | 1 | 0 | 21 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/features/lettuce`
- **Split Protocol:** Stratified split; train and validation isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Leaf crop instance diagnosis; crop labels inherited from whole-plant labels

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 703 |
| Validation | 128 |
| Test (Held-Out) | 0 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `EfficientNetB0`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "crop": "lettuce",
  "model_family": "efficientnet_b0",
  "model_name": null,
  "batch_size": 32,
  "epochs": 5,
  "lr": 0.001,
  "finetune_lr": 0.0001,
  "unfreeze_at_epoch": 3,
  "unfreeze_mode": "all",
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

- Model trained and evaluated on isolated lettuce leaf images using holdout validation split.
- Crop labels are inherited from whole-plant annotations; leaf-level variance may exist.
- Verified compute hardware: target=cpu, actually_used=cpu.
- Split manifest SHA256: 5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52.
- Evaluated strictly on the validation partition; test holdout remains untouched.
