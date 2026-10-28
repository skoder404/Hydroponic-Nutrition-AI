# Model Evaluation Report: Tomato — tomato_efficientnet_b0_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T17:11:43.105257+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.8748** | Primary optimization metric per §18.1 |
| Accuracy | 0.8715 | Overall sample accuracy |
| Macro Precision | 0.8758 | Unweighted class average |
| Macro Recall | 0.8754 | Unweighted class average |
| Evaluated Split Samples | 5034 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 0.8891 | 0.8494 | 0.8688 | 538 |
| `Tomato_Early_blight` | 0.8323 | 0.7996 | 0.8156 | 509 |
| `Tomato_Healthy` | 0.8851 | 0.9468 | 0.9149 | 602 |
| `Tomato_Late_blight` | 0.8489 | 0.8159 | 0.8320 | 592 |
| `Tomato_Leaf_Mold` | 0.8102 | 0.8699 | 0.8390 | 584 |
| `Tomato_Powdery_Mildew` | 0.8030 | 0.8670 | 0.8338 | 188 |
| `Tomato_Septoria_leaf_spot` | 0.8093 | 0.8033 | 0.8063 | 544 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0.9035 | 0.9421 | 0.9224 | 328 |
| `Tomato_Target_Spot` | 0.9369 | 0.8222 | 0.8758 | 343 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 0.9621 | 0.9646 | 0.9633 | 395 |
| `Tomato_Tomato_mosaic_virus` | 0.9535 | 0.9489 | 0.9512 | 411 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Tomato_Bacterial_spot` | `Tomato_Early_blight` | `Tomato_Healthy` | `Tomato_Late_blight` | `Tomato_Leaf_Mold` | `Tomato_Powdery_Mildew` | `Tomato_Septoria_leaf_spot` | `Tomato_Spider_mites Two-spotted_spider_mite` | `Tomato_Target_Spot` | `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Tomato_mosaic_virus` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 457 | 15 | 4 | 9 | 22 | 3 | 26 | 0 | 1 | 0 | 1 |
| `Tomato_Early_blight` | 2 | 407 | 12 | 21 | 24 | 1 | 33 | 3 | 5 | 0 | 1 |
| `Tomato_Healthy` | 1 | 3 | 570 | 5 | 11 | 2 | 3 | 1 | 2 | 0 | 4 |
| `Tomato_Late_blight` | 11 | 18 | 17 | 483 | 24 | 20 | 11 | 1 | 3 | 0 | 4 |
| `Tomato_Leaf_Mold` | 8 | 14 | 8 | 22 | 508 | 3 | 16 | 0 | 1 | 3 | 1 |
| `Tomato_Powdery_Mildew` | 2 | 1 | 8 | 7 | 0 | 163 | 7 | 0 | 0 | 0 | 0 |
| `Tomato_Septoria_leaf_spot` | 19 | 24 | 7 | 15 | 23 | 11 | 437 | 2 | 3 | 1 | 2 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0 | 1 | 2 | 0 | 3 | 0 | 2 | 309 | 4 | 5 | 2 |
| `Tomato_Target_Spot` | 6 | 6 | 13 | 2 | 1 | 0 | 3 | 25 | 282 | 4 | 1 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 8 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 381 | 3 |
| `Tomato_Tomato_mosaic_virus` | 0 | 0 | 2 | 5 | 9 | 0 | 2 | 1 | 0 | 2 | 390 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/features/tomato`
- **Split Protocol:** Stratified split; train and validation isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Leaf images as provided with team diagnosis labels

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 23496 |
| Validation | 5034 |
| Test (Held-Out) | 0 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `EfficientNetB0`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "crop": "tomato",
  "model_family": "efficientnet_b0",
  "model_name": null,
  "batch_size": 64,
  "epochs": 2,
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

- Model trained and evaluated on isolated tomato leaf images using holdout validation split.
- Input type: leaf image as provided for tomato.
- Verified compute hardware: target=cpu, actually_used=cpu.
- Split manifest SHA256: 8ecad33c13ca2c78d1a93afe874121e84dd020fb28aa042ffca50bcaa3d43d6e.
- Evaluated strictly on the validation partition; test holdout remains untouched.
