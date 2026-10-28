# Model Evaluation Report: Lettuce — lettuce_custom_cnn_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T11:14:31.303076+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.7045** | Primary optimization metric per §18.1 |
| Accuracy | 0.8047 | Overall sample accuracy |
| Macro Precision | 0.7514 | Unweighted class average |
| Macro Recall | 0.6792 | Unweighted class average |
| Evaluated Split Samples | 128 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Healthier` | 0.8592 | 0.9683 | 0.9104 | 63 |
| `NitrogenDeficiency` | 0.6667 | 0.4000 | 0.5000 | 10 |
| `PhosphorusDeficiency` | 0.7576 | 0.7576 | 0.7576 | 33 |
| `PotassiumDeficiency` | 0.7222 | 0.5909 | 0.6500 | 22 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Healthier` | `NitrogenDeficiency` | `PhosphorusDeficiency` | `PotassiumDeficiency` |
|---|---:|---:|---:|---:|
| `Healthier` | 61 | 0 | 1 | 1 |
| `NitrogenDeficiency` | 4 | 4 | 1 | 1 |
| `PhosphorusDeficiency` | 4 | 1 | 25 | 3 |
| `PotassiumDeficiency` | 2 | 1 | 6 | 13 |

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

- **Model Family:** `CustomCNN`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "crop": "lettuce",
  "model_family": "custom_cnn",
  "model_name": null,
  "batch_size": 32,
  "epochs": 25,
  "lr": 0.001,
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
| python | `3.13.5` |
| scikit-learn | `1.9.1` |
| torch | `2.7.1+cu118` |

## 7. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> The following known caveats and constraints must remain visible in documentation and UI:

- Model trained and evaluated on isolated lettuce leaf images using holdout validation split.
- Crop labels are inherited from whole-plant annotations; leaf-level variance may exist.
- Verified compute hardware: target=cuda:0, actually_used=cuda:0.
- Evaluated strictly on the validation partition; test holdout remains untouched.
