# Model Evaluation Report: Spinach — spinach_custom_cnn_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T13:53:34.602179+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.6567** | Primary optimization metric per §18.1 |
| Accuracy | 0.6459 | Overall sample accuracy |
| Macro Precision | 0.6678 | Unweighted class average |
| Macro Recall | 0.6644 | Unweighted class average |
| Evaluated Split Samples | 2745 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Spinach_Healthy` | 0.7381 | 0.7599 | 0.7489 | 1083 |
| `Spinach_Non fresh` | 0.9091 | 0.9783 | 0.9424 | 92 |
| `Spinach_anthracnose` | 0.7234 | 0.5812 | 0.6445 | 117 |
| `Spinach_bacterial spot` | 0.5522 | 0.7022 | 0.6183 | 685 |
| `Spinach_downy mildew` | 0.6340 | 0.7321 | 0.6795 | 265 |
| `Spinach_pest damage` | 0.4500 | 0.2326 | 0.3067 | 503 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Spinach_Healthy` | `Spinach_Non fresh` | `Spinach_anthracnose` | `Spinach_bacterial spot` | `Spinach_downy mildew` | `Spinach_pest damage` |
|---|---:|---:|---:|---:|---:|---:|
| `Spinach_Healthy` | 823 | 1 | 12 | 132 | 25 | 90 |
| `Spinach_Non fresh` | 1 | 90 | 0 | 1 | 0 | 0 |
| `Spinach_anthracnose` | 19 | 0 | 68 | 17 | 6 | 7 |
| `Spinach_bacterial spot` | 119 | 0 | 6 | 481 | 43 | 36 |
| `Spinach_downy mildew` | 27 | 0 | 1 | 33 | 194 | 10 |
| `Spinach_pest damage` | 126 | 8 | 7 | 207 | 38 | 117 |

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

- **Model Family:** `CustomCNN`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "crop": "spinach",
  "model_family": "custom_cnn",
  "model_name": null,
  "batch_size": 64,
  "epochs": 10,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "optimizer_name": "adamw",
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.0001,
  "seed": 42,
  "num_workers": 2,
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

- Model trained and evaluated on isolated spinach leaf images using holdout validation split.
- Input type: leaf image as provided for spinach.
- Verified compute hardware: target=cuda:0, actually_used=cuda:0.
- Evaluated strictly on the validation partition; test holdout remains untouched.
