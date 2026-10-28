# Model Evaluation Report: Tomato — tomato_custom_cnn_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T14:29:45.767291+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.8714** | Primary optimization metric per §18.1 |
| Accuracy | 0.8723 | Overall sample accuracy |
| Macro Precision | 0.8752 | Unweighted class average |
| Macro Recall | 0.8693 | Unweighted class average |
| Evaluated Split Samples | 5034 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 0.8743 | 0.8401 | 0.8569 | 538 |
| `Tomato_Early_blight` | 0.8609 | 0.7780 | 0.8173 | 509 |
| `Tomato_Healthy` | 0.8843 | 0.9518 | 0.9168 | 602 |
| `Tomato_Late_blight` | 0.7648 | 0.8733 | 0.8155 | 592 |
| `Tomato_Leaf_Mold` | 0.8945 | 0.8425 | 0.8677 | 584 |
| `Tomato_Powdery_Mildew` | 0.7238 | 0.6968 | 0.7100 | 188 |
| `Tomato_Septoria_leaf_spot` | 0.7956 | 0.7941 | 0.7948 | 544 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0.9244 | 0.9695 | 0.9464 | 328 |
| `Tomato_Target_Spot` | 0.9633 | 0.9184 | 0.9403 | 343 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 0.9639 | 0.9468 | 0.9553 | 395 |
| `Tomato_Tomato_mosaic_virus` | 0.9775 | 0.9513 | 0.9642 | 411 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Tomato_Bacterial_spot` | `Tomato_Early_blight` | `Tomato_Healthy` | `Tomato_Late_blight` | `Tomato_Leaf_Mold` | `Tomato_Powdery_Mildew` | `Tomato_Septoria_leaf_spot` | `Tomato_Spider_mites Two-spotted_spider_mite` | `Tomato_Target_Spot` | `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | `Tomato_Tomato_mosaic_virus` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Tomato_Bacterial_spot` | 452 | 11 | 9 | 25 | 10 | 11 | 18 | 0 | 1 | 1 | 0 |
| `Tomato_Early_blight` | 13 | 396 | 15 | 33 | 6 | 2 | 32 | 1 | 5 | 6 | 0 |
| `Tomato_Healthy` | 1 | 0 | 573 | 10 | 4 | 3 | 9 | 0 | 1 | 1 | 0 |
| `Tomato_Late_blight` | 15 | 14 | 13 | 517 | 10 | 15 | 4 | 2 | 0 | 2 | 0 |
| `Tomato_Leaf_Mold` | 13 | 12 | 5 | 22 | 492 | 3 | 34 | 1 | 0 | 2 | 0 |
| `Tomato_Powdery_Mildew` | 6 | 3 | 12 | 24 | 5 | 131 | 7 | 0 | 0 | 0 | 0 |
| `Tomato_Septoria_leaf_spot` | 6 | 20 | 12 | 38 | 18 | 15 | 432 | 2 | 1 | 0 | 0 |
| `Tomato_Spider_mites Two-spotted_spider_mite` | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 318 | 4 | 0 | 4 |
| `Tomato_Target_Spot` | 1 | 3 | 3 | 2 | 0 | 1 | 0 | 13 | 315 | 0 | 5 |
| `Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 10 | 0 | 2 | 1 | 0 | 0 | 2 | 6 | 0 | 374 | 0 |
| `Tomato_Tomato_mosaic_virus` | 0 | 1 | 3 | 3 | 5 | 0 | 5 | 1 | 0 | 2 | 391 |

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

- **Model Family:** `CustomCNN`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "crop": "tomato",
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

- Model trained and evaluated on isolated tomato leaf images using holdout validation split.
- Input type: leaf image as provided for tomato.
- Verified compute hardware: target=cuda:0, actually_used=cuda:0.
- Evaluated strictly on the validation partition; test holdout remains untouched.
