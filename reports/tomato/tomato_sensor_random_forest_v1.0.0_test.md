# Model Evaluation Report: Tomato — tomato_sensor_random_forest_v1.0.0

> **Evaluated Split:** `test` | **Generated:** `2026-10-27T12:20:20.489814+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`test`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.9685** | Primary optimization metric per §18.1 |
| Accuracy | 0.9689 | Overall sample accuracy |
| Macro Precision | 0.9692 | Unweighted class average |
| Macro Recall | 0.9689 | Unweighted class average |
| Evaluated Split Samples | 8830 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 0.9586 | 0.9955 | 0.9767 | 883 |
| `EC_Low_Deficiency` | 0.9675 | 0.9434 | 0.9553 | 883 |
| `Humidity_High_Mildew_Risk` | 0.9585 | 0.9943 | 0.9761 | 883 |
| `Humidity_Low_Stress` | 0.9670 | 0.9955 | 0.9810 | 883 |
| `K_Deficiency` | 0.9551 | 0.9887 | 0.9716 | 883 |
| `N_Deficiency` | 0.9590 | 0.9547 | 0.9569 | 883 |
| `Normal` | 1.0000 | 1.0000 | 1.0000 | 883 |
| `P_Deficiency` | 0.9771 | 0.9683 | 0.9727 | 883 |
| `Water_Temp_High` | 0.9612 | 0.9807 | 0.9709 | 883 |
| `pH_Imbalance` | 0.9884 | 0.8675 | 0.9240 | 883 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `K_Deficiency` | `N_Deficiency` | `Normal` | `P_Deficiency` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 879 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| `EC_Low_Deficiency` | 0 | 833 | 12 | 4 | 13 | 12 | 0 | 4 | 4 | 1 |
| `Humidity_High_Mildew_Risk` | 1 | 0 | 878 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| `Humidity_Low_Stress` | 2 | 0 | 0 | 879 | 0 | 1 | 0 | 0 | 0 | 1 |
| `K_Deficiency` | 1 | 2 | 3 | 2 | 873 | 2 | 0 | 0 | 0 | 0 |
| `N_Deficiency` | 7 | 1 | 8 | 8 | 7 | 843 | 0 | 5 | 4 | 0 |
| `Normal` | 0 | 0 | 0 | 0 | 0 | 0 | 883 | 0 | 0 | 0 |
| `P_Deficiency` | 10 | 1 | 2 | 3 | 5 | 2 | 0 | 855 | 5 | 0 |
| `Water_Temp_High` | 1 | 3 | 3 | 2 | 1 | 3 | 0 | 0 | 866 | 4 |
| `pH_Imbalance` | 16 | 21 | 10 | 11 | 13 | 16 | 0 | 11 | 19 | 766 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/sensor/tomato/manifest.json`
- **Split Protocol:** Preserved calendar month holdout (test.csv untouched until final evaluation)
- **Preprocessing Version:** `1.0.0`
- **Label-Generation Method:** Rule-derived thresholds mapped from physical hydroponic solution bands

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 29943 |
| Validation | 5377 |
| Test (Held-Out) | 8830 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `RandomForestClassifier`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "n_estimators": 100,
  "max_depth": null,
  "min_samples_leaf": 1,
  "criterion": "gini",
  "class_weight": null,
  "random_state": 42
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

- Final Test Evaluation: Performance evaluated strictly on the preserved held-out test partition after all model selection decisions were finalized.
- Rule-Derived Diagnostic Targets: Sensor labels are mathematically mapped from published physical action bands. Models learn the decision threshold geometry rather than independent biological plant health validated by agronomists.
- Un-oversampled Test Evaluation: The test split contains the untouched natural class distribution without synthetic oversampling.
- Strict Zero Leakage: Test partition remained unread during feature exploration, hyperparameter tuning, and candidate model selection.
- Experimental Status: Prepared from simulated Kolar precision agriculture dataset with synthetic water temperature proxy shifts; not validated for live campus hardware.
