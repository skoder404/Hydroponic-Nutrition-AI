# Model Evaluation Report: Spinach — spinach_sensor_decision_tree_v1.0.0

> **Evaluated Split:** `test` | **Generated:** `2026-10-27T12:22:58.540765+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`test`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **1.0000** | Primary optimization metric per §18.1 |
| Accuracy | 1.0000 | Overall sample accuracy |
| Macro Precision | 1.0000 | Unweighted class average |
| Macro Recall | 1.0000 | Unweighted class average |
| Evaluated Split Samples | 280 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 1.0000 | 1.0000 | 1.0000 | 40 |
| `EC_Low_Deficiency` | 1.0000 | 1.0000 | 1.0000 | 40 |
| `Humidity_High_Mildew_Risk` | 1.0000 | 1.0000 | 1.0000 | 40 |
| `Humidity_Low_Stress` | 1.0000 | 1.0000 | 1.0000 | 40 |
| `Normal` | 1.0000 | 1.0000 | 1.0000 | 40 |
| `Water_Temp_High` | 1.0000 | 1.0000 | 1.0000 | 40 |
| `pH_Imbalance` | 1.0000 | 1.0000 | 1.0000 | 40 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `Normal` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| `EC_Low_Deficiency` | 0 | 40 | 0 | 0 | 0 | 0 | 0 |
| `Humidity_High_Mildew_Risk` | 0 | 0 | 40 | 0 | 0 | 0 | 0 |
| `Humidity_Low_Stress` | 0 | 0 | 0 | 40 | 0 | 0 | 0 |
| `Normal` | 0 | 0 | 0 | 0 | 40 | 0 | 0 |
| `Water_Temp_High` | 0 | 0 | 0 | 0 | 0 | 40 | 0 |
| `pH_Imbalance` | 0 | 0 | 0 | 0 | 0 | 0 | 40 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/sensor/spinach/manifest.json`
- **Split Protocol:** Preserved 6-hour time-block holdout (test.csv untouched until final evaluation)
- **Preprocessing Version:** `1.0.0`
- **Label-Generation Method:** Rule-derived thresholds mapped from physical hydroponic solution bands

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 958 |
| Validation | 162 |
| Test (Held-Out) | 280 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `DecisionTreeClassifier`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "criterion": "gini",
  "max_depth": 8,
  "min_samples_leaf": 1,
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
- Mixed Real and Synthetic Features: Air temperature is real greenhouse telemetry; relative humidity is 85.7% real and 14.3% adjusted; water temperature, pH, and EC are synthetic distributions.
