# Model Evaluation Report: Lettuce — lettuce_sensor_decision_tree_v1.0.0

> **Evaluated Split:** `test` | **Generated:** `2026-10-27T12:20:10.734498+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`test`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.7613** | Primary optimization metric per §18.1 |
| Accuracy | 0.9877 | Overall sample accuracy |
| Macro Precision | 0.7563 | Unweighted class average |
| Macro Recall | 0.7681 | Unweighted class average |
| Evaluated Split Samples | 162 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High` | 1.0000 | 1.0000 | 1.0000 | 9 |
| `EC_Low` | 0.0000 | 0.0000 | 0.0000 | 0 |
| `Humidity_Low` | 1.0000 | 1.0000 | 1.0000 | 16 |
| `Multiple_Stress` | 1.0000 | 0.9130 | 0.9545 | 23 |
| `Normal` | 1.0000 | 1.0000 | 1.0000 | 87 |
| `WaterTemp_High` | 0.9500 | 1.0000 | 0.9744 | 19 |
| `WaterTemp_Low` | 0.0000 | 0.0000 | 0.0000 | 0 |
| `pH_High` | 0.8571 | 1.0000 | 0.9231 | 6 |
| `pH_Low` | 1.0000 | 1.0000 | 1.0000 | 2 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High` | `EC_Low` | `Humidity_Low` | `Multiple_Stress` | `Normal` | `WaterTemp_High` | `WaterTemp_Low` | `pH_High` | `pH_Low` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High` | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `EC_Low` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Humidity_Low` | 0 | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Multiple_Stress` | 0 | 0 | 0 | 21 | 0 | 1 | 0 | 1 | 0 |
| `Normal` | 0 | 0 | 0 | 0 | 87 | 0 | 0 | 0 | 0 |
| `WaterTemp_High` | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 0 |
| `WaterTemp_Low` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `pH_High` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| `pH_Low` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/sensor/lettuce/manifest.json`
- **Split Protocol:** Preserved calendar day / group holdout (test.csv untouched until final evaluation)
- **Preprocessing Version:** `1.0.0`
- **Label-Generation Method:** Rule-derived thresholds mapped from physical hydroponic solution bands

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 1295 |
| Validation | 125 |
| Test (Held-Out) | 162 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `DecisionTreeClassifier`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "criterion": "entropy",
  "max_depth": 5,
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
- Small Physical Dataset: Held-out test set contains 162 physical sensor readings across 12 disjoint calendar days. Low-frequency classes EC_Low and WaterTemp_Low were excluded due to lack of support.
