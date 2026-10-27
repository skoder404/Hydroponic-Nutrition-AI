# Model Evaluation Report: Tomato — tomato_sensor_random_forest_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T09:40:28.128800+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.9689** | Primary optimization metric per §18.1 |
| Accuracy | 0.9695 | Overall sample accuracy |
| Macro Precision | 0.9702 | Unweighted class average |
| Macro Recall | 0.9689 | Unweighted class average |
| Evaluated Split Samples | 5377 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 0.9468 | 1.0000 | 0.9727 | 534 |
| `EC_Low_Deficiency` | 0.9765 | 0.9596 | 0.9680 | 520 |
| `Humidity_High_Mildew_Risk` | 0.9545 | 0.9850 | 0.9695 | 533 |
| `Humidity_Low_Stress` | 0.9621 | 0.9870 | 0.9744 | 540 |
| `K_Deficiency` | 0.9691 | 0.9852 | 0.9771 | 542 |
| `N_Deficiency` | 0.9669 | 0.9615 | 0.9642 | 546 |
| `Normal` | 1.0000 | 1.0000 | 1.0000 | 549 |
| `P_Deficiency` | 0.9623 | 0.9678 | 0.9651 | 528 |
| `Water_Temp_High` | 0.9655 | 0.9790 | 0.9722 | 571 |
| `pH_Imbalance` | 0.9978 | 0.8638 | 0.9260 | 514 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `K_Deficiency` | `N_Deficiency` | `Normal` | `P_Deficiency` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 534 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `EC_Low_Deficiency` | 0 | 499 | 7 | 2 | 3 | 1 | 0 | 5 | 3 | 0 |
| `Humidity_High_Mildew_Risk` | 5 | 0 | 525 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| `Humidity_Low_Stress` | 4 | 0 | 0 | 533 | 1 | 0 | 0 | 1 | 1 | 0 |
| `K_Deficiency` | 2 | 0 | 1 | 1 | 534 | 1 | 0 | 1 | 2 | 0 |
| `N_Deficiency` | 4 | 0 | 5 | 3 | 4 | 525 | 0 | 3 | 1 | 1 |
| `Normal` | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 0 | 0 | 0 |
| `P_Deficiency` | 4 | 0 | 1 | 3 | 2 | 4 | 0 | 511 | 3 | 0 |
| `Water_Temp_High` | 4 | 1 | 2 | 1 | 1 | 3 | 0 | 0 | 559 | 0 |
| `pH_Imbalance` | 7 | 11 | 9 | 11 | 5 | 8 | 0 | 9 | 10 | 444 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/sensor/tomato/manifest.json`
- **Split Protocol:** Preserved calendar day / group holdout (test.csv untouched)
- **Preprocessing Version:** `1.0.0`
- **Label-Generation Method:** Rule-derived thresholds mapped from physical hydroponic solution bands

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 29943 |
| Validation | 5377 |
| Test (Held-Out) | 0 |

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
| python | `3.14.0` |
| scikit-learn | `1.9.1` |

## 7. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> The following known caveats and constraints must remain visible in documentation and UI:

- Experimental Status: Models are marked experimental=true in ModelMetadata until live-sensor telemetry is verified against campus greenhouse hardware (Overview §15.7).
- Proxy-Transformed & Simulated Features: The underlying Kolar tomato dataset originates from open-field precision agriculture with simulated noise. Nutrient solution features (water_temperature_C, pH, EC, N, P, K) were generated using calibrated proxy transformations rather than direct hydroponic tank sensing.
- Rule-Derived Diagnostic Targets: Labels are generated mathematically using multi-fault severity ranking over nine threshold rules (Water_Temp_High, EC_High/Low, pH_Imbalance, Humidity_High/Low, N/P/K_Deficiency).
- Strict Zero-Leakage Protocol: Grouped 5-fold cross-validation by calendar month (89 disjoint months) prevents temporal autocorrelation leakage. Test split remained completely held-out and unread during training and hyperparameter search.
- Excluded Features: n_rules_broken, source_row, source_timestamp, sample_id, and label_id were strictly excluded from model feature inputs.
