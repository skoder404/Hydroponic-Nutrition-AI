# Model Evaluation Report: Tomato — tomato_sensor_decision_tree_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T09:32:49.065987+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.9573** | Primary optimization metric per §18.1 |
| Accuracy | 0.9576 | Overall sample accuracy |
| Macro Precision | 0.9574 | Unweighted class average |
| Macro Recall | 0.9573 | Unweighted class average |
| Evaluated Split Samples | 5377 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 0.9648 | 0.9757 | 0.9702 | 534 |
| `EC_Low_Deficiency` | 0.9523 | 0.9596 | 0.9559 | 520 |
| `Humidity_High_Mildew_Risk` | 0.9682 | 0.9719 | 0.9700 | 533 |
| `Humidity_Low_Stress` | 0.9610 | 0.9593 | 0.9601 | 540 |
| `K_Deficiency` | 0.9580 | 0.9686 | 0.9633 | 542 |
| `N_Deficiency` | 0.9515 | 0.9341 | 0.9427 | 546 |
| `Normal` | 1.0000 | 1.0000 | 1.0000 | 549 |
| `P_Deficiency` | 0.9371 | 0.9318 | 0.9345 | 528 |
| `Water_Temp_High` | 0.9481 | 0.9597 | 0.9539 | 571 |
| `pH_Imbalance` | 0.9324 | 0.9125 | 0.9223 | 514 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `K_Deficiency` | `N_Deficiency` | `Normal` | `P_Deficiency` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 521 | 0 | 1 | 2 | 1 | 4 | 0 | 1 | 2 | 2 |
| `EC_Low_Deficiency` | 0 | 499 | 2 | 0 | 5 | 0 | 0 | 4 | 1 | 9 |
| `Humidity_High_Mildew_Risk` | 3 | 1 | 518 | 0 | 1 | 3 | 0 | 2 | 2 | 3 |
| `Humidity_Low_Stress` | 2 | 6 | 0 | 518 | 1 | 1 | 0 | 3 | 7 | 2 |
| `K_Deficiency` | 0 | 2 | 1 | 3 | 525 | 0 | 0 | 5 | 3 | 3 |
| `N_Deficiency` | 2 | 0 | 7 | 5 | 6 | 510 | 0 | 7 | 2 | 7 |
| `Normal` | 0 | 0 | 0 | 0 | 0 | 0 | 549 | 0 | 0 | 0 |
| `P_Deficiency` | 4 | 2 | 2 | 6 | 5 | 5 | 0 | 492 | 6 | 6 |
| `Water_Temp_High` | 4 | 6 | 1 | 2 | 1 | 5 | 0 | 2 | 548 | 2 |
| `pH_Imbalance` | 4 | 8 | 3 | 3 | 3 | 8 | 0 | 9 | 7 | 469 |

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

- **Model Family:** `DecisionTreeClassifier`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "criterion": "gini",
  "max_depth": null,
  "min_samples_leaf": 1,
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
