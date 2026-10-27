# Model Evaluation Report: Tomato — tomato_sensor_svm_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T10:02:51.035304+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.9187** | Primary optimization metric per §18.1 |
| Accuracy | 0.9204 | Overall sample accuracy |
| Macro Precision | 0.9255 | Unweighted class average |
| Macro Recall | 0.9193 | Unweighted class average |
| Evaluated Split Samples | 5377 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 0.8360 | 0.9831 | 0.9036 | 534 |
| `EC_Low_Deficiency` | 0.9516 | 0.9077 | 0.9291 | 520 |
| `Humidity_High_Mildew_Risk` | 0.8750 | 0.9719 | 0.9209 | 533 |
| `Humidity_Low_Stress` | 0.8771 | 0.9648 | 0.9189 | 540 |
| `K_Deficiency` | 0.9214 | 0.9520 | 0.9365 | 542 |
| `N_Deficiency` | 0.9606 | 0.8919 | 0.9250 | 546 |
| `Normal` | 0.9292 | 0.9800 | 0.9539 | 549 |
| `P_Deficiency` | 0.9515 | 0.8920 | 0.9208 | 528 |
| `Water_Temp_High` | 0.9604 | 0.9335 | 0.9467 | 571 |
| `pH_Imbalance` | 0.9919 | 0.7160 | 0.8316 | 514 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `K_Deficiency` | `N_Deficiency` | `Normal` | `P_Deficiency` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 525 | 0 | 2 | 5 | 1 | 0 | 1 | 0 | 0 | 0 |
| `EC_Low_Deficiency` | 0 | 472 | 16 | 14 | 6 | 2 | 1 | 4 | 4 | 1 |
| `Humidity_High_Mildew_Risk` | 7 | 0 | 518 | 0 | 5 | 0 | 1 | 1 | 0 | 1 |
| `Humidity_Low_Stress` | 14 | 0 | 0 | 521 | 3 | 0 | 1 | 1 | 0 | 0 |
| `K_Deficiency` | 7 | 3 | 5 | 6 | 516 | 1 | 3 | 0 | 1 | 0 |
| `N_Deficiency` | 16 | 4 | 15 | 10 | 8 | 487 | 1 | 3 | 2 | 0 |
| `Normal` | 5 | 0 | 2 | 3 | 1 | 0 | 538 | 0 | 0 | 0 |
| `P_Deficiency` | 15 | 2 | 5 | 14 | 7 | 4 | 5 | 471 | 4 | 1 |
| `Water_Temp_High` | 15 | 2 | 6 | 9 | 1 | 1 | 4 | 0 | 533 | 0 |
| `pH_Imbalance` | 24 | 13 | 23 | 12 | 12 | 12 | 24 | 15 | 11 | 368 |

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

- **Model Family:** `SVC`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "C": 2.5,
  "kernel": "rbf",
  "gamma": 0.1,
  "probability": true,
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
