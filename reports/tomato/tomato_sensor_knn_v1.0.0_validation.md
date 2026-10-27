# Model Evaluation Report: Tomato — tomato_sensor_knn_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T10:03:13.360210+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.7295** | Primary optimization metric per §18.1 |
| Accuracy | 0.7359 | Overall sample accuracy |
| Macro Precision | 0.7369 | Unweighted class average |
| Macro Recall | 0.7336 | Unweighted class average |
| Evaluated Split Samples | 5377 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 0.6846 | 0.6910 | 0.6878 | 534 |
| `EC_Low_Deficiency` | 0.7694 | 0.7442 | 0.7566 | 520 |
| `Humidity_High_Mildew_Risk` | 0.7241 | 0.7486 | 0.7362 | 533 |
| `Humidity_Low_Stress` | 0.7222 | 0.6981 | 0.7100 | 540 |
| `K_Deficiency` | 0.7420 | 0.7749 | 0.7581 | 542 |
| `N_Deficiency` | 0.7326 | 0.7527 | 0.7425 | 546 |
| `Normal` | 0.7493 | 0.9362 | 0.8324 | 549 |
| `P_Deficiency` | 0.7239 | 0.7348 | 0.7293 | 528 |
| `Water_Temp_High` | 0.7532 | 0.8179 | 0.7842 | 571 |
| `pH_Imbalance` | 0.7679 | 0.4377 | 0.5576 | 514 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `K_Deficiency` | `N_Deficiency` | `Normal` | `P_Deficiency` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 369 | 0 | 17 | 18 | 14 | 17 | 41 | 20 | 27 | 11 |
| `EC_Low_Deficiency` | 0 | 387 | 15 | 25 | 18 | 19 | 9 | 23 | 11 | 13 |
| `Humidity_High_Mildew_Risk` | 10 | 24 | 399 | 0 | 21 | 21 | 21 | 14 | 17 | 6 |
| `Humidity_Low_Stress` | 23 | 18 | 0 | 377 | 12 | 16 | 27 | 23 | 31 | 13 |
| `K_Deficiency` | 18 | 25 | 15 | 6 | 420 | 16 | 7 | 14 | 13 | 8 |
| `N_Deficiency` | 20 | 13 | 19 | 17 | 24 | 411 | 9 | 14 | 11 | 8 |
| `Normal` | 8 | 1 | 10 | 13 | 2 | 0 | 514 | 0 | 1 | 0 |
| `P_Deficiency` | 15 | 11 | 14 | 23 | 18 | 24 | 11 | 388 | 15 | 9 |
| `Water_Temp_High` | 23 | 7 | 19 | 23 | 9 | 3 | 12 | 8 | 467 | 0 |
| `pH_Imbalance` | 53 | 17 | 43 | 20 | 28 | 34 | 35 | 32 | 27 | 225 |

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

- **Model Family:** `KNeighborsClassifier`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "n_neighbors": 9,
  "weights": "distance"
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
