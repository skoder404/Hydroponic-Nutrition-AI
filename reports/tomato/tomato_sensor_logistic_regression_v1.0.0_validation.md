# Model Evaluation Report: Tomato — tomato_sensor_logistic_regression_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T10:03:48.058458+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.8587** | Primary optimization metric per §18.1 |
| Accuracy | 0.8709 | Overall sample accuracy |
| Macro Precision | 0.8524 | Unweighted class average |
| Macro Recall | 0.8684 | Unweighted class average |
| Evaluated Split Samples | 5377 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 0.9434 | 0.9981 | 0.9700 | 534 |
| `EC_Low_Deficiency` | 0.9262 | 0.9173 | 0.9217 | 520 |
| `Humidity_High_Mildew_Risk` | 0.9246 | 0.9887 | 0.9556 | 533 |
| `Humidity_Low_Stress` | 0.9402 | 0.9907 | 0.9648 | 540 |
| `K_Deficiency` | 0.9180 | 0.9705 | 0.9435 | 542 |
| `N_Deficiency` | 0.9206 | 0.9341 | 0.9273 | 546 |
| `Normal` | 0.7860 | 0.7760 | 0.7809 | 549 |
| `P_Deficiency` | 0.8929 | 0.9318 | 0.9120 | 528 |
| `Water_Temp_High` | 0.8818 | 0.9142 | 0.8977 | 571 |
| `pH_Imbalance` | 0.3902 | 0.2626 | 0.3140 | 514 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `K_Deficiency` | `N_Deficiency` | `Normal` | `P_Deficiency` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 533 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `EC_Low_Deficiency` | 0 | 477 | 6 | 3 | 6 | 1 | 4 | 2 | 5 | 16 |
| `Humidity_High_Mildew_Risk` | 3 | 0 | 527 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `Humidity_Low_Stress` | 2 | 0 | 0 | 535 | 0 | 0 | 1 | 0 | 0 | 2 |
| `K_Deficiency` | 2 | 0 | 2 | 2 | 526 | 0 | 2 | 0 | 1 | 7 |
| `N_Deficiency` | 4 | 0 | 6 | 6 | 6 | 510 | 1 | 0 | 2 | 11 |
| `Normal` | 1 | 0 | 0 | 0 | 0 | 0 | 426 | 0 | 0 | 122 |
| `P_Deficiency` | 4 | 0 | 3 | 4 | 2 | 2 | 2 | 492 | 4 | 15 |
| `Water_Temp_High` | 5 | 0 | 2 | 3 | 2 | 0 | 2 | 0 | 522 | 35 |
| `pH_Imbalance` | 11 | 38 | 24 | 15 | 31 | 41 | 104 | 57 | 58 | 135 |

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

- **Model Family:** `LogisticRegression`
- **Artifact Version:** `1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "C": 5.0,
  "max_iter": 1000,
  "random_state": 42,
  "class_weight": "balanced"
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
