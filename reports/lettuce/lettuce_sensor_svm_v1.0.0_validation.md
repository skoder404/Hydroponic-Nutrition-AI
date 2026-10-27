# Model Evaluation Report: Lettuce — lettuce_sensor_svm_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T09:16:45.326609+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.6771** | Primary optimization metric per §18.1 |
| Accuracy | 0.9040 | Overall sample accuracy |
| Macro Precision | 0.6518 | Unweighted class average |
| Macro Recall | 0.7361 | Unweighted class average |
| Evaluated Split Samples | 125 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High` | 0.5882 | 1.0000 | 0.7407 | 10 |
| `EC_Low` | 0.0000 | 0.0000 | 0.0000 | 0 |
| `Humidity_Low` | 0.9091 | 0.9091 | 0.9091 | 11 |
| `Multiple_Stress` | 1.0000 | 0.8065 | 0.8929 | 31 |
| `Normal` | 0.9804 | 0.9091 | 0.9434 | 55 |
| `WaterTemp_High` | 0.8889 | 1.0000 | 0.9412 | 16 |
| `WaterTemp_Low` | 0.0000 | 0.0000 | 0.0000 | 0 |
| `pH_High` | 1.0000 | 1.0000 | 1.0000 | 1 |
| `pH_Low` | 0.5000 | 1.0000 | 0.6667 | 1 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High` | `EC_Low` | `Humidity_Low` | `Multiple_Stress` | `Normal` | `WaterTemp_High` | `WaterTemp_Low` | `pH_High` | `pH_Low` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EC_High` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `EC_Low` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Humidity_Low` | 0 | 0 | 10 | 0 | 1 | 0 | 0 | 0 | 0 |
| `Multiple_Stress` | 4 | 0 | 1 | 25 | 0 | 1 | 0 | 0 | 0 |
| `Normal` | 3 | 0 | 0 | 0 | 50 | 1 | 0 | 0 | 1 |
| `WaterTemp_High` | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 0 |
| `WaterTemp_Low` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `pH_High` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `pH_Low` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/sensor/lettuce/manifest.json`
- **Split Protocol:** Preserved calendar day / group holdout (test.csv untouched)
- **Preprocessing Version:** `1.0.0`
- **Label-Generation Method:** Rule-derived thresholds mapped from physical hydroponic solution bands

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 1295 |
| Validation | 125 |
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
  "gamma": "scale",
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

- Rule-Derived Diagnostic Targets: Labels are generated mathematically from published physical action bands (pH 5.5-7.0, EC 1.2-2.5 mS/cm, water temp 18-26 C, humidity 50-80%, air temp 18-30 C). Models relearn the threshold geometry rather than independent biological plant health validated by agronomists.
- Small Real Dataset Size: The physical dataset contains 726 real sensor readings (430 in train pool, 125 in validation, 162 in test). The 1,295-row train.csv is class-balanced via oversampling; all cross-validation and evaluation are performed strictly on un-oversampled real holdouts.
- Dropped Low-Frequency Classes: 'EC_Low' (1 row) and 'WaterTemp_Low' (8 rows) were excluded from training and evaluation due to insufficient support to form valid cross-validation partitions.
- Strict Zero-Leakage Protocol: Grouped 5-fold cross-validation by calendar day (41 disjoint experiment-days) prevents adjacent reading leakage. Test split remained completely held-out and unread during training and hyperparameter search.
- Excluded Features: Metadata (experiment, date, slot, replicate, treatment), raw timestamps, TDS (redundant with EC), CO2, Lux, and all temporal trend/rolling features were strictly excluded from model feature inputs.
- Unpaired Modalities: The sensor dataset has no timestamp or environmental link to the image dataset (Kaggle Lettuce NPK); combinations are strictly rule-mediated.
