# Model Evaluation Report: Spinach — spinach_sensor_knn_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-27T08:17:38.021703+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.9883** | Primary optimization metric per §18.1 |
| Accuracy | 0.9877 | Overall sample accuracy |
| Macro Precision | 0.9876 | Unweighted class average |
| Macro Recall | 0.9897 | Unweighted class average |
| Evaluated Split Samples | 162 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `EC_High_Toxicity` | 1.0000 | 1.0000 | 1.0000 | 21 |
| `EC_Low_Deficiency` | 1.0000 | 1.0000 | 1.0000 | 24 |
| `Humidity_High_Mildew_Risk` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `Humidity_Low_Stress` | 1.0000 | 1.0000 | 1.0000 | 14 |
| `Normal` | 0.9130 | 1.0000 | 0.9545 | 21 |
| `Water_Temp_High` | 1.0000 | 0.9583 | 0.9787 | 24 |
| `pH_Imbalance` | 1.0000 | 0.9697 | 0.9846 | 33 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `EC_High_Toxicity` | `EC_Low_Deficiency` | `Humidity_High_Mildew_Risk` | `Humidity_Low_Stress` | `Normal` | `Water_Temp_High` | `pH_Imbalance` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `EC_High_Toxicity` | 21 | 0 | 0 | 0 | 0 | 0 | 0 |
| `EC_Low_Deficiency` | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| `Humidity_High_Mildew_Risk` | 0 | 0 | 25 | 0 | 0 | 0 | 0 |
| `Humidity_Low_Stress` | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| `Normal` | 0 | 0 | 0 | 0 | 21 | 0 | 0 |
| `Water_Temp_High` | 0 | 0 | 0 | 0 | 1 | 23 | 0 |
| `pH_Imbalance` | 0 | 0 | 0 | 0 | 1 | 0 | 32 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/sensor/spinach/manifest.json`
- **Split Protocol:** Preserved calendar day / group holdout (test.csv untouched)
- **Preprocessing Version:** `1.0.0`
- **Label-Generation Method:** Rule-derived thresholds mapped from physical hydroponic solution bands

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 958 |
| Validation | 162 |
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
| python | `3.13.5` |
| scikit-learn | `1.9.1` |
| torch | `2.7.1+cu118` |

## 7. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> The following known caveats and constraints must remain visible in documentation and UI:

- Mixed Real and Synthetic Features: Greenhouse ambient air temperature (air_temperature_C) is 100% real observational data from commercial loggers. Relative humidity (relative_humidity_percent) is 85.7% real and 14.3% adjusted. Solution water temperature (water_temperature_C), pH (ph), and electrical conductivity (ec_mS_cm) are synthetic simulated features.
- Rule-Derived Diagnostic Targets: Labels are mathematically generated from hard agronomic threshold bounds; models learn the threshold rules rather than independent plant biological responses.
- Artificially Balanced Class Distribution: Each stress class contains exactly 200 samples (160 train+val, 40 test), whereas real hydroponic greenhouses exhibit severe skew toward Normal status (>95%).
- Strict Zero-Leakage Protocol: 6-hour temporal block separation (bench | date | hour//6) prevents adjacent reading leakage. Test split remained completely held-out and unread during training and hyperparameter search.
- Excluded Non-Feature Columns: tds_ppm, label_id, humidity_source, sample_id, source_timestamp_EDT, and bench were strictly excluded from model feature inputs.
