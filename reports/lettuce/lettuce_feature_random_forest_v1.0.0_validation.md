# Model Evaluation Report: Lettuce — lettuce_feature_random_forest_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T08:43:14.545673+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.4930** | Primary optimization metric per §18.1 |
| Accuracy | 0.6094 | Overall sample accuracy |
| Macro Precision | 0.5106 | Unweighted class average |
| Macro Recall | 0.4901 | Unweighted class average |
| Evaluated Split Samples | 128 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Healthier` | 0.8679 | 0.7302 | 0.7931 | 63 |
| `NitrogenDeficiency` | 0.0870 | 0.2000 | 0.1212 | 10 |
| `PhosphorusDeficiency` | 0.6875 | 0.6667 | 0.6769 | 33 |
| `PotassiumDeficiency` | 0.4000 | 0.3636 | 0.3810 | 22 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Healthier` | `NitrogenDeficiency` | `PhosphorusDeficiency` | `PotassiumDeficiency` |
|---|---:|---:|---:|---:|
| `Healthier` | 46 | 12 | 2 | 3 |
| `NitrogenDeficiency` | 3 | 2 | 1 | 4 |
| `PhosphorusDeficiency` | 3 | 3 | 22 | 5 |
| `PotassiumDeficiency` | 1 | 6 | 7 | 8 |

## 4. Dataset Splits & Protocol

- **Dataset Manifest Reference:** `processed_data/features/lettuce`
- **Split Protocol:** Stratified split inherited from parent plant images; train/val isolated
- **Preprocessing Version:** `0.1.0`
- **Label-Generation Method:** Whole-plant agronomic condition inherited by individual leaf crops (weak label)

### Partition Counts

| Partition | Total Samples |
|---|---:|
| Train | 703 |
| Validation | 128 |
| Test (Held-Out) | 0 |

## 5. Model Architecture & Hyperparameters

- **Model Family:** `RandomForest`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "n_estimators": 150,
  "max_depth": 8,
  "min_samples_split": 3,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
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

- Features are extracted from segmented lettuce leaf crops derived from whole-plant imagery.
- Ground truth labels inherit the whole-plant condition (weak labels); an estimated 15.8% of leaf crops on deficient plants present normal green morphology.
- Model performance is evaluated exclusively on the validation holdout split; test.csv remains untouched.
