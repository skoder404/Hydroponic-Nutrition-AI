# Model Evaluation Report: Lettuce — lettuce_feature_svm_v1.0.0

> **Evaluated Split:** `validation` | **Generated:** `2026-10-28T08:43:14.701090+00:00`

## 1. Executive Summary & Core Metrics

Metrics below reflect performance exclusively on the **`validation`** partition.

| Metric | Score | Note |
|---|---:|---|
| **Macro-F1** (Primary Selection) | **0.4840** | Primary optimization metric per §18.1 |
| Accuracy | 0.5156 | Overall sample accuracy |
| Macro Precision | 0.5428 | Unweighted class average |
| Macro Recall | 0.5071 | Unweighted class average |
| Evaluated Split Samples | 128 | Total samples in evaluated partition |

## 2. Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Healthier` | 0.8378 | 0.4921 | 0.6200 | 63 |
| `NitrogenDeficiency` | 0.1026 | 0.4000 | 0.1633 | 10 |
| `PhosphorusDeficiency` | 0.7826 | 0.5455 | 0.6429 | 33 |
| `PotassiumDeficiency` | 0.4483 | 0.5909 | 0.5098 | 22 |

## 3. Confusion Matrix

Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):

| True \ Pred | `Healthier` | `NitrogenDeficiency` | `PhosphorusDeficiency` | `PotassiumDeficiency` |
|---|---:|---:|---:|---:|
| `Healthier` | 31 | 25 | 2 | 5 |
| `NitrogenDeficiency` | 3 | 4 | 0 | 3 |
| `PhosphorusDeficiency` | 2 | 5 | 18 | 8 |
| `PotassiumDeficiency` | 1 | 5 | 3 | 13 |

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

- **Model Family:** `SVM`
- **Artifact Version:** `v1.0.0`
- **Target Device:** `cpu`
- **Random Seeds:** `{"seed": 42}`

```json
{
  "C": 2.0,
  "gamma": "scale",
  "kernel": "rbf",
  "class_weight": "balanced",
  "probability": true,
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
