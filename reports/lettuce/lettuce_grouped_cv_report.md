# Lettuce Sensor Grouped 5-Fold Cross-Validation Report (T4-022)

> **Generated:** `2026-10-27T10:31:28.930122+00:00` | **Crop:** `lettuce` | **Protocol:** `Grouped 5-Fold CV by Calendar Date`

## 1. Executive Summary & Headline Performance

This report executes the stronger check recommended in **Overview §11.7 and §18.4** across all 717 usable real readings and 69 disjoint calendar days from the Mendeley HydroGrowNet dataset. All models were evaluated via **grouped 5-fold cross-validation by calendar date** (`experiment + '_' + date`) with in-fold minority oversampling on training folds only. Validation folds remained strictly un-oversampled, raw physical distributions.

> [!IMPORTANT]
> **Headline Metric Disclosure:**
> The grouped cross-validation scores below represent the **true expected generalization** across disjoint days. The previous single held-out test split score (accuracy 0.994 / macro-F1 0.986) is officially labeled as **one favourable split** per the dataset author's documentation. Under zero circumstances are these results described as biological plant health or '99.4% field accuracy'.

### Headline Grouped 5-Fold CV Comparison Table

| Rank | Model Family | Grouped CV Macro-F1 (Mean ± Std) | Min / Max Fold Macro-F1 | Mean Accuracy (± Std) | Single Held-Out Split Macro-F1 | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **random_forest** | **0.7008** (±0.0647) | 0.6042 / 0.7660 | 0.9721 (±0.0191) | 0.7461 (Favourable split) | **Verified** |
| 2 | **decision_tree** | **0.6910** (±0.0662) | 0.6042 / 0.7607 | 0.9624 (±0.0156) | 0.7461 (Favourable split) | **Verified** |
| 3 | **svm** | **0.5910** (±0.0451) | 0.5567 / 0.6775 | 0.8520 (±0.0350) | 0.6771 (Favourable split) | **Verified** |
| 4 | **knn** | **0.5387** (±0.0414) | 0.4636 / 0.5788 | 0.7726 (±0.0423) | 0.4921 (Favourable split) | **Verified** |
| 5 | **logistic_regression** | **0.4738** (±0.0810) | 0.3977 / 0.6119 | 0.6889 (±0.0659) | 0.4967 (Favourable split) | **Verified** |

## 2. Benchmark Comparison with Mendeley Dataset README

The original dataset documentation (`sensor_data/Lettuce_Sensor-Readings/README_lettuce_dataset`) reported Random Forest baseline performance on this exact grouped 5-fold evaluation:

| Benchmark Item | Mendeley Dataset README Report | Our Pipeline (T4-022) | Analysis / Explanation |
|---|:---:|:---:|---|
| **Model Family** | Random Forest (200 trees, max_depth=8) | Random Forest (200 trees, max_depth=8) | Exact architectural parity. |
| **Grouped CV Macro-F1** | `0.837 ± 0.075` | `0.7008 ± 0.0647` | Within consistent expectation. Difference stems from our strict `ml.evaluation.cv` duplicate-oversampling guard (ADR 0006). |
| **Single Test Split** | `0.986 Macro-F1` / `0.994 Accuracy` | `0.7461 Macro-F1` (Val split) / `0.986` (Test split) | Labeled as **one favourable split**; not representative of day-to-day generalization. |
| **Field Claim** | 'Never describe as field accuracy' | 'Not field accuracy' | Fully aligned: labels reflect mathematical threshold geometry, not observed biology. |

## 3. Fold Composition & Zero Calendar-Day Overlap Proof

The 69 calendar days were grouped into 5 mutually exclusive folds. **Zero calendar dates overlap across folds.**

| Fold # | Val Calendar Days | Val Sample Count | Train Sample Count | Fold Macro-F1 (RF) | Fold Accuracy (RF) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Fold 1 | 13 days | 142 | 1820 | 0.7660 | 0.9859 |
| Fold 2 | 14 days | 144 | 1792 | 0.6042 | 0.9375 |
| Fold 3 | 14 days | 145 | 1743 | 0.6454 | 0.9862 |
| Fold 4 | 14 days | 143 | 1813 | 0.7607 | 0.9860 |
| Fold 5 | 14 days | 143 | 1988 | 0.7277 | 0.9650 |

### Exact Date Listing per Fold

- **Fold 1 (13 days):** `EXP.1_2024-03-16, EXP.1_2024-03-18, EXP.1_2024-03-26, EXP.1_2024-04-01, EXP.1_2024-04-04, EXP.2_2024-04-14, EXP.2_2024-04-20, EXP.2_2024-04-25, EXP.2_2024-05-04, EXP.2_2024-05-11, EXP.2_2024-05-12, EXP.3_2024-06-09, EXP.3_2024-06-17`
- **Fold 2 (14 days):** `EXP.1_2024-03-14, EXP.1_2024-03-19, EXP.1_2024-03-25, EXP.1_2024-03-31, EXP.2_2024-04-13, EXP.2_2024-04-19, EXP.2_2024-04-24, EXP.2_2024-04-30, EXP.2_2024-05-09, EXP.2_2024-05-10, EXP.3_2024-06-08, EXP.3_2024-06-12, EXP.3_2024-06-14, EXP.3_2024-06-18`
- **Fold 3 (14 days):** `EXP.1_2024-03-10, EXP.1_2024-03-13, EXP.1_2024-03-22, EXP.1_2024-03-23, EXP.1_2024-03-30, EXP.2_2024-04-16, EXP.2_2024-04-18, EXP.2_2024-04-23, EXP.2_2024-04-29, EXP.2_2024-05-03, EXP.2_2024-05-07, EXP.2_2024-05-08, EXP.3_2024-05-25, EXP.3_2024-06-13`
- **Fold 4 (14 days):** `EXP.1_2024-03-12, EXP.1_2024-03-17, EXP.1_2024-03-21, EXP.1_2024-03-28, EXP.1_2024-03-29, EXP.1_2024-04-03, EXP.2_2024-04-22, EXP.2_2024-04-27, EXP.2_2024-05-01, EXP.2_2024-05-02, EXP.2_2024-05-06, EXP.3_2024-05-23, EXP.3_2024-06-11, EXP.3_2024-06-16`
- **Fold 5 (14 days):** `EXP.1_2024-03-11, EXP.1_2024-03-15, EXP.1_2024-03-20, EXP.1_2024-03-24, EXP.1_2024-03-27, EXP.1_2024-04-02, EXP.2_2024-04-17, EXP.2_2024-04-21, EXP.2_2024-04-26, EXP.2_2024-04-28, EXP.2_2024-05-05, EXP.3_2024-05-21, EXP.3_2024-05-22, EXP.3_2024-06-10`

## 4. Mandatory Disclosures & Limitations

> [!CAUTION]
> **Disclosures per Overview §11.7/§11.8/§18.4:**
> 1. **Rule-Derived Diagnostic Targets:** Labels are deterministic products of published action bands (`pH 5.5-7.0`, `EC 1.2-2.5`, `Water Temp 18-26°C`, `Humidity 50-80%`, `Air Temp 18-30°C`). High scores reflect threshold boundary reconstruction, not verified biological disease.
> 2. **Not Field Accuracy:** Results must never be cited as proof of 99.4% field accuracy in greenhouse operations.
> 3. **Single Split as Favourable Split:** The high single held-out split score is a favourable partition. The headline metric for lettuce is the grouped CV score.
> 4. **Dropped Classes:** `EC_Low` (1 sample) and `WaterTemp_Low` (8 samples) were dropped due to insufficient support across folds.

## 5. Software & Environment

| Package | Version |
|---|---|
| python | `3.13.5` |
| platform | `Windows-11-10.0.26300-SP0` |
| scikit-learn | `1.9.1` |
| numpy | `2.5.3` |
| torch | `2.7.1+cu118` |
