# Spinach Sensor Experiments: Model Comparison & Validation Report (T4-018)

> **Generated:** `2026-10-27T08:17:39.580264+00:00` | **Crop:** `spinach` | **Primary Selection:** `Macro-F1`

## 1. Executive Summary

This report evaluates five classification model families on the spinach 7-class sensor dataset following the strict validation protocol established in TECH_RULES and Overview Section 18.2. All models were tuned via group-aware cross-validation and evaluated exclusively on the held-out validation partition (11 six-hour blocks, 162 samples). The final test partition (280 samples, 22 blocks) remained completely untouched and unread throughout training and model selection.

- **Best Overall Model:** `decision_tree`
- **Authorized Sensor Features:** `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm` (5 features)
- **Evaluation Split:** `validation` only (test metrics are never accessed during experimentation)

## 2. Model Performance Comparison Table

| Rank | Model Family | Best Hyperparameters | Grouped CV Macro-F1 (± Std) | Validation Macro-F1 | Validation Accuracy |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **decision_tree** | `criterion=gini`, `max_depth=8`, `min_samples_leaf=1` | **1.0000** (±0.0000) | **1.0000** | 1.0000 |
| 2 | **random_forest** | `class_weight=None`, `max_depth=6`, `min_samples_leaf=1`, `n_estimators=25` | **1.0000** (±0.0000) | **1.0000** | 1.0000 |
| 3 | **svm** | `C=2.5`, `gamma=0.1`, `kernel=rbf` | **0.9805** (±0.0062) | **1.0000** | 1.0000 |
| 4 | **knn** | `n_neighbors=9`, `weights=distance` | **0.9629** (±0.0040) | **0.9883** | 0.9877 |
| 5 | **logistic_regression** | `C=1.0`, `class_weight=None` | **0.8508** (±0.0095) | **0.8332** | 0.8086 |

## 3. Excluded Fields (Data Leakage & Target Protection)

Per the spinach sensor specification and T4-018 acceptance criteria, the following 6 fields are strictly excluded from model feature inputs to prevent target leakage, duplicate indexing, and spurious temporal correlation:

| Excluded Field | Role / Classification | Reason for Exclusion |
|---|---|---|
| `tds_ppm` | Target / Derived | Derived/redundant with EC (EC * 700 + noise); high data leakage risk |
| `label_id` | Target / Derived | Direct integer encoding of target label (0-6); target leakage |
| `humidity_source` | Metadata / Grouping | Dataset generation metadata flag ('real' vs 'adjusted'); provenance leakage |
| `sample_id` | Metadata / Grouping | Arbitrary unique row identifier (S0001-S1400); non-generalizable |
| `source_timestamp_EDT` | Metadata / Grouping | Logger timestamp metadata; leaks temporal sequence across holdouts |
| `bench` | Metadata / Grouping | Greenhouse physical bench identifier; location metadata |

## 4. Group Separation & Stratification Protocol

The spinach sensor partition enforces strict six-hour block holdouts to eliminate autocorrelation leakage:
- **Block Key Construction:** `bench + '|' + source_timestamp_EDT (YYYY-MM-DD) + '|' + (hour // 6)`
- **Training Partition:** 958 rows across 77 whole 6-hour blocks
- **Validation Partition:** 162 rows across 11 whole 6-hour blocks (held out from train pool)
- **Test Partition (Untouched):** 280 rows across 22 whole 6-hour blocks
- **Total Rows:** 1400 | **Total Blocks:** 110
- **Block Overlap Across Partitions:** `0` (0% overlap verified)
- **Sample ID Overlap Across Partitions:** `0` (0% overlap verified)
- **Per-Class Split Preservation:** Exactly 160 samples per class in the train+val pool, and 40 samples per class in the test pool across all 7 classes.

## 5. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> Per Section 18.2, all models and reports must prominently disclose the following caveats:

- **Mixed Real and Synthetic Features:**  Greenhouse ambient air temperature (air_temperature_C) is 100% real observational data from commercial loggers. Relative humidity (relative_humidity_percent) is 85.7% real and 14.3% adjusted. Solution water temperature (water_temperature_C), pH (ph), and electrical conductivity (ec_mS_cm) are synthetic simulated features.
- **Rule-Derived Diagnostic Targets:**  Labels are mathematically generated from hard agronomic threshold bounds; models learn the threshold rules rather than independent plant biological responses.
- **Artificially Balanced Class Distribution:**  Each stress class contains exactly 200 samples (160 train+val, 40 test), whereas real hydroponic greenhouses exhibit severe skew toward Normal status (>95%).
- **Strict Zero-Leakage Protocol:**  6-hour temporal block separation (bench | date | hour//6) prevents adjacent reading leakage. Test split remained completely held-out and unread during training and hyperparameter search.
- **Excluded Non-Feature Columns:**  tds_ppm, label_id, humidity_source, sample_id, source_timestamp_EDT, and bench were strictly excluded from model feature inputs.

## 6. Software & Runtime Environment

| Package / Environment | Version |
|---|---|
| numpy | `2.5.3` |
| platform | `Windows-11-10.0.26300-SP0` |
| python | `3.13.5` |
| scikit-learn | `1.9.1` |
| torch | `2.7.1+cu118` |
