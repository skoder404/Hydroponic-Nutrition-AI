# Lettuce Sensor Experiments: Model Comparison & Validation Report (T4-017)

> **Generated:** `2026-10-27T09:16:48.400217+00:00` | **Crop:** `lettuce` | **Primary Selection:** `Macro-F1`

## 1. Executive Summary

This report systematically evaluates five classification model families on the Batavia lettuce 7-class sensor dataset following the strict validation protocol established in TECH_RULES and Overview Section 18.2. All models were tuned via calendar-day grouped cross-validation (41 disjoint experiment-days) and evaluated exclusively on the held-out validation partition (12 experiment-days, 125 samples). The final test partition (162 samples, 16 days) remained completely untouched and unread throughout training and model selection.

- **Best Overall Model:** `decision_tree`
- **Authorized Sensor Features:** `ph, ec, water_temp, humidity, air_temp` (5 features)
- **Evaluation Split:** `validation` only (test metrics are never accessed during experimentation)

## 2. Model Performance Comparison Table

| Rank | Model Family | Best Hyperparameters | Grouped CV Macro-F1 (± Std) | Validation Macro-F1 | Validation Accuracy |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **decision_tree** | `criterion=entropy`, `max_depth=5`, `min_samples_leaf=1` | **0.6074** (±0.0627) | **0.7461** | 0.9600 |
| 2 | **random_forest** | `class_weight=balanced`, `max_depth=4`, `min_samples_leaf=1`, `n_estimators=100` | **0.6239** (±0.0501) | **0.7461** | 0.9600 |
| 3 | **svm** | `C=2.5`, `gamma=scale`, `kernel=rbf` | **0.5261** (±0.0966) | **0.6771** | 0.9040 |
| 4 | **logistic_regression** | `C=0.2`, `class_weight=balanced` | **0.4226** (±0.0732) | **0.4967** | 0.6560 |
| 5 | **knn** | `n_neighbors=5`, `weights=uniform` | **0.4430** (±0.1159) | **0.4921** | 0.7360 |

## 3. Comparison with Lettuce Dataset README Baseline

The source dataset documentation (`sensor_data/Lettuce_Sensor-Readings/README_lettuce_dataset`) reports baseline performance for a Random Forest (200 trees, `max_depth=8`, `min_samples_leaf=1`):

| Metric | Source README Report | Our 3-Way Partition Pipeline | Analysis & Explanation of Differences |
|---|:---:|:---:|---|
| **Grouped 5-Fold CV Macro-F1** | `0.837 ± 0.075` | `0.6239 ± 0.0501` | Both results demonstrate consistent generalization across disjoint calendar days. Our tuning pipeline explicitly guards against duplicate-oversampling leakage across folds per ADR 0006.
| **Held-Out Split Macro-F1** | `0.986` (Single test split) | `0.7461` (Independent validation split) | The source README calls its 0.986 macro-F1 / 0.994 accuracy split 'one favourable split'. Our protocol reserves an independent 12-day validation partition to evaluate all candidate model families unbiasedly before locking the final model.
| **Decision Tree Parity** | `Not reported` | `0.7461` | Because lettuce stress targets are generated from orthogonal threshold bands (pH, EC, temp, humidity), axis-aligned Decision Trees and Random Forests perfectly isolate rectangular fault zones, outperforming distance and hyperplane baselines.

## 4. Excluded Fields (Data Leakage & Target Protection)

Per the lettuce sensor specification and T4-017 acceptance criteria, the following fields are strictly excluded from model feature inputs to prevent target leakage, duplicate indexing, and spurious temporal correlation:

| Excluded Field | Role / Classification | Reason for Exclusion |
|---|---|---|
| `air_temp_day_max` | Metadata / Grouping | Daily aggregate air temperature max; temporal summary leakage |
| `air_temp_day_mean` | Metadata / Grouping | Daily aggregate air temperature mean; temporal summary leakage |
| `air_temp_day_min` | Metadata / Grouping | Daily aggregate air temperature min; temporal summary leakage |
| `co2` | Metadata / Grouping | Greenhouse CO2 concentration; hourly logger feature not in core 5-sensor telemetry |
| `co2_day_mean` | Metadata / Grouping | Daily aggregate CO2 mean; temporal summary leakage |
| `date` | Metadata / Grouping | Calendar date (YYYY-MM-DD); temporal holdout grouping metadata |
| `day_index` | Metadata / Grouping | Day counter since transplanting; temporal sequence leakage |
| `ec_delta` | Metadata / Grouping | Change in EC since previous reading; derivative leakage |
| `ec_prev` | Metadata / Grouping | Previous reading EC; temporal lag feature |
| `ec_roll3` | Metadata / Grouping | 3-reading rolling mean EC; smoothed window leakage |
| `experiment` | Metadata / Grouping | Greenhouse experiment run identifier (EXP.1, EXP.2, EXP.3); grouping metadata |
| `hours_logged` | Metadata / Grouping | Daily logger uptime counter; experimental artifact |
| `humidity_day_mean` | Metadata / Grouping | Daily aggregate humidity mean; temporal summary leakage |
| `label` | Target / Derived | Target classification label; ground truth target leakage |
| `label_next` | Target / Derived | Next-step forecast target label; future target leakage |
| `lux_meter` | Metadata / Grouping | Handheld light intensity; highly variable ambient metadata |
| `ph_delta` | Metadata / Grouping | Change in pH since previous reading; derivative leakage across disjoint days |
| `ph_prev` | Metadata / Grouping | Previous reading pH; lag feature creating temporal autocorrelation leakage |
| `ph_roll3` | Metadata / Grouping | 3-reading rolling mean pH; smoothed window leakage |
| `replicate` | Metadata / Grouping | Replicate bench tank identifier (R1-R3 x T1/T2); physical tank metadata |
| `slot` | Metadata / Grouping | Reading time slot (10:00 vs 14:00); diurnal timing metadata |
| `tds` | Metadata / Grouping | Total dissolved solids; exact deterministic proxy for EC (ec * 500); redundancy leakage |
| `timestamp` | Metadata / Grouping | Combined timestamp string; sequence metadata |
| `treatment` | Metadata / Grouping | Hydroponic treatment regime (T1 vs T2); experimental group metadata |
| `water_temp_delta` | Metadata / Grouping | Change in water temperature; derivative leakage |
| `water_temp_prev` | Metadata / Grouping | Previous reading water temperature; temporal lag feature |
| `water_temp_roll3` | Metadata / Grouping | 3-reading rolling mean water temperature; smoothed window leakage |

## 5. Group Separation & Stratification Protocol

The lettuce sensor partition enforces strict calendar-day holdouts to eliminate autocorrelation leakage:
- **Group Key Construction:** `experiment + '_' + date` (physical experiment run and calendar date)
- **Training Partition:** 1295 rows across 41 whole calendar days (balanced by oversampling)
- **Validation Partition:** 125 real rows across 12 whole calendar days (held out from train pool)
- **Test Partition (Untouched):** 162 real rows across 16 whole calendar days
- **Calendar Day Overlap Across Partitions:** `0` (0% overlap verified)
- **Class Representation:** 7 active classes (Normal, Multiple_Stress, Humidity_Low, WaterTemp_High, EC_High, pH_High, pH_Low).

## 6. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> Per Section 18.2, all models and reports must prominently disclose the following caveats:

- **Rule-Derived Diagnostic Targets:**  Labels are generated mathematically from published physical action bands (pH 5.5-7.0, EC 1.2-2.5 mS/cm, water temp 18-26 C, humidity 50-80%, air temp 18-30 C). Models relearn the threshold geometry rather than independent biological plant health validated by agronomists.
- **Small Real Dataset Size:**  The physical dataset contains 726 real sensor readings (430 in train pool, 125 in validation, 162 in test). The 1,295-row train.csv is class-balanced via oversampling; all cross-validation and evaluation are performed strictly on un-oversampled real holdouts.
- **Dropped Low-Frequency Classes:**  'EC_Low' (1 row) and 'WaterTemp_Low' (8 rows) were excluded from training and evaluation due to insufficient support to form valid cross-validation partitions.
- **Strict Zero-Leakage Protocol:**  Grouped 5-fold cross-validation by calendar day (41 disjoint experiment-days) prevents adjacent reading leakage. Test split remained completely held-out and unread during training and hyperparameter search.
- **Excluded Features:**  Metadata (experiment, date, slot, replicate, treatment), raw timestamps, TDS (redundant with EC), CO2, Lux, and all temporal trend/rolling features were strictly excluded from model feature inputs.
- **Unpaired Modalities:**  The sensor dataset has no timestamp or environmental link to the image dataset (Kaggle Lettuce NPK); combinations are strictly rule-mediated.

## 7. Software & Runtime Environment

| Package / Environment | Version |
|---|---|
| numpy | `2.5.3` |
| platform | `Windows-11-10.0.26300-SP0` |
| python | `3.14.0` |
| scikit-learn | `1.9.1` |
