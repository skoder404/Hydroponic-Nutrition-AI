# Tomato Sensor Experiments: Model Comparison & Validation Report (T4-019)

> **Generated:** `2026-10-27T10:05:13.221598+00:00` | **Crop:** `tomato` | **Primary Selection:** `Macro-F1` | **Status:** `experimental=True`

## 1. Executive Summary

This report systematically evaluates five classification model families on the tomato 10-class sensor dataset following the strict validation protocol established in TECH_RULES and Overview Section 18.2. All models were tuned via calendar-month grouped cross-validation (89 disjoint calendar months) and evaluated exclusively on the held-out validation partition (16 calendar months, 5,377 samples). The final test partition (8,830 samples, 27 months) remained completely untouched and unread throughout training and model selection.

- **Best Overall Model:** `random_forest`
- **Authorized Sensor Features:** `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm, nitrogen_mg_L, phosphorus_mg_L, potassium_mg_L` (8 features)
- **Artifact Status:** `experimental=True` (designated experimental until verified against physical greenhouse telemetry)
- **Evaluation Split:** `validation` only (test metrics are never accessed during experimentation)

## 2. Model Performance Comparison Table

| Rank | Model Family | Experimental Flag | Best Hyperparameters | Grouped CV Macro-F1 (± Std) | Validation Macro-F1 | Validation Accuracy |
|:---:|---|:---:|---|:---:|:---:|:---:|
| 1 | **random_forest** | `True` | `class_weight=None`, `max_depth=None`, `min_samples_leaf=1`, `n_estimators=100` | **0.9651** (±0.0048) | **0.9689** | 0.9695 |
| 2 | **decision_tree** | `True` | `criterion=gini`, `max_depth=None`, `min_samples_leaf=1` | **0.9492** (±0.0043) | **0.9573** | 0.9576 |
| 3 | **svm** | `True` | `C=2.5`, `gamma=0.1`, `kernel=rbf` | **0.9101** (±0.0054) | **0.9187** | 0.9204 |
| 4 | **logistic_regression** | `True` | `C=5.0`, `class_weight=balanced` | **0.8536** (±0.0058) | **0.8587** | 0.8709 |
| 5 | **knn** | `True` | `n_neighbors=9`, `weights=distance` | **0.7106** (±0.0118) | **0.7295** | 0.7359 |

## 3. Physical Proxy Transformations & Synthetic Feature Documentation

Per T4-019 acceptance criteria, the underlying Kolar open-field precision agriculture telemetry was transformed into hydroponic root-zone metrics using calibrated physical proxy formulas. These transformations simulate hydroponic reservoir dynamics and nutrient solution concentrations:

| Feature | Derived / Transformed From | Hydroponic Proxy Formula | Physical Justification |
|---|---|---|---|
| `air_temperature_C` | `Kolar source` | `Ambient_Temp (unchanged from Kolar open-field observational logger)` | Direct ambient temperature logger |
| `relative_humidity_percent` | `Kolar source` | `Humidity + 15.0 (calibrated offset to match enclosed greenhouse humidity)` | Enclosed canopy humidity offset |
| `water_temperature_C` | `Kolar source` | `8.0 + 0.55 * air + 0.003 * (solar - 700) + Normal(0, 0.6) [simulated reservoir thermal dynamics, seed 42]` | Solar radiation & air thermal equilibrium |
| `ph` | `Kolar source` | `Soil_pH - 1.0 (calibrated offset mapping soil pH to hydroponic nutrient solution range)` | Soil-to-solution acidity mapping |
| `ec_mS_cm` | `Kolar source` | `EC_mS_cm * 1.4 (scaled proxy mapping bulk soil salinity to hydroponic EC)` | Salinity scaling to hydroponic EC range |
| `nitrogen_mg_L` | `Kolar source` | `Nitrogen_Level * 4.0 (scaled proxy converting soil fertility test index to solution mg/L)` | Macronutrient solution concentration proxy |
| `phosphorus_mg_L` | `Kolar source` | `Phosphorus_Level * 3.0 (scaled proxy converting soil phosphorus index to solution mg/L)` | Soil-to-solution acidity mapping |
| `potassium_mg_L` | `Kolar source` | `Potassium_Level * 6.0 (scaled proxy converting soil potassium index to solution mg/L)` | Macronutrient solution concentration proxy |

## 4. Excluded Fields (Data Leakage & Target Protection)

Per the tomato sensor specification and T4-019 acceptance criteria, the following fields are strictly excluded from model feature inputs to prevent target leakage, rule indexing, and temporal sequence leakage:

| Excluded Field | Role / Classification | Reason for Exclusion |
|---|---|---|
| `label` | Target / Derived | Target classification label; ground truth target leakage |
| `label_id` | Target / Derived | Direct integer encoding of target label (0-9); target leakage |
| `n_rules_broken` | Target / Derived | Count of threshold rule violations; direct target and severity leakage |
| `sample_id` | Metadata / Grouping | Arbitrary unique row identifier (TM000001+); non-generalizable metadata |
| `source_row` | Metadata / Grouping | Original row pointer in raw 200k Kolar precision agriculture dataset; provenance metadata |
| `source_timestamp` | Metadata / Grouping | Logger timestamp string; temporal sequence and calendar-month leakage |

## 5. Group Separation & Calendar-Month Protocol

The tomato sensor partition enforces strict calendar-month holdouts to eliminate autocorrelation leakage:
- **Group Key Construction:** `source_timestamp.str[:7]` (YYYY-MM calendar month)
- **Training Partition:** 29943 rows across 89 whole calendar months
- **Validation Partition:** 5377 rows across 16 whole calendar months (held out from train pool)
- **Test Partition (Untouched):** 8830 rows across 27 whole calendar months
- **Calendar Month Overlap Across Partitions:** `0` (0% overlap verified)
- **Sample ID Overlap Across Partitions:** `0` (0% overlap verified)
- **Class Representation:** 10 active diagnostic classes (Normal, Water_Temp_High, EC_High/Low, pH_Imbalance, Humidity_High/Low, N/P/K_Deficiency).

## 6. Mandatory Limitations & Disclosures

> [!IMPORTANT]
> Per Section 18.2 and Overview §15.7, all models and reports must prominently disclose the following caveats:

- **Experimental Status:**  Models are marked experimental=true in ModelMetadata until live-sensor telemetry is verified against campus greenhouse hardware (Overview §15.7).
- **Proxy-Transformed & Simulated Features:**  The underlying Kolar tomato dataset originates from open-field precision agriculture with simulated noise. Nutrient solution features (water_temperature_C, pH, EC, N, P, K) were generated using calibrated proxy transformations rather than direct hydroponic tank sensing.
- **Rule-Derived Diagnostic Targets:**  Labels are generated mathematically using multi-fault severity ranking over nine threshold rules (Water_Temp_High, EC_High/Low, pH_Imbalance, Humidity_High/Low, N/P/K_Deficiency).
- **Strict Zero-Leakage Protocol:**  Grouped 5-fold cross-validation by calendar month (89 disjoint months) prevents temporal autocorrelation leakage. Test split remained completely held-out and unread during training and hyperparameter search.
- **Excluded Features:**  n_rules_broken, source_row, source_timestamp, sample_id, and label_id were strictly excluded from model feature inputs.

## 7. Software & Runtime Environment

| Package / Environment | Version |
|---|---|
| numpy | `2.5.3` |
| platform | `Windows-11-10.0.26300-SP0` |
| python | `3.14.0` |
| scikit-learn | `1.9.1` |
