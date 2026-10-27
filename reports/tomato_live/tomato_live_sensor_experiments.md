# Tomato Live-Compatible Sensor Experiment Report (T4-020)

- **Generated:** 2026-10-27T16:18:47.902607+00:00
- **Crop:** Tomato (`live_compatible` variant)
- **Feature Set:** `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm` (5 features)
- **Excluded Features:** `nitrogen_mg_L`, `phosphorus_mg_L`, `potassium_mg_L` (unmetered / deleted in gateway hardware)
- **Target Classes (7):** `EC_High_Toxicity, EC_Low_Deficiency, Humidity_High_Mildew_Risk, Humidity_Low_Stress, Normal, Water_Temp_High, pH_Imbalance`
- **Excluded Deficiencies:** `N_Deficiency`, `P_Deficiency`, `K_Deficiency`
- **Experimental Status:** `True` (Overview §15.7 & PRD D11)
- **Best Candidate Family:** `random_forest` (Validation Macro-F1: **0.9829**)

## Feature Availability Table (Campus Server Probe vs Model Features)

| Model Feature | Server Field | Server Unit | Model Unit | Conversion / Mapping | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `air_temperature_C` | `temp_avg` | °C | degC | Direct 1:1 float (fallback: temperature1/temperature2) | **Available** | Verified live Modbus slave 5 (~41.5 °C). |
| `relative_humidity_percent` | `humidity_avg` | % | % | Direct 1:1 float (fallback: humidity1/humidity2) | **Available** | Verified live Modbus slave 5 (~34.7 %). |
| `water_temperature_C` | `None (unmetered)` | N/A | degC | Thermal proxy: 8.0 + 0.55 * air_temp | **Derived Proxy** | Physical campus tank lacks immersion probe; modeled from ambient thermal equilibrium. |
| `ph` | `ph` | pH | pH | Direct 1:1 float | **Available** | Verified live Modbus slave 1 (~7.66 pH). |
| `ec_mS_cm` | `ec` | µS/cm | mS/cm | Scaled: ec / 1000.0 (if raw > 10.0) | **Available** | Verified live Modbus slave 6 (~36.8 µS/cm -> 0.0368 mS/cm). |
| `nitrogen_mg_L` | `nitrogen` | mg/L | mg/L | N/A — Excluded from live variant | **Missing / Disabled** | Hardware sensor deleted in gateway config._deleted_keys (ADR 0008). |
| `phosphorus_mg_L` | `phosphorus` | mg/L | mg/L | N/A — Excluded from live variant | **Missing / Disabled** | Hardware sensor deleted in gateway config._deleted_keys (ADR 0008). |
| `potassium_mg_L` | `potassium` | mg/L | mg/L | N/A — Excluded from live variant | **Missing / Disabled** | Hardware sensor deleted in gateway config._deleted_keys (ADR 0008). |

## Model Comparison Summary

| Model Family | Grouped CV Macro-F1 | Validation Macro-F1 | Validation Accuracy | Best Hyperparameters |
|---|---|---|---|---|
| `random_forest` | 0.9835 (+/- 0.0022) | 0.9829 | 0.9832 | `{"class_weight": null, "max_depth": null, "min_samples_leaf": 1, "n_estimators": 100}` |
| `decision_tree` | 0.9774 (+/- 0.0025) | 0.9796 | 0.9798 | `{"criterion": "gini", "max_depth": null, "min_samples_leaf": 1}` |
| `svm` | 0.9497 (+/- 0.0045) | 0.9498 | 0.9508 | `{"C": 2.5, "gamma": "scale", "kernel": "rbf"}` |
| `knn` | 0.8846 (+/- 0.0062) | 0.8916 | 0.8939 | `{"n_neighbors": 5, "weights": "distance"}` |
| `logistic_regression` | 0.8332 (+/- 0.0074) | 0.8379 | 0.8492 | `{"C": 5.0, "class_weight": "balanced"}` |

## Comparison with Full 8-Feature Tomato Model

| Dimension | Full Model (T4-019) | Live Variant (T4-020) | Rationale |
|---|---|---|---|
| **Feature Count** | 8 features | 5 features | Campus gateway lacks physical N/P/K selective ion electrodes |
| **Class Count** | 10 classes | 7 classes | Classes depending exclusively on unmetered nutrients are removed |
| **Excluded Classes** | None | N, P, K Deficiencies | Avoids false-confidence predictions when inputs cannot be observed |
| **Experimental** | `True` | `True` | Both remain experimental until validated against ongoing live polyhouse cycles |
| **Metadata Tag** | Standard | `feature_set: live_compatible` | Enforces schema validation during live inference |

## Limitations & Governance

- Live-Compatible Variant: Trained exclusively on the 5 features confirmed available or proxy-derivable from the campus hardware gateway.
- Excluded Deficiencies: Nitrogen, phosphorus, and potassium sensors are disabled/deleted on the campus gateway (ADR 0008). Therefore, N_Deficiency, P_Deficiency, and K_Deficiency classes are excluded from this variant.
- Experimental Status: Models are marked experimental=true in ModelMetadata until direct live telemetry is validated across seasonal temperature variations.
- Thermal Water Proxy: Reservoir water temperature is inferred via ambient thermal dynamics (8.0 + 0.55 * air_temp) in the absence of immersion hardware.
- Zero-Leakage Protocol: Calendar-month group cross-validation guarantees zero temporal autocorrelation leakage.
