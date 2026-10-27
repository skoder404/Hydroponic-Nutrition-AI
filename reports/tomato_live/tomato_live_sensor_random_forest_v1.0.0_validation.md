# Tomato Live-Compatible Sensor Model Validation Report: `tomato_live_sensor_random_forest_v1.0.0`

- **Generated:** 2026-10-27T16:12:53.701501+00:00
- **Crop:** Tomato (Live-Compatible Variant)
- **Model Family:** `random_forest`
- **Feature Set:** `live_compatible` (5 features: `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm`)
- **Classes:** 7 classes (`EC_High_Toxicity, EC_Low_Deficiency, Humidity_High_Mildew_Risk, Humidity_Low_Stress, Normal, Water_Temp_High, pH_Imbalance`)
- **Experimental Status:** `True` (campus hardware lacks Modbus N/P/K probes)
- **Cross-Validation Macro-F1 (Grouped Calendar Months):** 0.9835 (+/- 0.0022)
- **Validation Macro-F1:** 0.9829
- **Validation Accuracy:** 0.9832

## Feature Availability Table (ADR 0008 vs Model Schema)

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

## Hyperparameters

```json
{
  "class_weight": null,
  "max_depth": null,
  "min_samples_leaf": 1,
  "n_estimators": 100
}
```

## Validation Classification Report

```text
                           precision    recall  f1-score   support

         EC_High_Toxicity     0.9674    1.0000    0.9834       534
        EC_Low_Deficiency     0.9790    0.9846    0.9818       520
Humidity_High_Mildew_Risk     0.9724    0.9906    0.9814       533
      Humidity_Low_Stress     0.9817    0.9907    0.9862       540
                   Normal     1.0000    1.0000    1.0000       549
          Water_Temp_High     0.9843    0.9860    0.9851       571
             pH_Imbalance     1.0000    0.9280    0.9627       514

                 accuracy                         0.9832      3761
                macro avg     0.9835    0.9829    0.9829      3761
             weighted avg     0.9835    0.9832    0.9831      3761

```

## Confusion Matrix (Labels: ['EC_High_Toxicity', 'EC_Low_Deficiency', 'Humidity_High_Mildew_Risk', 'Humidity_Low_Stress', 'Normal', 'Water_Temp_High', 'pH_Imbalance'])

```text
[[534   0   0   0   0   0   0]
 [  0 512   4   1   0   3   0]
 [  5   0 528   0   0   0   0]
 [  3   1   0 535   0   1   0]
 [  0   0   0   0 549   0   0]
 [  3   2   2   1   0 563   0]
 [  7   8   9   8   0   5 477]]
```

## Limitations & Governance

- Live-Compatible Variant: Trained exclusively on the 5 features confirmed available or proxy-derivable from the campus hardware gateway.
- Excluded Deficiencies: Nitrogen, phosphorus, and potassium sensors are disabled/deleted on the campus gateway (ADR 0008). Therefore, N_Deficiency, P_Deficiency, and K_Deficiency classes are excluded from this variant.
- Experimental Status: Models are marked experimental=true in ModelMetadata until direct live telemetry is validated across seasonal temperature variations.
- Thermal Water Proxy: Reservoir water temperature is inferred via ambient thermal dynamics (8.0 + 0.55 * air_temp) in the absence of immersion hardware.
- Zero-Leakage Protocol: Calendar-month group cross-validation guarantees zero temporal autocorrelation leakage.
