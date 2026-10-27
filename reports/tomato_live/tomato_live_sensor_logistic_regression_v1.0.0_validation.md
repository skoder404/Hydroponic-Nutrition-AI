# Tomato Live-Compatible Sensor Model Validation Report: `tomato_live_sensor_logistic_regression_v1.0.0`

- **Generated:** 2026-10-27T16:18:47.815514+00:00
- **Crop:** Tomato (Live-Compatible Variant)
- **Model Family:** `logistic_regression`
- **Feature Set:** `live_compatible` (5 features: `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm`)
- **Classes:** 7 classes (`EC_High_Toxicity, EC_Low_Deficiency, Humidity_High_Mildew_Risk, Humidity_Low_Stress, Normal, Water_Temp_High, pH_Imbalance`)
- **Experimental Status:** `True` (campus hardware lacks Modbus N/P/K probes)
- **Cross-Validation Macro-F1 (Grouped Calendar Months):** 0.8332 (+/- 0.0074)
- **Validation Macro-F1:** 0.8379
- **Validation Accuracy:** 0.8492

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
  "C": 5.0,
  "class_weight": "balanced"
}
```

## Validation Classification Report

```text
                           precision    recall  f1-score   support

         EC_High_Toxicity     0.9604    1.0000    0.9798       534
        EC_Low_Deficiency     0.9028    0.9288    0.9156       520
Humidity_High_Mildew_Risk     0.9312    0.9906    0.9600       533
      Humidity_Low_Stress     0.9537    0.9926    0.9728       540
                   Normal     0.7407    0.7286    0.7346       549
          Water_Temp_High     0.8597    0.9229    0.8902       571
             pH_Imbalance     0.4794    0.3619    0.4124       514

                 accuracy                         0.8492      3761
                macro avg     0.8326    0.8465    0.8379      3761
             weighted avg     0.8343    0.8492    0.8402      3761

```

## Confusion Matrix (Labels: ['EC_High_Toxicity', 'EC_Low_Deficiency', 'Humidity_High_Mildew_Risk', 'Humidity_Low_Stress', 'Normal', 'Water_Temp_High', 'pH_Imbalance'])

```text
[[534   0   0   0   0   0   0]
 [  0 483   8   6   2   5  16]
 [  3   0 528   0   0   0   2]
 [  1   0   0 536   1   0   2]
 [  1   0   0   0 400   0 148]
 [  4   0   2   3   1 527  34]
 [ 13  52  29  17 136  81 186]]
```

## Limitations & Governance

- Live-Compatible Variant: Trained exclusively on the 5 features confirmed available or proxy-derivable from the campus hardware gateway.
- Excluded Deficiencies: Nitrogen, phosphorus, and potassium sensors are disabled/deleted on the campus gateway (ADR 0008). Therefore, N_Deficiency, P_Deficiency, and K_Deficiency classes are excluded from this variant.
- Experimental Status: Models are marked experimental=true in ModelMetadata until direct live telemetry is validated across seasonal temperature variations.
- Thermal Water Proxy: Reservoir water temperature is inferred via ambient thermal dynamics (8.0 + 0.55 * air_temp) in the absence of immersion hardware.
- Zero-Leakage Protocol: Calendar-month group cross-validation guarantees zero temporal autocorrelation leakage.
