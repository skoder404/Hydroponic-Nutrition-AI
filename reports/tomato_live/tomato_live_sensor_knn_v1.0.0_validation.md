# Tomato Live-Compatible Sensor Model Validation Report: `tomato_live_sensor_knn_v1.0.0`

- **Generated:** 2026-10-27T16:18:31.493498+00:00
- **Crop:** Tomato (Live-Compatible Variant)
- **Model Family:** `knn`
- **Feature Set:** `live_compatible` (5 features: `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm`)
- **Classes:** 7 classes (`EC_High_Toxicity, EC_Low_Deficiency, Humidity_High_Mildew_Risk, Humidity_Low_Stress, Normal, Water_Temp_High, pH_Imbalance`)
- **Experimental Status:** `True` (campus hardware lacks Modbus N/P/K probes)
- **Cross-Validation Macro-F1 (Grouped Calendar Months):** 0.8846 (+/- 0.0062)
- **Validation Macro-F1:** 0.8916
- **Validation Accuracy:** 0.8939

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
  "n_neighbors": 5,
  "weights": "distance"
}
```

## Validation Classification Report

```text
                           precision    recall  f1-score   support

         EC_High_Toxicity     0.8455    0.9326    0.8869       534
        EC_Low_Deficiency     0.9130    0.9077    0.9103       520
Humidity_High_Mildew_Risk     0.8589    0.9250    0.8907       533
      Humidity_Low_Stress     0.8617    0.9463    0.9020       540
                   Normal     0.9186    0.9454    0.9318       549
          Water_Temp_High     0.9374    0.8914    0.9138       571
             pH_Imbalance     0.9474    0.7004    0.8054       514

                 accuracy                         0.8939      3761
                macro avg     0.8975    0.8927    0.8916      3761
             weighted avg     0.8976    0.8939    0.8923      3761

```

## Confusion Matrix (Labels: ['EC_High_Toxicity', 'EC_Low_Deficiency', 'Humidity_High_Mildew_Risk', 'Humidity_Low_Stress', 'Normal', 'Water_Temp_High', 'pH_Imbalance'])

```text
[[498   0  11  11   5   6   3]
 [  0 472  17  18   3   5   5]
 [ 10  13 493   0   5   3   9]
 [  9  10   0 511   4   5   1]
 [ 15   0   4   9 519   1   1]
 [ 21   2  15  17   6 509   1]
 [ 36  20  34  27  23  14 360]]
```

## Limitations & Governance

- Live-Compatible Variant: Trained exclusively on the 5 features confirmed available or proxy-derivable from the campus hardware gateway.
- Excluded Deficiencies: Nitrogen, phosphorus, and potassium sensors are disabled/deleted on the campus gateway (ADR 0008). Therefore, N_Deficiency, P_Deficiency, and K_Deficiency classes are excluded from this variant.
- Experimental Status: Models are marked experimental=true in ModelMetadata until direct live telemetry is validated across seasonal temperature variations.
- Thermal Water Proxy: Reservoir water temperature is inferred via ambient thermal dynamics (8.0 + 0.55 * air_temp) in the absence of immersion hardware.
- Zero-Leakage Protocol: Calendar-month group cross-validation guarantees zero temporal autocorrelation leakage.
