# Tomato Live-Compatible Sensor Model Validation Report: `tomato_live_sensor_svm_v1.0.0`

- **Generated:** 2026-10-27T16:18:22.982673+00:00
- **Crop:** Tomato (Live-Compatible Variant)
- **Model Family:** `svm`
- **Feature Set:** `live_compatible` (5 features: `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm`)
- **Classes:** 7 classes (`EC_High_Toxicity, EC_Low_Deficiency, Humidity_High_Mildew_Risk, Humidity_Low_Stress, Normal, Water_Temp_High, pH_Imbalance`)
- **Experimental Status:** `True` (campus hardware lacks Modbus N/P/K probes)
- **Cross-Validation Macro-F1 (Grouped Calendar Months):** 0.9497 (+/- 0.0045)
- **Validation Macro-F1:** 0.9498
- **Validation Accuracy:** 0.9508

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
  "C": 2.5,
  "gamma": "scale",
  "kernel": "rbf"
}
```

## Validation Classification Report

```text
                           precision    recall  f1-score   support

         EC_High_Toxicity     0.9057    0.9888    0.9454       534
        EC_Low_Deficiency     0.9723    0.9442    0.9580       520
Humidity_High_Mildew_Risk     0.9215    0.9906    0.9548       533
      Humidity_Low_Stress     0.9343    0.9741    0.9538       540
                   Normal     0.9608    0.9818    0.9712       549
          Water_Temp_High     0.9783    0.9457    0.9617       571
             pH_Imbalance     1.0000    0.8249    0.9041       514

                 accuracy                         0.9508      3761
                macro avg     0.9532    0.9500    0.9498      3761
             weighted avg     0.9532    0.9508    0.9503      3761

```

## Confusion Matrix (Labels: ['EC_High_Toxicity', 'EC_Low_Deficiency', 'Humidity_High_Mildew_Risk', 'Humidity_Low_Stress', 'Normal', 'Water_Temp_High', 'pH_Imbalance'])

```text
[[528   0   2   4   0   0   0]
 [  0 491  15  12   0   2   0]
 [  5   0 528   0   0   0   0]
 [ 10   2   0 526   1   1   0]
 [  8   0   2   0 539   0   0]
 [ 13   0   6   8   4 540   0]
 [ 19  12  20  13  17   9 424]]
```

## Limitations & Governance

- Live-Compatible Variant: Trained exclusively on the 5 features confirmed available or proxy-derivable from the campus hardware gateway.
- Excluded Deficiencies: Nitrogen, phosphorus, and potassium sensors are disabled/deleted on the campus gateway (ADR 0008). Therefore, N_Deficiency, P_Deficiency, and K_Deficiency classes are excluded from this variant.
- Experimental Status: Models are marked experimental=true in ModelMetadata until direct live telemetry is validated across seasonal temperature variations.
- Thermal Water Proxy: Reservoir water temperature is inferred via ambient thermal dynamics (8.0 + 0.55 * air_temp) in the absence of immersion hardware.
- Zero-Leakage Protocol: Calendar-month group cross-validation guarantees zero temporal autocorrelation leakage.
