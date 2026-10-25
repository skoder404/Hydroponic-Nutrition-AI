# Tomato Sensor Dataset Technical Contract & Audit

- **Document Role:** Technical contract, feature specification, and audit findings for the hydroponic tomato sensor dataset.
- **Related Task:** T2-007
- **Source Files:**
  - `sensor_data/Tomato_Sensor-Readings/tomato_hydroponic_full.csv` (44,150 rows)
  - `sensor_data/Tomato_Sensor-Readings/tomato_hydroponic_train.csv` (35,320 rows)
  - `sensor_data/Tomato_Sensor-Readings/tomato_hydroponic_test.csv` (8,830 rows)
  - `sensor_data/Tomato_Sensor-Readings/build_tomato_dataset.py`
  - `sensor_data/Tomato_Sensor-Readings/README_tomato_dataset.md`
- **Output Manifest:** `processed_data/sensor/tomato/manifest.json`

---

## 1. Feature Contract & Schema

The tomato sensor classification models ingest strictly **8 physical features** in this exact column order:

| Index | Feature Name | Unit | Physical Target / Sensor | Proxy Derivation from Kolar |
|:---:|---|:---:|---|---|
| 0 | `air_temperature_C` | °C | Greenhouse ambient air temperature | `Ambient_Temp` (unchanged) |
| 1 | `relative_humidity_percent` | % | Greenhouse relative humidity | `Humidity + 15.0` (proxy shift) |
| 2 | `water_temperature_C` | °C | Nutrient solution root temperature | **Synthetic**: `8.0 + 0.55 * air + 0.003 * (solar - 700) + Normal(0, 0.6)` [seed 42, unclipped] |
| 3 | `ph` | pH | Solution acidity / alkalinity | `Soil_pH - 1.0` (proxy shift) |
| 4 | `ec_mS_cm` | mS/cm | Electrical conductivity (total ionic strength) | `EC_mS_cm * 1.4` (proxy scale) |
| 5 | `nitrogen_mg_L` | mg/L | Dissolved nitrate / ammoniacal nitrogen | `Nitrogen_Level * 4.0` (proxy scale) |
| 6 | `phosphorus_mg_L` | mg/L | Dissolved phosphate phosphorus | `Phosphorus_Level * 3.0` (proxy scale) |
| 7 | `potassium_mg_L` | mg/L | Dissolved potassium cation | `Potassium_Level * 6.0` (proxy scale) |

### Diagnostic & Non-Feature Columns
The following columns are present in the checked-in CSVs for provenance and auditing, and **MUST NOT** be fed into the model as inputs:
- `sample_id`: Unique identifier (e.g., `T00001` through `T44150`).
- `label`: Categorical target string (10 classes).
- `label_id`: Integer class index (0–9).
- `n_rules_broken`: Diagnostic rule violation count per sample.
- `source_row`: Index pointer to the original row in the upstream Kolar file.
- `source_timestamp`: Original timestamp string (`YYYY-MM-DD HH:MM:SS`) used strictly for monthly split partitioning.

---

## 2. Action Bands, Fault Thresholds & Labeling Mode

Hydroponic tomato reference bands are derived from Ohio State CFAES, UA-CEA, and Oklahoma State Extension guidelines:

| Fault Label | Rule Trigger Threshold | Reference / Rationale |
|---|---|---|
| `Normal` | No rules violated | All parameters within healthy hydroponic tolerance bands |
| `Water_Temp_High` | `water_temperature_C > 25.0 °C` | Root respiration stress & pythium risk (Hort Americas) |
| `EC_High_Toxicity` | `ec_mS_cm > 4.0 mS/cm` | Osmotic stress and leaf burn (Oklahoma State upper bound) |
| `EC_Low_Deficiency` | `ec_mS_cm < 2.0 mS/cm` | Insufficient nutrient ion concentration (Oklahoma State lower bound) |
| `pH_Imbalance` | `ph < 6.0` or `ph > 6.5` | CFAES / UA-CEA optimum drip range (5.5–6.5; 6.0–6.5 core) |
| `Humidity_High_Mildew_Risk` | `relative_humidity_percent > 85.0 %` | Stagnant air and fungal pathogen risk (Baptista et al.) |
| `Humidity_Low_Stress` | `relative_humidity_percent < 50.0 %` | Excessive vapor pressure deficit (VPD) and desiccation |
| `N_Deficiency` | `nitrogen_mg_L < 90.0 mg/L` | Ohio State CFAES tomato solution floor (90–190 mg/L) |
| `P_Deficiency` | `phosphorus_mg_L < 35.0 mg/L` | CFAES standard target ~47 mg/L (floor 35 mg/L) |
| `K_Deficiency` | `potassium_mg_L < 144.0 mg/L` | CFAES tomato solution floor (144–350 mg/L) |

### Labeling Mode: `primary` Confirmed
The checked-in dataset contains **10 classes**, confirming that the generator ran with default `--mode primary`.
- Under `primary` mode, a row violating multiple rules is assigned the single class of its **most severe fault**, defined as:
  $$\text{severity} = \frac{\text{deviation past threshold}}{\text{max deviation in dataset for that rule}}$$
- In contrast, `--mode single` would have added an 11th class (`Multiple_Faults`), which is **absent** in all checked-in CSVs.

---

## 3. Dataset Splits & Monthly Partitioning

To avoid autocorrelation leakage in continuous time-series data, the split is grouped by **whole calendar months**:

- **Total Calendar Months:** 132 unique months (spanning 2015-01 through 2025-12).
- **Test Months (27 months, ~20.45% of months):**
  `2015-02`, `2015-05`, `2016-01`, `2016-04`, `2016-06`, `2016-09`, `2017-02`, `2017-04`, `2017-11`, `2018-09`, `2018-11`, `2019-02`, `2019-04`, `2020-02`, `2020-04`, `2020-08`, `2020-11`, `2021-07`, `2021-08`, `2022-04`, `2022-08`, `2023-10`, `2024-09`, `2024-10`, `2025-01`, `2025-09`, `2025-10`.
- **Train Months:** 105 calendar months.
- **Leakage Check:** Exactly **zero calendar months overlap** between train and test partitions.
- **Validation Partition:** None in the initial split (validation folds are formed via cross-validation within `train`).

### Sample & Class Counts (Audited from Disk)
Classes are strictly balanced by equal random sampling per partition:
- **Full Dataset (`full.csv`):** 44,150 rows (exactly 4,415 rows per class × 10 classes).
- **Train Split (`train.csv`):** 35,320 rows (exactly 3,532 rows per class × 10 classes; 80.0%).
- **Test Split (`test.csv`):** 8,830 rows (exactly 883 rows per class × 10 classes; 20.0%).

---

## 4. Discrepancies & Resolutions

1. **Row Count vs. Overview:**
   - *Overview (§15)* cites 44,141 rows.
   - *README* estimated ~33,850 rows based on a preliminary 200k mock.
   - *Actual on Disk:* **44,150 rows** (4,415 per class × 10 classes).
2. **Number of Classes:**
   - Script documentation mentions 11 potential classes under `--mode single`.
   - The checked-in dataset contains exactly **10 classes** (`Multiple_Faults` is omitted), confirming `--mode primary`.

---

## 5. Critical Caveats & Limitations

As highlighted in the upstream README and project governance rules:

1. **Simulated Origin of Kolar Data:**
   - The upstream Kolar file (`kolar_tomato_precision_agri_dataset.csv`) **looks simulated, not real IoT sensor data**.
   - Variables are uniformly and independently distributed with flat quartiles (e.g., pH 6.2–8.5, EC 0.5–3.0, N 10–50), timestamps have no natural diurnal/seasonal curves, and `Nutrient_Lockout_Prob` is a direct synthetic function of pH.
   - It must **never** be cited as real-world field-validated hydroponic IoT data.
2. **Synthetic Water Temperature:**
   - Water temperature does not exist in the raw Kolar dataset and is synthetically modeled from ambient temperature and solar radiation with Gaussian noise.
3. **Arbitrary Proxy Transforms:**
   - The shifts and multipliers applied to humidity, pH, EC, N, P, and K are engineering heuristic proxies to align land-based numbers with hydroponic nutrient solution ranges.
4. **Reproducibility Dependency (⚠️ UNCLEAR):**
   - The raw 200,000-row Kolar CSV is not retained in git due to file-size constraints.
   - Because `primary` mode normalizes deviations against the global dataset maximums (`scales`), re-running `build_tomato_dataset.py` from scratch requires the original upstream file. The checked-in CSVs (`full`, `train`, `test`) remain the authoritative frozen source of truth.
