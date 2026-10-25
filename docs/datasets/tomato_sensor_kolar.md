# Dataset Provenance: Kolar Land-Based to Hydroponic Tomato Sensor Dataset

- **Document Version:** 1.0.0
- **Author:** Yashwant
- **Related Tasks:** T2-004, T1-001, T4-001
- **Target Crop:** Tomato (*Solanum lycopersicum*)
- **Modality:** Nutrient Solution & Environmental Time-Series
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** Autonomous Tomato Cultivation Dataset in Kolar (Kaggle)
- **Source URL:** https://www.kaggle.com/datasets/eshwaranece91/autonomous-tomato-cultivation-dataset-in-kolar
- **Authors / Contributor:** Eshwaran (Kaggle dataset author)
- **Source Version / Date Downloaded:** Published on Kaggle / Ingested October 2026
- **Original Filename(s):** `kolar_tomato_precision_agri_dataset.csv` (200,000 rows, 29 columns; excluded from git commits per repository governance rules)
- **Current Repository Location:** `sensor_data/Tomato_Sensor-Readings/`
  - Preparation script: `sensor_data/Tomato_Sensor-Readings/build_tomato_dataset.py`
  - Dataset documentation: `sensor_data/Tomato_Sensor-Readings/README_tomato_dataset.md`
  - Output files: `tomato_hydroponic_full.csv`, `tomato_hydroponic_train.csv`, `tomato_hydroponic_test.csv`

---

## 2. License, Terms & Redistribution Status

- **License Type:** `unverified` (standard Kaggle community upload; explicit license terms unrecorded in project overview)
- **Original Terms / Rights Holder:** `unverified`
- **Redistribution Status:** Publicly available on Kaggle; raw 200,000-row source CSV remains local, while prepared subsets and the reproducible transformation script are committed in the repository.
- **Commercial / Non-Commercial Constraints:** `unverified` (restricted to academic research and educational decision support).

---

## 3. Sample Counts & Volume (Audited from Disk)

Counts audited directly from disk inspection of `sensor_data/Tomato_Sensor-Readings/`:

| Metric Stage | Sample Count (Rows) | Columns | Disk Size (Bytes) | Notes / Filtering Rationale |
|---|:---:|:---:|:---:|---|
| **Original Upstream Raw (Kolar)** | 200,000 rows | 29 | ~30 MB | Full land-based precision agriculture source CSV |
| **Filtered / Dropped Features** | — | 15 dropped | — | Soil moisture, transport properties, solute flux, 15 RSSI columns dropped |
| **Prepared: Full (`tomato_hydroponic_full.csv`)** | **44,150 rows** | **14** | **4,333,446 B** | Balanced 10-class dataset (4,415 rows per class in `primary` fault mode) |
| **Prepared Split: Train (`tomato_hydroponic_train.csv`)** | **35,320 rows** | **14** | **3,467,475 B** | 80% training partition (3,532 rows per class), partitioned by calendar month |
| **Prepared Split: Test (`tomato_hydroponic_test.csv`)** | **8,830 rows** | **14** | **866,161 B** | 20% test partition (883 rows per class), partitioned by calendar month |

> **Audit Verification:** Null counts across all 14 columns in `full.csv`, `train.csv`, and `test.csv` are exactly 0.

---

## 4. Class Labels & Distribution (Active Files on Disk)

Labels are assigned in `primary` fault mode: a sample that breaks multiple rules is assigned the label of its most severe normalized deviation past threshold.

| Class Label | Label ID | Full Count | Train Count | Test Count | Prevalence (%) | Diagnostic Threshold Definition |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `Normal` | 0 | 4,415 | 3,532 | 883 | 10.00% | No rule broken (all parameters within healthy hydroponic bands) |
| `Water_Temp_High` | 1 | 4,415 | 3,532 | 883 | 10.00% | Water temperature > 25.0 °C (Hort Americas optimum root ceiling) |
| `EC_High_Toxicity` | 2 | 4,415 | 3,532 | 883 | 10.00% | Electrical conductivity > 4.0 mS/cm (Oklahoma State upper bound) |
| `EC_Low_Deficiency` | 3 | 4,415 | 3,532 | 883 | 10.00% | Electrical conductivity < 2.0 mS/cm (Oklahoma State lower bound) |
| `pH_Imbalance` | 4 | 4,415 | 3,532 | 883 | 10.00% | pH < 6.0 or pH > 6.5 (OSU CFAES drip hydroponic range) |
| `Humidity_High_Mildew_Risk` | 5 | 4,415 | 3,532 | 883 | 10.00% | Relative humidity > 85.0% (leaf mold risk threshold) |
| `Humidity_Low_Stress` | 6 | 4,415 | 3,532 | 883 | 10.00% | Relative humidity < 50.0% |
| `N_Deficiency` | 7 | 4,415 | 3,532 | 883 | 10.00% | Nitrogen < 90 mg/L (Ohio State CFAES / UA-CEA solution range) |
| `P_Deficiency` | 8 | 4,415 | 3,532 | 883 | 10.00% | Phosphorus < 35 mg/L (Ohio State CFAES / UA-CEA solution range) |
| `K_Deficiency` | 9 | 4,415 | 3,532 | 883 | 10.00% | Potassium < 144 mg/L (Ohio State CFAES / UA-CEA solution range) |
| **Total** | | **44,150** | **35,320** | **8,830** | **100.00%** | |

---

## 5. Soil-to-Hydroponic Proxy Transformations & Derived Values

As documented in the dataset README (`sensor_data/Tomato_Sensor-Readings/README_tomato_dataset.md`), the original Kolar land-based soil measurements were adapted to hydroponic nutrient solution ranges using explicit proxy transformations:

| Prepared Feature | Source Feature in Kolar | Transformation Formula | Qualification & Provenance |
|---|---|---|---|
| `air_temperature_C` | `Ambient_Temp` | Unchanged | Retained direct ambient temperature |
| `relative_humidity_percent` | `Humidity` | `Humidity + 15` | Proxy shift to match greenhouse humidity levels |
| `ph` | `Soil_pH` | `Soil_pH - 1.0` | Proxy shift to map alkaline soil pH to acidic hydroponic target (5.5–6.5) |
| `ec_mS_cm` | `EC_mS_cm` | `EC_mS_cm * 1.4` | Proxy scaling to map soil EC to tomato nutrient solution conductivity |
| `nitrogen_mg_L` | `Nitrogen_Level` | `Nitrogen_Level * 4` | Proxy scaling to align with dissolved nitrogen solution mg/L |
| `phosphorus_mg_L` | `Phosphorus_Level` | `Phosphorus_Level * 3` | Proxy scaling to align with dissolved phosphorus solution mg/L |
| `potassium_mg_L` | `Potassium_Level` | `Potassium_Level * 6` | Proxy scaling to align with dissolved potassium solution mg/L |
| `water_temperature_C` | `Ambient_Temp`, `Solar_Radiation` | `8 + 0.55*Ambient_Temp + 0.003*(Solar_Radiation - 700) + N(0, 0.6)` | **Fully synthetic feature**: modeled thermal equilibrium with diurnal noise |

> **Attributed README Observation:** The README explicitly notes: *"The soil-to-hydroponic mappings are arbitrary proxy transforms chosen so the ranges overlap hydroponic norms. They are not physical conversions."*

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** Grouped holdout by calendar month (80% train, 20% test).
- **Split Rationale:** No source row and no calendar month appears in both training and test partitions.
- **Traceability:** Every row in `full.csv`, `train.csv`, and `test.csv` maintains `source_row` and `source_timestamp` pointing directly to its upstream position in `kolar_tomato_precision_agri_dataset.csv`.

---

## 7. Observed Empirical Characteristics (Simulation Observation)

Per the explicit requirement of task T2-004, the observations documented in `sensor_data/Tomato_Sensor-Readings/README_tomato_dataset.md` and Project Overview (§15.3) are recorded here:

- **Attributed README Observation on Simulation:**  
  The tomato dataset README states as an observation:
  > *"The Kolar file looks simulated, not real field data. Every variable is uniformly distributed and independent (e.g. pH 6.2-8.5, EC 0.5-3.0, N 10-50 with flat quartiles), timestamps are random with no daily or seasonal pattern, and `Nutrient_Lockout_Prob` is a function of pH. Do not describe it as real IoT sensor data."*
- **Qualification as Observation (Not Fact):**  
  In accordance with repository scientific integrity standards, this characteristic is recorded as an **empirical observation of data properties** rather than as a proven statement of fact regarding the upstream author's generation methodology. The statistical uniformity, lack of diurnal periodicity in solar/temperature fields, and independence across physical parameters strongly indicate synthetic generation, but the external author's exact pipeline remains unverified.

---

## 8. Column / Field Taxonomy Table (14 Prepared Columns)

All 14 columns present in `tomato_hydroponic_full.csv` with their provenance classification and unit verification status:

| Column Name | Data Type | Provenance Tag | Unit / Value Range | Description & Verification Status |
|---|---|:---:|---|---|
| `sample_id` | `integer` | `derived` | 0 – 44,149 | Sequential unique sample identifier |
| `air_temperature_C` | `float` | `adjusted` | 18.0 – 38.0 | Ambient temperature from Kolar; original unit likely °C (`unverified`) |
| `relative_humidity_percent` | `float` | `adjusted` | 45.0 – 90.0 % | Proxy adjusted relative humidity (`Humidity + 15`); original unit `unverified` |
| `water_temperature_C` | `float` | `synthetic` | 16.5 – 30.5 °C | Synthetically generated solution temperature; formula in §5 |
| `ph` | `float` | `adjusted` | 5.2 – 7.5 | Proxy adjusted solution pH (`Soil_pH - 1.0`); original unit `unverified` |
| `ec_mS_cm` | `float` | `adjusted` | 0.7 – 4.2 | Proxy adjusted electrical conductivity; original unit likely mS/cm (`unverified`) |
| `nitrogen_mg_L` | `float` | `adjusted` | 40.0 – 200.0 | Proxy scaled nitrogen level; original units `unverified` |
| `phosphorus_mg_L` | `float` | `adjusted` | 15.0 – 75.0 | Proxy scaled phosphorus level; original units `unverified` |
| `potassium_mg_L` | `float` | `adjusted` | 90.0 – 360.0 | Proxy scaled potassium level; original units `unverified` |
| `label` | `string` | `rule-generated` | 10 categories | Target multi-class fault classification assigned in `primary` mode |
| `label_id` | `integer` | `rule-generated` | 0 – 9 | Numeric integer encoding of `label` |
| `n_rules_broken` | `integer` | `derived` | 0 – 6 | Diagnostic counter of simultaneous thresholds violated |
| `source_row` | `integer` | `derived` | 0 – 199,999 | Line index in `kolar_tomato_precision_agri_dataset.csv` |
| `source_timestamp` | `datetime` | `measured` | ISO 8601 | Original timestamp string from Kolar dataset (2015–2025) |

### Upstream Kolar Raw Columns Dropped (Units Requiring Verification)
- `Solar_Radiation`: 400 – 1000; unit likely W/m² (`unverified`)
- `Soil_Moisture_VWC`: 0.10 – 0.35; volumetric water content (`unverified`)
- `Pore_Water_Velocity_v`: 0.01 – 0.05; unit `unverified`
- `Dispersion_Coeff_D`: 0.005 – 0.02; unit `unverified`
- `Nutrient_Lockout_Prob`: 0.1 – 0.7; dimensionless probability (`unverified`)
- `Solute_Flux_J`: 0.11 – 2.5; unit `unverified`
- `Sensor_Node_1_Signal_RSSI` to `15`: -110 to -40; unit likely dBm (`unverified`)

---

## 9. Known Limitations & Unsupported Interpretations

- **Simulation & Proxy Validity:** The dataset incorporates proxy transformations and synthetic water temperatures. It is intended for evaluating multi-class ML classification algorithms, **not for agricultural decision making in commercial farms**.
- **Live Campus Incompatibility Caution (PRD D11):** The campus polyhouse server does not supply individual nitrogen, phosphorus, and potassium (NPK) measurements. Therefore, models trained on this 8-feature tomato dataset cannot run live against the campus server without either training a live-compatible variant (omitting N/P/K) or providing manual NPK entry.
- **Unpaired with Tomato Leaf Imagery:** Sourced independently from Mendeley/Kaggle leaf photographs; cross-modal end-to-end learning is strictly forbidden.
- **Rule Recovery vs. Diagnostic Intelligence:** High model classification accuracy reflects the ability of algorithms (such as Random Forest or Decision Trees) to recover deterministic threshold rules, not predictive biological acuity.
