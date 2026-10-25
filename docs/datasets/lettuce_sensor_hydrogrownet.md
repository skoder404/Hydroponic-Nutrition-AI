# Dataset Provenance: HydroGrowNet Batavia Lettuce Sensor Dataset

- **Document Version:** 1.0.0
- **Author:** Yashwant
- **Related Tasks:** T2-002, T1-001, T4-001
- **Target Crop:** Batavia Lettuce (*Lactuca sativa* L.)
- **Modality:** Nutrient Solution & Environmental Time-Series
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** HydroGrowNet of Batavia Dataset (Mendeley Data)
- **Source URL:** https://doi.org/10.17632/g6cm3v3wdp.5
- **DOI:** 10.17632/g6cm3v3wdp.5
- **Authors / Institution:** Arab Academy for Science, Technology and Maritime Transport (AASTMT)
- **Facility / Context:** Nutrient Film Technique (NFT) hydroponic research greenhouse, three 30-day experimental runs conducted between 10 March 2024 and 19 June 2024.
- **Source Version / Date Downloaded:** Version 5 (2024) / Ingested October 2026
- **Original Filename(s):** Upstream workbooks `EXP_1.xlsx`, `EXP_2.xlsx`, `EXP_3.xlsx` (retained externally to prevent multi-megabyte binary bloat in git)
- **Current Repository Location:** `sensor_data/Lettuce_Sensor-Readings/`
  - Processing script: `sensor_data/Lettuce_Sensor-Readings/build_lettuce_dataset`
  - Dataset documentation: `sensor_data/Lettuce_Sensor-Readings/README_lettuce_dataset`
  - Prepared CSVs: `hydroponic_lettuce_full.csv`, `hydroponic_lettuce_train.csv`, `hydroponic_lettuce_test.csv`

---

## 2. License, Terms & Redistribution Status

- **License Type:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original Terms / Rights Holder:** Arab Academy for Science, Technology and Maritime Transport; authors of Mendeley dataset `g6cm3v3wdp.5`
- **Redistribution Status:** Publicly accessible with mandatory academic citation. Derived and cleaned CSV files and the preparation script are stored in this repository; upstream raw multi-sheet Excel files remain in the external DOI repository.
- **Commercial / Non-Commercial Constraints:** `unverified` (assumed educational / academic research use per CC BY 4.0).

---

## 3. Sample Counts & Volume (Audited from Disk)

Counts audited directly from disk inspection of `sensor_data/Lettuce_Sensor-Readings/`:

| Metric Stage | Sample Count (Rows) | Columns | Size on Disk | Notes / Filtering Rationale |
|---|:---:|:---:|:---:|---|
| **Raw Portable Meter Readings** | 726 rows | 32 | 203,658 B | All valid twice-daily readings across 6 replicate tanks (R1–R3 × T1/T2) |
| **Filtered Sensor Faults** | 2 readings blanked | — | — | 1 pH reading of 2.0 and 1 EC reading of 26.7 blanked to NaN as sensor faults |
| **Prepared: Full (`hydroponic_lettuce_full.csv`)** | **726 rows** | **32** | 203,658 B | Every joined water reading with matching hourly air metrics and action labels |
| **Prepared Split: Train (`hydroponic_lettuce_train.csv`)** | **1,680 rows** | **32** | 471,075 B | EXP.1 + EXP.2, oversampled to 240 rows/class across 7 retained classes |
| **Prepared Split: Test (`hydroponic_lettuce_test.csv`)** | **162 rows** | **32** | 46,488 B | Held-out every 4th calendar day of each experiment; real distribution |
| **Hourly Air Logger Series** | 2,069 rows | — | — | Separate ambient logger series without water chemistry (not artificially inflated) |

> **Note on Ceiling:** 726 is the true physical ceiling for readings containing pH and EC. Artificially repeating twice-daily pH/EC against 2,069 hourly air readings would give ~12,000 synthetic rows with duplicated water data, which is rejected to avoid artificial inflation of model metrics.

---

## 4. Class Labels & Distribution (Audited from Disk)

Labels are rule-derived from action thresholds based on warm-climate lettuce culture:

### Action Bands
- **pH:** 5.5 – 7.0
- **EC:** 1.2 – 2.5 mS/cm
- **Air Temperature:** 18.0 – 30.0 °C
- **Relative Humidity:** 50.0 – 80.0 %
- **Water Temperature:** 18.0 – 26.0 °C

### Class Distribution on Disk

| Class Label | Full Count (`full.csv`) | Full (%) | Train Count (`train.csv`) | Test Count (`test.csv`) | Diagnostic Definition |
|---|:---:|:---:|:---:|:---:|---|
| `Normal` | 327 | 45.04% | 240 | 87 | All 5 variables within respective action bands |
| `Multiple_Stress` | 135 | 18.60% | 240 | 23 | Two or more environmental/solution variables out of band |
| `Humidity_Low` | 87 | 11.98% | 240 | 16 | Relative humidity < 50.0% (single out-of-band fault) |
| `WaterTemp_High` | 75 | 10.33% | 240 | 19 | Solution temperature > 26.0 °C |
| `EC_High` | 58 | 7.99% | 240 | 9 | Electrical conductivity > 2.5 mS/cm |
| `pH_High` | 25 | 3.44% | 240 | 6 | pH > 7.0 |
| `pH_Low` | 10 | 1.38% | 240 | 2 | pH < 5.5 |
| `WaterTemp_Low` | 8 | 1.10% | 0 *(dropped)* | 0 *(dropped)* | Solution temperature < 18.0 °C (too rare to train) |
| `EC_Low` | 1 | 0.14% | 0 *(dropped)* | 0 *(dropped)* | Electrical conductivity < 1.2 mS/cm (too rare to train) |
| **Total** | **726** | **100.00%** | **1,680** | **162** | |

---

## 5. Preprocessing & Transformation Pipeline

- **Script:** `sensor_data/Lettuce_Sensor-Readings/build_lettuce_dataset`
- **Corrections Applied:**
  1. **Date Transposition:** Portable meter sheet recorded day and month swapped (`2024-10-03` for 10 March 2024). Inverted back using experiment date windows.
  2. **Header Reordering:** Fixed inconsistent column order in EXP.2 where 14:00 block headers contradicted data.
  3. **Sensor Fault Blanking:** Physically impossible values (pH 2.0 and EC 26.7) blanked to `NaN` (never imputed with arbitrary replacement values).
  4. **Hourly Merging:** Portable readings matched to exact-hour air logger readings (690/726 exact matches; portable meter's internal ambient sensor filled the remaining 36).
- **Trend Feature Engineering:** Computed per-tank lagged features (`prev`, `delta`, `roll3`) for `ph`, `ec`, and `water_temp`.

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** Date-based calendar holdout (`every=4`). Every 4th calendar day of each experiment is held out as the test set.
- **Split Rationale:** Tanks in the same NFT greenhouse share ambient climate and schedule. Holding out entire calendar days prevents temporal and environmental leakage between training and testing.
- **Oversampling Policy:** Balancing (to 240 rows per class) is applied **exclusively to the training split**. The test split preserves the untouched real empirical class distribution.
- **Class Exclusion:** `WaterTemp_Low` (8 rows) and `EC_Low` (1 row) are dropped from train/test splits due to extreme rarity.

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

- **Synthetic Elements:** None. All base measurements originate from physical instruments.
- **Derived Features:**
  - Daily summary aggregates: `air_temp_day_mean`, `air_temp_day_min`, `air_temp_day_max`, `humidity_day_mean`, `co2_day_mean`, `hours_logged`
  - Per-tank dynamics: `ph_prev`, `ph_delta`, `ph_roll3`, `ec_prev`, `ec_delta`, `ec_roll3`, `water_temp_prev`, `water_temp_delta`, `water_temp_roll3`
  - Elapsed time: `day_index`
- **Rule-Generated Labels:** `label` and `label_next` assigned deterministically from multi-variable action threshold evaluation.

---

## 8. Column / Field Taxonomy Table (32 Columns)

Every column in `hydroponic_lettuce_full.csv` is tagged with its provenance classification:

| Column Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `experiment` | `string` | `measured` | `EXP_1`, `EXP_2`, `EXP_3` | Identifier of the 30-day experimental campaign |
| `date` | `date` | `adjusted` | `2024-03-10` to `2024-06-19` | Calendar date (day/month transposition corrected) |
| `day_index` | `integer` | `derived` | 1 – 30 | Days elapsed since experiment initiation |
| `slot` | `string` | `measured` | `10:00`, `14:00` | Scheduled portable meter collection slot |
| `timestamp` | `datetime` | `adjusted` | ISO 8601 | Joined date and slot timestamp |
| `replicate` | `string` | `measured` | `R1T1` to `R3T2` | Replicate tank identifier (Replicates 1–3, Treatments 1–2) |
| `treatment` | `string` | `measured` | `T1`, `T2` | Nutrient solution formulation treatment |
| `ph` | `float` | `measured` | 4.9 – 7.8 | Handheld portable potentiometric pH reading (faults blanked) |
| `ec` | `float` | `measured` | 0.9 – 4.2 mS/cm | Handheld electrical conductivity meter reading (faults blanked) |
| `tds` | `float` | `measured` | ppm | Total dissolved solids measured by portable meter |
| `water_temp` | `float` | `measured` | 16.0 – 32.0 °C | Nutrient solution temperature from handheld meter |
| `humidity` | `float` | `adjusted` | 32.0 – 89.0 % | Hourly logger relative humidity (meter backup for 36 rows) |
| `air_temp` | `float` | `adjusted` | 19.0 – 38.0 °C | Hourly logger air temperature (meter backup for 36 rows) |
| `co2` | `float` | `measured` | 350 – 1200 ppm | Carbon dioxide concentration from hourly ambient logger |
| `air_temp_day_mean` | `float` | `derived` | °C | Daily mean ambient temperature across logged hours |
| `air_temp_day_min` | `float` | `derived` | °C | Daily minimum ambient temperature |
| `air_temp_day_max` | `float` | `derived` | °C | Daily maximum ambient temperature |
| `humidity_day_mean` | `float` | `derived` | % | Daily mean relative humidity |
| `co2_day_mean` | `float` | `derived` | ppm | Daily mean carbon dioxide concentration |
| `hours_logged` | `integer` | `derived` | 1 – 24 | Number of hourly ambient sensor readings recorded on date |
| `ph_prev` | `float` | `derived` | pH units | Previous slot reading's pH for the same replicate tank |
| `ph_delta` | `float` | `derived` | pH units | Instantaneous rate of change: `ph - ph_prev` |
| `ph_roll3` | `float` | `derived` | pH units | 3-reading trailing rolling mean pH |
| `ec_prev` | `float` | `derived` | mS/cm | Previous slot reading's EC for the same replicate tank |
| `ec_delta` | `float` | `derived` | mS/cm | Instantaneous rate of change: `ec - ec_prev` |
| `ec_roll3` | `float` | `derived` | mS/cm | 3-reading trailing rolling mean EC |
| `water_temp_prev` | `float` | `derived` | °C | Previous slot reading's water temperature |
| `water_temp_delta` | `float` | `derived` | °C | Instantaneous rate of change: `water_temp - water_temp_prev` |
| `water_temp_roll3` | `float` | `derived` | °C | 3-reading trailing rolling mean water temperature |
| `lux_meter` | `float` | `measured` | Lux | Light intensity recorded at portable meter sampling time |
| `label` | `string` | `rule-generated` | 9 categories | Diagnostic fault classification from current action bands |
| `label_next` | `string` | `rule-generated` | 9 categories | Forward-looking condition at next scheduled slot reading |

---

## 9. Known Limitations & Unsupported Interpretations

- **Rule-Derived Labels, Not Clinical Pathology:** Labels are deterministic functions of action threshold rules. No plant pathologist or agronomist inspected plants to confirm biological symptoms for each label. High accuracy on single splits reflects the model relearning thresholds, not independent disease diagnosis.
- **Limited Real Sample Size:** 726 total rows is a small dataset. While `train.csv` contains 1,680 rows, those are oversampled duplicates of minority instances, not new observations.
- **Sensor Scope:** Portable meter data provides bulk solution parameters (`pH`, `EC`, `TDS`, `water_temp`). It does **not** provide individual ionic macro-nutrients (Nitrogen, Phosphorus, Potassium).
- **Unpaired with Images:** Sourced from an Egyptian greenhouse trial; completely unlinked to the Kaggle lettuce leaf images. Cross-modal models cannot be trained end-to-end on these datasets.
- **Crop Generalisation:** Data represents Batavia lettuce in an NFT system; findings must not be generalized directly to spinach or tomato crops.
