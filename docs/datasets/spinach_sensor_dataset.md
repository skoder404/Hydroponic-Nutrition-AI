# Dataset Provenance: Hybrid 7-Class Hydroponic Spinach Sensor Dataset

- **Document Version:** 1.0.0
- **Author:** Sushil
- **Related Tasks:** T2-003, T1-001, T4-001
- **Target Crop:** Baby Spinach (*Spinacia oleracea*)
- **Modality:** Environmental & Nutrient Solution Time-Series
- **Last Verified Date:** 2026-10-25

---

## 1. Source Identification & Origin

- **Source Name:** USDA Ag Data Commons & Journal of Agriculture and Food Research (Meng et al., 2025)
- **Dataset Canonical URL:** https://doi.org/10.15482/USDA.ADC/33383695
- **Publication DOI:** 10.1016/j.jafr.2025.102530
- **Authors:** Qingwu Meng, Kishan Biradar, Yujin Park, Patrick N. Kelley (University of Delaware)
- **Publication Reference:** *Journal of Agriculture and Food Research*, Vol. 24, Article 102530 (2025)
- **Source Version / Date Downloaded:** Published 2025 / Ingested October 2026
- **Original Filename(s):** `Meng_Biradar_Park_Kelley_2025_greenhouse_temperature_humidity.csv` (Ag Data Commons repository)
- **Current Repository Location:** `sensor_data/Spinach_Sensor-Readings/`

---

## 2. License, Terms & Redistribution Status

- **License Type:** CC0 1.0 Universal (Public Domain Dedication) / Open Data Commons
- **Original Terms / Rights Holder:** USDA Ag Data Commons / Open Access publication under Elsevier CC BY license
- **Redistribution Status:** Publicly accessible; the derived dataset and regeneration script are committed in the repository, while raw multi-gigabyte greenhouse logs remain in the external archive.
- **Commercial / Non-Commercial Constraints:** Unrestricted public research data.

---

## 3. Sample Counts & Volume

| Metric Stage | Sample Count (Rows) | Notes / Filtering Rationale |
|---|---|---|
| **Original Upstream Log Count** | Multi-thousand | 10-minute sensor intervals across ~15 days in research greenhouse |
| **Filtered / Sampled Count** | 1,400 rows | Subsampled and structured into balanced 7 classes (200 rows per class) |
| **Prepared Split: Train** | 1,120 rows | 80% partition (160 rows per class), partitioned by 6-hour time blocks |
| **Prepared Split: Test** | 280 rows | 20% partition (40 rows per class), partitioned by 6-hour time blocks |

---

## 4. Class Labels & Distribution

Balanced 7-class synthetic fault classification (200 rows per class in full dataset):

| Class Label | Label ID | Full Count | Train Count | Test Count | Prevalence (%) | Threshold Diagnostic Definition |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `Normal` | 0 | 200 | 160 | 40 | 14.28% | Water $\le$ 21°C, EC 1.8–2.3 mS/cm, pH 6.0–7.0, Humidity 45–75% |
| `Water_Temp_High` | 1 | 200 | 160 | 40 | 14.28% | Water temperature > 21.0°C |
| `EC_High_Toxicity` | 2 | 200 | 160 | 40 | 14.28% | Electrical conductivity > 2.3 mS/cm |
| `EC_Low_Deficiency` | 3 | 200 | 160 | 40 | 14.28% | Electrical conductivity < 1.8 mS/cm |
| `pH_Imbalance` | 4 | 200 | 160 | 40 | 14.28% | pH < 6.0 (low) or pH > 7.0 (high) |
| `Humidity_High_Mildew_Risk` | 5 | 200 | 160 | 40 | 14.28% | Relative humidity > 75.0% |
| `Humidity_Low_Stress` | 6 | 200 | 160 | 40 | 14.28% | Relative humidity < 45.0% |

Each non-Normal sample breaks exactly one rule and keeps every other parameter in its healthy band.

---

## 5. Preprocessing & Transformation Pipeline

- **Preprocessing Script & Version:** `sensor_data/Spinach_Sensor-Readings/generate_7class_dataset.py`
- **Cleaning & Imputation Steps:** Subsampled real environmental logger readings; no missing sensor values present in exported CSVs.
- **Normalization / Scaling:** No pre-scaling applied in raw CSVs; scalers must be fit strictly on training splits during modeling.
- **Data Partitions:** Split generated using 6-hour contiguous time blocks to prevent proximate temporal data leakage between train and test.

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** Stratified 6-hour time-block partitioning (80% train / 20% test).
- **Split Rationale:** Consecutive 10-minute greenhouse readings exhibit intense temporal auto-correlation. Splitting contiguous 6-hour blocks ensures neighbouring readings do not cross between train and test splits.
- **Manifest / Files:**
  - `sensor_data/Spinach_Sensor-Readings/spinach_7class_full.csv` (1,400 rows)
  - `sensor_data/Spinach_Sensor-Readings/spinach_7class_train.csv` (1,120 rows)
  - `sensor_data/Spinach_Sensor-Readings/spinach_7class_test.csv` (280 rows)

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values (Overview §13.4)

This dataset is a hybrid real/synthetic dataset:
- **Real Environmental Measurements:** `air_temperature_C` across all 1,400 rows is real logger data from the USDA Meng et al. experiment.
- **Real/Adjusted Humidity:** `relative_humidity_percent` is real measured data for 1,200 rows. Because the real greenhouse never exceeded 74.05% RH, the 200 rows for `Humidity_High_Mildew_Risk` were adjusted by shifting the highest real humidity readings into the 75.9%–84.4% range.
- **Simulated Solution Chemistry:** `water_temperature_C`, `ph`, and `ec_mS_cm` were simulated. Water temperature follows air temperature via `water_temp ≈ 0.55 × air_temp + 8.5°C + Normal(0, 0.4°C)`.
- **Derived TDS:** `tds_ppm` is derived from `ec_mS_cm` via `tds = ec_mS_cm × 700 + 2% noise`. It is redundant with EC and must be excluded from model inputs.
- **Rule-Generated Labels:** All labels are assigned by hard conditional thresholding rules.

---

## 8. Column / Field Taxonomy Table (Overview §13.3 & §13.4)

| Column Name | Data Type | Provenance Tag | Model Input Status | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|:---:|---|---|
| `air_temperature_C` | `float` | `measured` | **Included (Feature 1)** | 14.1°C – 32.8°C | Real greenhouse logger temperature (Meng et al. 2025) |
| `relative_humidity_percent` | `float` | `adjusted` | **Included (Feature 2)** | 20.3% – 84.4% | Real for 1,200 rows; adjusted >75% for 200 high-humidity rows |
| `water_temperature_C` | `float` | `synthetic` | **Included (Feature 3)** | 16.5°C – 26.5°C | Simulated: `0.55 × air_temp + 8.5 + N(0, 0.4)` |
| `ph` | `float` | `synthetic` | **Included (Feature 4)** | 5.2 – 7.8 | Simulated within or outside healthy 6.0–7.0 band |
| `ec_mS_cm` | `float` | `synthetic` | **Included (Feature 5)** | 1.2 – 2.9 mS/cm | Simulated within or outside healthy 1.8–2.3 band |
| `tds_ppm` | `float` | `derived` | **EXCLUDED (Redundant / Derived)** | ppm | Calculated as `ec_mS_cm × 700 + 2% noise`; redundant with EC |
| `label` | `string` | `rule-generated` | **Target (Supervised Label)** | 7 Categories | Multi-class target label assigned by single-fault threshold rule |
| `label_id` | `integer` | `rule-generated` | **EXCLUDED (Direct Leakage)** | 0 – 6 | Integer mapping of target class; direct leakage of target |
| `humidity_source` | `string` | `derived` | **EXCLUDED (Shortcut Leakage)** | `real` / `adjusted` | Reveals if humidity was shifted; creates artificial shortcut |
| `sample_id` | `string` | `synthetic` | **EXCLUDED (Identifier)** | S0001 – S1400 | Unique instance surrogate identifier |
| `source_timestamp_EDT`| `string` | `measured` | **EXCLUDED (Timestamp)** | ISO Timestamp | Original logger timestamp of source reading |
| `bench` | `string` | `measured` | **EXCLUDED (Location Identifier)**| Bench A – E | Greenhouse bench spatial location identifier |

---

## 9. Known Limitations & Unsupported Interpretations

1. **Threshold Rule Artifacts:** The classification labels reflect synthetic threshold rules rather than actual plant pathology or biological tissue analysis. Machine learning models (decision trees / random forests) achieve near 100% test accuracy simply by learning these exact split boundaries.
2. **Not Biological Validation:** Near-perfect model accuracy on this dataset must never be claimed as proving real-world hydroponic crop diagnostics.
3. **No Multimodal Pairing:** Per Overview §19, these sensor rows have no paired alignment with the spinach leaf photographs under `image_data/Spinach_Images/`. Multimodal fusion is prohibited.
