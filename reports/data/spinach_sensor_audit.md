# Spinach Sensor Dataset Audit Report

**Generated**: 2026-10-25T16:04:43Z
**Script**: `scripts/audit_sensor_split.py --crop spinach`
**Data**: `sensor_data/Spinach_Sensor-Readings/`

**Result**: ✅ ALL CHECKS PASSED

---

## 1. Row Counts

- ✅ `full`: 1400 rows (expected 1400)
- ✅ `train`: 1120 rows (expected 1120)
- ✅ `test`: 280 rows (expected 280)

## 2. Class Distribution

### Full dataset

| Class | Expected | Actual | Match |
|-------|----------|--------|-------|
| Normal | 200 | 200 | ✅ |
| Water_Temp_High | 200 | 200 | ✅ |
| EC_High_Toxicity | 200 | 200 | ✅ |
| EC_Low_Deficiency | 200 | 200 | ✅ |
| pH_Imbalance | 200 | 200 | ✅ |
| Humidity_High_Mildew_Risk | 200 | 200 | ✅ |
| Humidity_Low_Stress | 200 | 200 | ✅ |

### Train dataset (80% stratified)

| Class | Count | Expected (160) | Match |
|-------|-------|----------------|-------|
| Normal | 160 | 160 | ✅ |
| Water_Temp_High | 160 | 160 | ✅ |
| EC_High_Toxicity | 160 | 160 | ✅ |
| EC_Low_Deficiency | 160 | 160 | ✅ |
| pH_Imbalance | 160 | 160 | ✅ |
| Humidity_High_Mildew_Risk | 160 | 160 | ✅ |
| Humidity_Low_Stress | 160 | 160 | ✅ |

### Test dataset (20% stratified)

| Class | Count | Expected (40) | Match |
|-------|-------|---------------|-------|
| Normal | 40 | 40 | ✅ |
| Water_Temp_High | 40 | 40 | ✅ |
| EC_High_Toxicity | 40 | 40 | ✅ |
| EC_Low_Deficiency | 40 | 40 | ✅ |
| pH_Imbalance | 40 | 40 | ✅ |
| Humidity_High_Mildew_Risk | 40 | 40 | ✅ |
| Humidity_Low_Stress | 40 | 40 | ✅ |

## 3. Split Integrity: 6-Hour Temporal Block Holdout

- ✅ **6-hour temporal block overlap between train and test**: 0 blocks
  - Train unique blocks: 88
  - Test unique blocks: 22
  - ✅ Zero temporal blocks are shared between train and test splits; no 10-minute neighbouring readings leak across splits.

- ✅ **Sample ID overlap**: 0 samples

## 4. Column Provenance & Feature Classification

| Column | Provenance Type | Model Feature? | Details |
|--------|-----------------|----------------|---------|
| `air_temperature_C` | Real (100%) | ✅ Feature | Greenhouse loggers from Meng et al. (2025), Ag Data Commons |
| `relative_humidity_percent` | Real (85.7%) / Adjusted (14.3%) | ✅ Feature | Real for 1,200 rows; 200 `Humidity_High_Mildew_Risk` rows shifted to 75.9-84.4% |
| `water_temperature_C` | Synthetic | ✅ Feature | Simulated: 0.55 * air_temp + 8.5 + noise (elevated for Water_Temp_High) |
| `ph` | Synthetic | ✅ Feature | Simulated: optimal 6.05-6.95, altered for pH_Imbalance |
| `ec_mS_cm` | Synthetic | ✅ Feature | Simulated: optimal 1.82-2.28, altered for EC_High / EC_Low |
| `tds_ppm` | **Derived (Excluded)** | ❌ Excluded | EC x 700 + 2% noise (redundant with EC, leakage risk) |
| `humidity_source` | Metadata | ❌ Excluded | Indicator `real` vs `adjusted` |
| `source_timestamp_EDT` | Metadata / Provenance | ❌ Excluded | Timestamp of real greenhouse logger reading |
| `bench` | Metadata / Provenance | ❌ Excluded | Logger bench identifier |
| `sample_id` | Metadata | ❌ Excluded | Shuffled sample identifier S0001-S1400 |
| `label_id` | Target | ❌ Excluded | Numeric label ID (0-6) |

- ✅ `humidity_source` counts: `real`=1200 (85.71%), `adjusted`=200 (14.29%)

- ✅ `tds_ppm` derivation check: mean ratio TDS / (EC * 700) = 1.0004 (within expected ±2% noise band)

## 5. Labelling Rules & Single-Violation Verification

- ✅ **Rule agreement**: 0 disagreements between re-derived rules and assigned labels
- ✅ **Single-rule violation constraint**: 0 rows broke more than one rule (expected 0)

## 6. Dataset Limitations & Source Availability

- **Rule-based synthetic targets**: Because labels are hard threshold rules without overlap, classifiers trivially achieve near 100% test accuracy. This should not be reported as real-world sensor diagnostic performance.
- **Simulated chemistry**: pH, EC, and water temperature are synthetic; real nutrient solution logs were not included in the source deposit.
- **Class balance artifact**: Equal balance (200 per class) was intentionally drawn; real-world greenhouse operations would be heavily skewed towards Normal.
- **Source file availability**: The raw file `Meng_Biradar_Park_Kelley_2025_greenhouse_temperature_humidity.csv` is not tracked in the repository, so `generate_7class_dataset.py` cannot be re-executed directly without obtaining the raw CSV from USDA Ag Data Commons.
