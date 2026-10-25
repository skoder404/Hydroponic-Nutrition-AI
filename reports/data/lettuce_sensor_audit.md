# Lettuce Sensor Dataset Audit Report

**Generated**: 2026-10-25T13:57:42Z
**Script**: `scripts/audit_sensor_split.py`
**Data**: `sensor_data/Lettuce_Sensor-Readings/`

**Result**: ✅ ALL CHECKS PASSED

---

## 1. Row Counts

- ✅ `full`: 726 rows (expected 726)
- ✅ `train`: 1680 rows (expected 1680)
- ✅ `test`: 162 rows (expected 162)

## 2. Class Distribution

### Full dataset

| Class | Expected | Actual | Match |
|-------|----------|--------|-------|
| Normal | 327 | 327 | ✅ |
| Multiple_Stress | 135 | 135 | ✅ |
| Humidity_Low | 87 | 87 | ✅ |
| WaterTemp_High | 75 | 75 | ✅ |
| EC_High | 58 | 58 | ✅ |
| pH_High | 25 | 25 | ✅ |
| pH_Low | 10 | 10 | ✅ |
| WaterTemp_Low | 8 | 8 | ✅ |
| EC_Low | 1 | 1 | ✅ |

**Total full classes**: 9 (9 with dropped, 7 kept)

### Train dataset

| Class | Count | Expected (240) | Match |
|-------|-------|----------------|-------|
| pH_Low | 240 | 240 | ✅ |
| Humidity_Low | 240 | 240 | ✅ |
| Normal | 240 | 240 | ✅ |
| WaterTemp_High | 240 | 240 | ✅ |
| pH_High | 240 | 240 | ✅ |
| Multiple_Stress | 240 | 240 | ✅ |
| EC_High | 240 | 240 | ✅ |

**Train classes**: 7 ✅ (expected 7)
- ✅ Dropped class `WaterTemp_Low` correctly absent from train and test
- ✅ Dropped class `EC_Low` correctly absent from train and test

## 3. Action Bands & Label Assignment Rules

| Variable | Low Threshold | High Threshold | Status |
|----------|---------------|----------------|--------|
| ph | 5.5 | 7.0 | ✅ |
| ec | 1.2 | 2.5 | ✅ |
| air_temp | 18.0 | 30.0 | ✅ |
| humidity | 50.0 | 80.0 | ✅ |
| water_temp | 18.0 | 26.0 | ✅ |

- Action bands define stress when values fall outside the target range.
- A single variable out-of-bounds yields that specific variable's stress class.
- Two or more out-of-bounds variables yield `Multiple_Stress`.
- Rare classes (< 8 rows in training raw split) are dropped: `WaterTemp_Low` (3 train rows) and `EC_Low` (0 train rows).

## 4. Every-4th-Day Split Rule

### EXP.1

- Total unique days: 26
- Expected test days (every 4th): 6
- Actual test days: 6
- ✅ Days match: True

### EXP.2

- Total unique days: 29
- Expected test days (every 4th): 7
- Actual test days: 7
- ✅ Days match: True

### EXP.3

- Total unique days: 14
- Expected test days (every 4th): 3
- Actual test days: 3
- ✅ Days match: True

## 5. Train/Test Separation (No Data Leakage)

- ✅ No (experiment, date) pair appears in both train and test

## 6. Oversampling Quantification

- Raw training split rows before class filtering: **558** (matches expected 558)
- Rare class occurrences in training split dropped: 3 rows (`WaterTemp_Low`: 3, `EC_Low`: 0)
- Real training rows kept across 7 classes: **555**
- Total train rows after oversampling: **1680**
- Duplicated rows created by oversampling: **1125**
- Oversampling expansion ratio: **3.03x**

### Per-class oversampling breakdown

| Class | Real Rows in Train Split | Oversampled Total | Duplicate Rows Added | Expansion Factor |
|-------|--------------------------|-------------------|----------------------|------------------|
| EC_High | 49 | 240 | +191 | 4.90x |
| Humidity_Low | 71 | 240 | +169 | 3.38x |
| Multiple_Stress | 112 | 240 | +128 | 2.14x |
| Normal | 240 | 240 | +0 | 1.00x |
| WaterTemp_High | 56 | 240 | +184 | 4.29x |
| pH_High | 19 | 240 | +221 | 12.63x |
| pH_Low | 8 | 240 | +232 | 30.00x |

## 7. Known Discrepancies (Documented from Task Spec)

### 7.1 Script docstring vs code
- **Docstring** (lines 15-16 in `build_lettuce_dataset`): mentions EXP.3 as test set
- **Actual Code** (`day_holdout_split`, lines 266-281): holds out every 4th calendar day across all 3 experiments
- **README** (lines 94-95): confirms every-4th-day holdout rule
- ✅ **Verified**: docstring is outdated; code and README agree on the every-4th-day rule.

### 7.2 README filename references
- **README** (line 17): references `build_dataset.py` and `train_model.py`
- **Repo Actual**: contains `build_lettuce_dataset` (without `.py` extension) and no `train_model.py`
- `build_lettuce_dataset` present: ✅ Yes
- `train_model.py` present: ❌ No (confirmed discrepancy)

### 7.3 Non-model-input columns in CSVs
- Model uses 5 core physical inputs: `['air_temp', 'ec', 'humidity', 'ph', 'water_temp']`
- Full CSV retains 25 additional metadata/engineering columns: `['air_temp_day_max', 'air_temp_day_mean', 'air_temp_day_min', 'co2', 'co2_day_mean', 'date', 'day_index', 'ec_delta', 'ec_prev', 'ec_roll3', 'experiment', 'hours_logged', 'humidity_day_mean', 'lux_meter', 'ph_delta', 'ph_prev', 'ph_roll3', 'replicate', 'slot', 'tds', 'timestamp', 'treatment', 'water_temp_delta', 'water_temp_prev', 'water_temp_roll3']`
- Includes trend features (`ec_trend_3h`), daily aggregates (`ph_daily_mean`), secondary sensors (`tds`, `co2`, `lux_meter`), and target forecasting (`label_next`).

### 7.4 Raw Excel workbooks (`EXP_*.xlsx`)
- The three original raw Excel workbooks (`EXP_1.xlsx`, `EXP_2.xlsx`, `EXP_3.xlsx`) are not stored in the repository.
- Consequently, `build_lettuce_dataset` cannot be re-executed without downloading them, which is not required for this project.
- The generated CSVs (`hydroponic_lettuce_{full,train,test}.csv`) are the checked-in source of truth.
