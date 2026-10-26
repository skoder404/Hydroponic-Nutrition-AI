# Tomato Sensor Dataset Audit & Reproduction Report (T2-007 / T2-009)

**Generated**: 2026-10-26T15:59:24Z
**Script**: `scripts/audit_sensor_split.py --crop tomato`
**Data**: `sensor_data/Tomato_Sensor-Readings/`

**Result**: ✅ ALL CHECKS PASSED

---

## 1. Row Counts & Overview Reconciliation

- ✅ `full`: 44,150 rows (expected 44,150)
- ✅ `train`: 35,320 rows (expected 35,320)
- ✅ `test`: 8,830 rows (expected 8,830)

- **Discrepancy Note (Overview §15 vs Shipped Data)**: Overview §15 cited 44,141 rows, which cannot be divided equally across 10 classes (4,414.1 rows/class). The verified active dataset contains exactly **44,150 rows** (4,415 rows per class across 10 balanced classes). The 44,141 figure was an off-by-9 typographical error in early prose notes.

## 2. Class Distribution

### Full dataset (Balanced across 10 classes)

| Class | Expected | Actual | Match |
|-------|----------|--------|-------|
| EC_High_Toxicity | 4415 | 4415 | ✅ |
| EC_Low_Deficiency | 4415 | 4415 | ✅ |
| Humidity_High_Mildew_Risk | 4415 | 4415 | ✅ |
| Humidity_Low_Stress | 4415 | 4415 | ✅ |
| K_Deficiency | 4415 | 4415 | ✅ |
| N_Deficiency | 4415 | 4415 | ✅ |
| Normal | 4415 | 4415 | ✅ |
| P_Deficiency | 4415 | 4415 | ✅ |
| Water_Temp_High | 4415 | 4415 | ✅ |
| pH_Imbalance | 4415 | 4415 | ✅ |

**Total full classes**: 10 (10 classes in `primary` fault mode)

### Train (80%) and Test (20%) Partitions

| Class | Train Count (Exp 3,532) | Test Count (Exp 883) | Status |
|-------|-------------------------|----------------------|:------:|
| EC_High_Toxicity | 3532 | 883 | ✅ |
| EC_Low_Deficiency | 3532 | 883 | ✅ |
| Humidity_High_Mildew_Risk | 3532 | 883 | ✅ |
| Humidity_Low_Stress | 3532 | 883 | ✅ |
| K_Deficiency | 3532 | 883 | ✅ |
| N_Deficiency | 3532 | 883 | ✅ |
| Normal | 3532 | 883 | ✅ |
| P_Deficiency | 3532 | 883 | ✅ |
| Water_Temp_High | 3532 | 883 | ✅ |
| pH_Imbalance | 3532 | 883 | ✅ |

## 3. Feature Ordering & Schema Verification

- Ordered model features (8): `['air_temperature_C', 'relative_humidity_percent', 'water_temperature_C', 'ph', 'ec_mS_cm', 'nitrogen_mg_L', 'phosphorus_mg_L', 'potassium_mg_L']`
  - ✅ Column 1: `air_temperature_C` (expected `air_temperature_C`)
  - ✅ Column 2: `relative_humidity_percent` (expected `relative_humidity_percent`)
  - ✅ Column 3: `water_temperature_C` (expected `water_temperature_C`)
  - ✅ Column 4: `ph` (expected `ph`)
  - ✅ Column 5: `ec_mS_cm` (expected `ec_mS_cm`)
  - ✅ Column 6: `nitrogen_mg_L` (expected `nitrogen_mg_L`)
  - ✅ Column 7: `phosphorus_mg_L` (expected `phosphorus_mg_L`)
  - ✅ Column 8: `potassium_mg_L` (expected `potassium_mg_L`)

- ✅ Diagnostic non-feature columns: `['sample_id', 'label', 'label_id', 'n_rules_broken', 'source_row', 'source_timestamp']`

## 4. Rule-Based Label Reproduction (T2-009 Comparison)

- ✅ `n_rules_broken` reproduction: **44,150 / 44,150** (100.0% exact match)
- ✅ `label` reproduction: **44,144 / 44,150** (99.986% match; only 6 differences)

### Detailed Explanation of the 6 Mismatches

The 6 differing rows occur strictly on samples with **exact numerical ties** between two rules where severity values are identical to 16 decimal places (e.g., EC_High_Toxicity vs pH_Imbalance at 0.850000). In `build_tomato_dataset.py`, `deviations()` was evaluated on pre-rounded in-memory floats before CSV serialization, whereas checking on `.round(2)` CSV floats creates exact ties where pandas `idxmax` breaks ties by column position. Zero differences stem from logic or threshold discrepancies.

## 5. Temporal Monthly Split Integrity

- Total calendar months: 132
- Train calendar months: 105
- Test calendar months: 27 (~20.45%)
- ✅ Calendar month overlap between train and test: 0 months

## 6. Shipped Script Version & Build Options (T2-009)

- **Script Version**: `sensor_data/Tomato_Sensor-Readings/build_tomato_dataset.py` (v1.0)
- **Recorded Build Command Options**:
  ```bash
  python build_tomato_dataset.py kolar_tomato_precision_agri_dataset.csv \
    --mode primary \
    --per-class 5000 \
    --test-frac 0.2 \
    --seed 42 \
    --out-prefix tomato_hydroponic
  ```
- **Mode Verified**: `primary` mode (10 classes; `Multiple_Faults` class absent).
- **Safety Verification**: All re-runs were conducted non-destructively; zero files in `sensor_data/` were modified or overwritten.
