# Hybrid 7-Class Hydroponic Spinach Dataset

1,400 samples, 200 per class, split 80/20 into train (1,120) and test (280), stratified (160 + 40 per class).

## Files

| File | Rows | Use |
|---|---|---|
| `spinach_7class_train.csv` | 1,120 | Training |
| `spinach_7class_test.csv` | 280 | Testing |
| `spinach_7class_full.csv` | 1,400 | Everything together (for your own CV or splits) |
| `generate_7class_dataset.py` | | Regenerates everything (`--seed`, `--per-class` options) |

## What is real and what is simulated

| Column | Source |
|---|---|
| `air_temperature_C` | **Real**: Meng et al. 2025 greenhouse loggers (Ag Data Commons), all 1,400 rows |
| `relative_humidity_percent` | **Real for 1,200 rows**; **adjusted for the 200 Humidity_High rows** (see below) |
| `water_temperature_C` | Simulated, follows the real air temperature (about 0.55 x air + 8.5 C, plus noise) |
| `ph`, `ec_mS_cm`, `tds_ppm` | Simulated (TDS = EC x 700, plus 2% noise) |

`source_timestamp_EDT` and `bench` point to the exact real logger reading used in each row, so you can trace the real part back to the source file.

### The Humidity_High caveat

The real greenhouse never exceeded **74.05 %** relative humidity, so no real reading can be "Humidity_High (> 75 %)". For that class I took the 200 real readings with the highest humidity (65 - 74 %) and shifted them to 75.9 - 84.4 %, keeping their rank order. These rows are marked `humidity_source = "adjusted"`. Their air temperature is still real. Say this in your report: 85.7 % of rows have fully real air temperature and humidity, and 14.3 % have real temperature with adjusted humidity.

## Columns

| Column | Meaning |
|---|---|
| `sample_id` | S0001 to S1400 (shuffled) |
| `air_temperature_C`, `relative_humidity_percent`, `water_temperature_C`, `ph`, `ec_mS_cm`, `tds_ppm` | **Features** |
| `label` | Target class name |
| `label_id` | Target class as integer 0-6 |
| `humidity_source` | `real` or `adjusted` |
| `source_timestamp_EDT`, `bench` | Provenance of the real reading. Do not use as features. |

## Classes and rules (from your reference)

| id | Label | Rule |
|---|---|---|
| 0 | Normal | water <= 21 C, EC 1.8 - 2.3, pH 6.0 - 7.0, humidity 45 - 75 % |
| 1 | Water_Temp_High | water temperature > 21 C |
| 2 | EC_High_Toxicity | EC > 2.3 mS/cm |
| 3 | EC_Low_Deficiency | EC < 1.8 mS/cm |
| 4 | pH_Imbalance | pH < 6.0 or > 7.0 (half low, half high) |
| 5 | Humidity_High_Mildew_Risk | humidity > 75 % |
| 6 | Humidity_Low_Stress | humidity < 45 % |

Each non-Normal sample breaks exactly one rule and keeps every other variable in its healthy band. The script re-checks all 1,400 labels against the rules before saving (0 mismatches, 0 rows breaking more than one rule).

## Checks run

- Air temperature matches the source logger for all 1,400 rows, and humidity matches for all 1,200 "real" rows.
- No real reading appears twice, and none is shared between train and test.
- The split is by 6-hour time blocks (not random rows), so no neighbouring 10-minute readings leak across train and test.

## Limits to state in your report

- **The labels are threshold rules, so the problem is easy.** A random forest scores 100 % and a depth-6 decision tree 94.6 % on the test set. That reflects the rules, not real-world performance, and you should not present it as a result. To make the task harder, regenerate with overlapping or noisy classes, or add label noise.
- **pH, EC, TDS and water temperature are simulated**, with ranges taken from published spinach guidance. Real pH and EC logs from the original study were not in the deposit.
- The class balance is a design choice (200 each). Real operation would be mostly Normal.
- The real data comes from baby spinach in a dry greenhouse over about 15 days (mean humidity 40.7 %), so the Humidity_Low class reflects that setting.
- Cite the source: Meng, Q., Biradar, K., Park, Y., Kelley, P.N. (2025), *J. Agric. Food Res.* 24:102530, DOI 10.1016/j.jafr.2025.102530, USDA Ag Data Commons.
