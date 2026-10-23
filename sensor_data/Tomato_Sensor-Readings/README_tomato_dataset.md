# Hydroponic Tomato Dataset (built from the Kolar land-based file)

`build_tomato_dataset.py` converts `kolar_tomato_precision_agri_dataset.csv` (200,000 rows) into a balanced, labelled hydroponic training set with a train/test split.

## Run it

```
python build_tomato_dataset.py kolar_tomato_precision_agri_dataset.csv
```

Needs only `pandas` and `numpy`. Options: `--mode primary|single`, `--per-class N`, `--test-frac 0.2`, `--seed 42`, `--out-prefix tomato_hydroponic`.

Outputs: `tomato_hydroponic_train.csv`, `tomato_hydroponic_test.csv`, `tomato_hydroponic_full.csv`.

## Expected size on the full file

Default (`primary`) mode gives 10 classes with about 3,385 rows each (about 33,850 total; train about 27,080, test about 6,770). The Normal class is the limit, because only about 1.7% of rows break no rule. Sizes were estimated on a 200k mock with the same ranges, so your numbers may differ slightly.

## Columns kept

| Column | Source |
|---|---|
| `air_temperature_C` | Kolar `Ambient_Temp`, unchanged |
| `relative_humidity_percent` | Kolar `Humidity` + 15 (proxy) |
| `ph` | Kolar `Soil_pH` - 1.0 (proxy) |
| `ec_mS_cm` | Kolar `EC_mS_cm` x 1.4 (proxy) |
| `nitrogen_mg_L` | Kolar `Nitrogen_Level` x 4 (proxy) |
| `phosphorus_mg_L` | Kolar `Phosphorus_Level` x 3 (proxy) |
| `potassium_mg_L` | Kolar `Potassium_Level` x 6 (proxy) |
| `water_temperature_C` | **Synthetic**: 8 + 0.55 x air temp + 0.003 x (solar - 700) + noise (sd 0.6) |

`label`, `label_id`, `n_rules_broken` are targets or diagnostics. `source_row` and `source_timestamp` trace each row to the original file; do not use them as features.

Dropped: soil moisture, pore-water velocity, dispersion, solute flux, all 15 RSSI columns (not relevant to a hydroponic solution), and `Nutrient_Lockout_Prob` (it is derived from pH, so keeping it would leak the label).

## Classes and thresholds

| Label | Rule | Basis |
|---|---|---|
| Normal | no rule broken | |
| Water_Temp_High | water > 25 C | optimum tomato root temperature 25 C (Hort Americas) |
| EC_High_Toxicity | EC > 4.0 mS/cm | Oklahoma State tomato EC range 2.0-4.0 |
| EC_Low_Deficiency | EC < 2.0 | Oklahoma State |
| pH_Imbalance | pH < 6.0 or > 6.5 | Oklahoma State (OSU CFAES drip range is 5.5-6.5) |
| Humidity_High_Mildew_Risk | humidity > 85 % | leaf mould risk |
| Humidity_Low_Stress | humidity < 50 % | |
| N_Deficiency | N < 90 mg/L | Ohio State CFAES / UA-CEA tomato solution (N 90-190) |
| P_Deficiency | P < 35 mg/L | same (P about 47) |
| K_Deficiency | K < 144 mg/L | same (K 144-350) |

In `primary` mode a row that breaks several rules is labelled with its most severe fault (deviation past the threshold, scaled by the worst deviation in the file). `--mode single` keeps only one-fault rows and adds a `Multiple_Faults` class, but yields only about 470 rows per class.

## Validation done

- Every mapped value traces back to `source_row` exactly (7 of 7 columns).
- No source row and no calendar month appears in both train and test (split is by month).
- Labels re-derived from the saved values match for every row.
- Quick model check on the 200k mock: logistic regression 86.6%, decision tree (depth 8) 79.2%, decision tree (depth 20) 94.6%, random forest 96.7% accuracy. Rows that break 0-1 rules are classified perfectly. Errors come from multi-fault rows.

## Limits to state in your report

1. **The Kolar file looks simulated, not real field data.** Every variable is uniformly distributed and independent (e.g. pH 6.2-8.5, EC 0.5-3.0, N 10-50 with flat quartiles), timestamps are random with no daily or seasonal pattern, and `Nutrient_Lockout_Prob` is a function of pH. Do not describe it as real IoT sensor data.
2. **The soil-to-hydroponic mappings are arbitrary proxy transforms** chosen so the ranges overlap hydroponic norms. They are not physical conversions.
3. **Water temperature is fully synthetic.**
4. **Labels are threshold rules**, so the task is learnable by construction. High accuracy shows the model recovers the rules, not real-world performance.
5. Class balance is a design choice; real operation would be mostly Normal.
