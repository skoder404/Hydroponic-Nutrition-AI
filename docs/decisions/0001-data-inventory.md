# ADR 0001: Data and Scripts Inventory

- **Status:** Accepted
- **Date:** 2026-10-24
- **Driver:** Yashwant Vadhan
- **Navigator:** Sushil
- **Task ID:** T1-001

## Context

The repository contains raw/processed image datasets under `image_data/` and sensor datasets under `sensor_data/` spanning three target crops: Lettuce, Spinach, and Tomato. Prior to designing models, feature pipelines, or writing application code, this decision record inventories all existing data files, folder hierarchies, formats, file sizes, column schemas, row counts, and script provenance.

In accordance with repo safety rules:
- No existing files were modified, moved, or renamed.
- Raw sensor and image data are treated strictly as read-only.

## Reproducibility & Verification

To programmatically reproduce and verify all metrics in this ADR at any time, run:
```bash
python scripts/inventory_data.py
```
This script dynamically inspects the folder structures, sizes, formats, CSV schemas, and row counts directly from disk.

---

## 1. Directory Inventory: Counts, Formats, and Sizes

### 1.1 `image_data/` Hierarchy

| Directory Path | File Count | Formats Breakdown | Total Size (Bytes) | Formatted Size |
|---|---|---|---|---|
| `image_data` (root) | 0 | None (empty directory) | 0 | 0 B |
| `image_data/Lettuce_Images` | 2 | `.csv` (2) | 154,543 | 150.92 KB |
| `image_data/Lettuce_Images/Healthier` | 389 | `.jpg` (389) | 32,525,551 | 31.02 MB |
| `image_data/Lettuce_Images/NitrogenDeficiency` | 57 | `.jpg` (57) | 15,682,602 | 14.96 MB |
| `image_data/Lettuce_Images/PhosphorusDeficiency` | 66 | `.jpg` (66) | 24,105,602 | 22.99 MB |
| `image_data/Lettuce_Images/PotassiumDeficiency` | 68 | `.jpg` (68) | 19,060,541 | 18.18 MB |
| **Subtotal: Lettuce Images** | **582** | **`.jpg` (580), `.csv` (2)** | **91,528,839** | **87.29 MB** |
| `image_data/Spinach_Images` (root) | 0 | None (empty directory) | 0 | 0 B |
| `image_data/Spinach_Images/Spinach_Healthy` | 7,218 | `.jpeg` (6,823), `.png` (395) | 437,831,812 | 417.55 MB |
| `image_data/Spinach_Images/Spinach_Unhealthy` (root) | 0 | None (empty directory) | 0 | 0 B |
| `image_data/Spinach_Images/Spinach_Unhealthy/Spinach_anthracnose` | 783 | `.jpeg` (783) | 427,667,071 | 407.86 MB |
| `image_data/Spinach_Images/Spinach_Unhealthy/Spinach_bacterial spot` | 4,565 | `.jpeg` (4,565) | 273,622,601 | 260.95 MB |
| `image_data/Spinach_Images/Spinach_Unhealthy/Spinach_downy mildew` | 1,767 | `.jpeg` (1,767) | 116,265,230 | 110.88 MB |
| `image_data/Spinach_Images/Spinach_Unhealthy/Spinach_Non fresh` | 616 | `.png` (616) | 51,067,656 | 48.70 MB |
| `image_data/Spinach_Images/Spinach_Unhealthy/Spinach_pest damage` | 3,353 | `.jpeg` (3,353) | 213,962,942 | 204.05 MB |
| **Subtotal: Spinach Images** | **18,302** | **`.jpeg` (17,291), `.png` (1,011)** | **1,520,417,312** | **1,450.00 MB (1.42 GB)** |
| `image_data/Tomato_Images` (root) | 0 | None (empty directory) | 0 | 0 B |
| `image_data/Tomato_Images/Tomato_Healthy` | 4,014 | `.jpg` (3,737), `.png` (277) | 383,733,844 | 365.96 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy` (root) | 0 | None (empty directory) | 0 | 0 B |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Bacterial_spot` | 3,590 | `.jpg` (3,590) | 175,416,299 | 167.29 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Early_blight` | 3,393 | `.jpg` (3,390), `.jpeg` (2), `.png` (1) | 383,888,625 | 366.10 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Late_blight` | 3,945 | `.jpg` (3,719), `.png` (226) | 276,385,452 | 263.58 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Leaf_Mold` | 3,893 | `.jpg` (3,893) | 278,563,030 | 265.66 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Powdery_Mildew` | 1,256 | `.jpg` (1,256) | 11,500,355 | 10.97 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Septoria_leaf_spot` | 3,628 | `.jpg` (3,626), `.jpeg` (2) | 235,536,153 | 224.62 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Spider_mites Two-spotted_spider_mite` | 2,187 | `.jpg` (2,187) | 49,209,112 | 46.93 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Target_Spot` | 2,287 | `.jpg` (2,287) | 45,400,181 | 43.30 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Tomato_mosaic_virus` | 2,737 | `.jpg` (2,735), `.png` (2) | 230,740,590 | 220.05 MB |
| `image_data/Tomato_Images/Tomato_Unhealthy/Tomato_Tomato_Yellow_Leaf_Curl_Virus` | 2,634 | `.jpg` (2,634) | 64,454,714 | 61.47 MB |
| **Subtotal: Tomato Images** | **33,564** | **`.jpg` (33,058), `.png` (502), `.jpeg` (4)** | **2,134,828,355** | **2,035.93 MB (1.99 GB)** |
| **Total `image_data/`** | **52,448** | **52,446 images, 2 CSVs** | **3,746,774,506** | **3,573.22 MB (3.49 GB)** |

### 1.2 `sensor_data/` Hierarchy

| Directory Path | File Count | Formats Breakdown | Total Size (Bytes) | Formatted Size |
|---|---|---|---|---|
| `sensor_data` (root) | 0 | None (empty directory) | 0 | 0 B |
| `sensor_data/Lettuce_Sensor-Readings` | 5 | `.csv` (3), no extension (2) | 741,465 | 724.09 KB |
| `sensor_data/Spinach_Sensor-Readings` | 5 | `.csv` (3), `.py` (1), `.md` (1) | 270,663 | 264.32 KB |
| `sensor_data/Tomato_Sensor-Readings` | 5 | `.csv` (3), `.py` (1), `.md` (1) | 8,680,847 | 8.28 MB |
| **Total `sensor_data/`** | **15** | **`.csv` (9), `.py` (2), `.md` (2), no extension (2)** | **9,692,975** | **9.24 MB** |

---

## 2. Sensor Data Files: Row Counts, Columns, and Script Provenance

Every file residing in `sensor_data/` is detailed below with its exact row count (excluding header), column list, and the script that produced or manages it.

### 2.1 Lettuce Sensor Readings (`sensor_data/Lettuce_Sensor-Readings/`)

| File Name | Format | Size (Bytes) | Rows / Lines | Columns (if CSV) | Producing Script / Origin |
|---|---|---|---|---|---|
| `README_lettuce_dataset` | Markdown (no extension) | 7,036 | 144 lines | N/A | Authored dataset documentation. Describes source data, corrections, columns, labels, splits, and limitations. |
| `build_lettuce_dataset` | Python (no extension) | 13,208 | 317 lines | N/A | Original preparation script. Parses HydroGrowNet Batavia workbooks (`EXP_1`, `EXP_2`, `EXP_3`), fixes date/header order, drops invalid sensor readings, assigns rule-based labels, and exports full, train, and test CSV splits. |
| `hydroponic_lettuce_full.csv` | CSV | 203,658 | 726 rows | 32 columns (see list below) | Generated by `build_lettuce_dataset`. Contains all 726 portable-meter readings with valid pH/EC joined with hourly air loggers. |
| `hydroponic_lettuce_train.csv` | CSV | 471,075 | 1,680 rows | 32 columns (see list below) | Generated by `build_lettuce_dataset`. Training split built from EXP.1 and EXP.2, oversampled to 240 rows per class across 7 retained classes (rare `EC_Low` and `WaterTemp_Low` dropped). |
| `hydroponic_lettuce_test.csv` | CSV | 46,488 | 162 rows | 32 columns (see list below) | Generated by `build_lettuce_dataset`. Held-out split (every 4th calendar day of each experiment), keeping untouched real class distribution. |

#### Lettuce CSV Columns (32 columns)
`experiment`, `date`, `day_index`, `slot`, `timestamp`, `replicate`, `treatment`, `ph`, `ec`, `tds`, `water_temp`, `humidity`, `air_temp`, `co2`, `air_temp_day_mean`, `air_temp_day_min`, `air_temp_day_max`, `humidity_day_mean`, `co2_day_mean`, `hours_logged`, `ph_prev`, `ph_delta`, `ph_roll3`, `ec_prev`, `ec_delta`, `ec_roll3`, `water_temp_prev`, `water_temp_delta`, `water_temp_roll3`, `lux_meter`, `label`, `label_next`

---

### 2.2 Spinach Sensor Readings (`sensor_data/Spinach_Sensor-Readings/`)

| File Name | Format | Size (Bytes) | Rows / Lines | Columns (if CSV) | Producing Script / Origin |
|---|---|---|---|---|---|
| `README_7class_dataset.md` | Markdown (`.md`) | 4,071 | 67 lines | N/A | Authored dataset documentation. Details the hybrid simulation methodology, 7 target classes, feature provenance, and split integrity. |
| `generate_7class_dataset.py` | Python (`.py`) | 9,229 | 252 lines | N/A | Dataset generation script. Samples real greenhouse ambient data (Meng et al. 2025, USDA Ag Data Commons), synthesizes solution metrics (`ph`, `ec_mS_cm`, `tds_ppm`, `water_temperature_C`), applies single-fault rules, and exports splits. |
| `spinach_7class_full.csv` | CSV | 128,605 | 1,400 rows | 12 columns (see list below) | Generated by `generate_7class_dataset.py`. Complete balanced dataset (200 rows per class across 7 classes). |
| `spinach_7class_train.csv` | CSV | 102,945 | 1,120 rows | 12 columns (see list below) | Generated by `generate_7class_dataset.py`. 80% train split (160 rows per class), partitioned by 6-hour time blocks to prevent temporal data leakage. |
| `spinach_7class_test.csv` | CSV | 25,813 | 280 rows | 12 columns (see list below) | Generated by `generate_7class_dataset.py`. 20% test split (40 rows per class), partitioned by 6-hour time blocks. |

#### Spinach CSV Columns (12 columns)
`sample_id`, `air_temperature_C`, `relative_humidity_percent`, `water_temperature_C`, `ph`, `ec_mS_cm`, `tds_ppm`, `label`, `label_id`, `humidity_source`, `source_timestamp_EDT`, `bench`

---

### 2.3 Tomato Sensor Readings (`sensor_data/Tomato_Sensor-Readings/`)

| File Name | Format | Size (Bytes) | Rows / Lines | Columns (if CSV) | Producing Script / Origin |
|---|---|---|---|---|---|
| `README_tomato_dataset.md` | Markdown (`.md`) | 4,183 | 67 lines | N/A | Authored dataset documentation. Outlines conversion from Kolar land-based dataset to hydroponic target, 10 primary fault classes, and split validation. |
| `build_tomato_dataset.py` | Python (`.py`) | 9,582 | 266 lines | N/A | Dataset preparation script. Ingests `kolar_tomato_precision_agri_dataset.csv`, maps soil features to hydroponic nutrient ranges, synthesizes water temp, assigns fault labels in `primary` mode, and exports splits. |
| `tomato_hydroponic_full.csv` | CSV | 4,333,446 | 44,150 rows | 14 columns (see list below) | Generated by `build_tomato_dataset.py`. Balanced multi-class dataset (approx. 4,415 rows per class across 10 classes in primary fault mode). |
| `tomato_hydroponic_train.csv` | CSV | 3,467,475 | 35,320 rows | 14 columns (see list below) | Generated by `build_tomato_dataset.py`. 80% training split partitioned by calendar month (approx. 3,532 rows per class). |
| `tomato_hydroponic_test.csv` | CSV | 866,161 | 8,830 rows | 14 columns (see list below) | Generated by `build_tomato_dataset.py`. 20% test split partitioned by calendar month (approx. 883 rows per class). |

#### Tomato CSV Columns (14 columns)
`sample_id`, `air_temperature_C`, `relative_humidity_percent`, `water_temperature_C`, `ph`, `ec_mS_cm`, `nitrogen_mg_L`, `phosphorus_mg_L`, `potassium_mg_L`, `label`, `label_id`, `n_rules_broken`, `source_row`, `source_timestamp`

---

## 3. Discrepancy Analysis & Renaming Decisions

### 3.1 Extension-less Files in `sensor_data/Lettuce_Sensor-Readings/`
1. `build_lettuce_dataset`:
   - Inspecting lines 1–40 confirms it is a standard Python 3 CLI script with imports (`glob`, `os`, `argparse`, `pandas`, `numpy`) and execution blocks.
   - It currently lacks the `.py` extension.
2. `README_lettuce_dataset`:
   - Inspecting the content confirms it is GitHub Flavored Markdown documentation.
   - It currently lacks the `.md` extension.

**Decision:**
To respect the task requirement ("Nothing was modified, moved, or renamed") and preserve git history, no files were renamed in this commit. It is recommended to perform a clean, dedicated rename commit (`git mv`) in a subsequent task to normalize these to `build_lettuce_dataset.py` and `README_lettuce_dataset.md`.

### 3.2 Referencing Discrepancies in `README_lettuce_dataset`
- `README_lettuce_dataset` refers to:
  - `build_dataset.py` — Not present in repository (the actual script is named `build_lettuce_dataset`).
  - `train_model.py` — Not present in repository.
  - `hydroponic_air_hourly.csv` (2,069 rows) — Not present in repository.
- Raw upstream input workbooks (`EXP_1.xlsx`, `EXP_2.xlsx`, `EXP_3.xlsx` from HydroGrowNet, and `kolar_tomato_precision_agri_dataset.csv`) are not committed to git, which complies with repository policies on committing large raw sources.

### 3.3 Metadata Files in `image_data/Lettuce_Images/`
- `image_data/Lettuce_Images/manifest.csv` (943 rows, 10 columns: `new_filename`, `class`, `original_filename`, `original_folder`, `was_augmented`, `width`, `height`, `mode_before_conversion`, `sha256`, `status`).
- `image_data/Lettuce_Images/class_counts.csv` (4 rows, 3 columns: `class`, `final_count`, `from_kept_orphans`).
- These CSV files document original filenames, image dimensions, SHA-256 hashes, and augmentation tracking for the 580 physical lettuce images present in the subdirectories.

---

## 4. Monorepo Layout & Deviations Log (T1-002)

The monorepo directory hierarchy was adapted from the proposed layout in `docs/TECH_RULES.md`:
- `frontend/`
- `backend/app/{api,schemas,services,database,core}`
- `ml/{image_models,sensor_models,segmentation,features,fusion,recommendations,evaluation,registry}`
- `processed_data/{splits,features,sensor,manifests}`
- `models/{image,sensor,manifests}`
- `reports/`
- `notebooks/`
- `scripts/`
- `tests/{unit,integration,contracts,data}`
- `docs/{learning,datasets,decisions}`

### Deviations Logged
1. **Documentation Placement:** `docs/TECH_RULES.md` depicts `PRD.md`, `DESIGN.md`, and `TECH_RULES.md` at root. In this repository, they were already established under `docs/` (`docs/PRD.md`, `docs/DESIGN.md`, `docs/TECH_RULES.md`, `docs/ROADMAP.md`). In accordance with the rule "Never move or duplicate an existing equivalent folder", they are retained in `docs/` without creating duplicate root copies.
2. **Immutable Raw Data:** `image_data/` and `sensor_data/` remain at the root as audited; no duplicate or parallel data directories were introduced.
3. **Fusion Policy:** `ml/fusion/README.md` is committed as a permanent rule file forbidding model training on unpaired image and sensor observations per Overview §19 and TECH_RULES §15.

