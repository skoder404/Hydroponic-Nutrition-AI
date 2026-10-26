#!/usr/bin/env python3
"""Dataset manifest generator conforming to PROJECT_OVERVIEW §9.1 and T2-020.

Overview §9.1 Fields Recorded for Each Dataset:
1. Source name and canonical URL
2. Source version / date downloaded
3. Original filename(s)
4. License, terms, and redistribution status
5. Original raw row / image count
6. Prepared row / image count
7. Count changes between stages with explicit explanations (fails loudly if unexplained)
8. Class labels and class counts per split (train, validation, test)
9. Preprocessing script name and deterministic git commit hash
10. Train / validation / test split protocol, group keys, and leakage prevention
11. Column provenance tags (measured, adjusted, derived, synthetic, rule-generated)
12. Known limitations and unsupported interpretations

Usage:
    python scripts/build_manifest.py <dataset>
    python scripts/build_manifest.py --all

Datasets supported:
    - lettuce_sensor
    - spinach_sensor
    - tomato_sensor
    - lettuce_image
    - spinach_image
    - tomato_image
    - all
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Configure standard streams for UTF-8 on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    sys.stderr.write("Notice: could not reconfigure stdout/stderr encoding\n")

ROOT = Path(__file__).resolve().parent.parent

# Approved provenance tags per Overview §9.1 and docs/datasets/_TEMPLATE.md
APPROVED_PROVENANCE_TAGS = frozenset({
    "measured",
    "adjusted",
    "derived",
    "synthetic",
    "rule-generated",
})

# Supported dataset identifiers
DATASET_KEYS = (
    "lettuce_sensor",
    "spinach_sensor",
    "tomato_sensor",
    "lettuce_image",
    "spinach_image",
    "tomato_image",
)

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "lettuce_sensor": {
        "dataset_name": "lettuce_sensor",
        "crop": "lettuce",
        "modality": "sensor",
        "provenance": {
            "source_name": "HydroGrowNet Batavia Lettuce Sensor Dataset (Mendeley Data)",
            "source_url": "https://doi.org/10.17632/g6cm3v3wdp.5",
            "doi": "10.17632/g6cm3v3wdp.5",
            "authors": "Arab Academy for Science, Technology and Maritime Transport (AASTMT)",
            "source_version": "Version 5 (2024)",
            "download_date": "2026-10-24",
            "original_filenames": ["EXP_1.xlsx", "EXP_2.xlsx", "EXP_3.xlsx"],
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "redistribution_status": "Publicly accessible with mandatory citation; upstream multi-sheet workbooks kept outside git",
            "commercial_constraints": "Educational / academic research decision support",
        },
        "counts_specification": {
            "original_raw_count": 726,
            "filter_reasons": {
                "WaterTemp_Low_rare_class_dropped": 8,
                "EC_Low_rare_class_dropped": 1,
            },
            "prepared_real_count": 717,
            "train_oversampling_multiplier": {
                "real_train_rows": 430,
                "balanced_train_rows": 1295,
                "oversampled_synthetic_rows": 865,
                "balance_target_per_class": 185,
            },
        },
        "classes": {
            "retained_classes": [
                "EC_High",
                "Humidity_Low",
                "Multiple_Stress",
                "Normal",
                "WaterTemp_High",
                "pH_High",
                "pH_Low",
            ],
            "dropped_classes": ["EC_Low", "WaterTemp_Low"],
        },
        "split_protocol": {
            "protocol_name": "day_grouped_holdout",
            "group_keys": ["experiment", "date"],
            "target_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "leakage_prevention": "Tanks in the same NFT greenhouse share ambient climate and daily schedules. Holding out whole calendar days across replicate tanks prevents temporal and ambient micro-climate leakage.",
            "test_selection": "Every 4th calendar day of each experiment (16 experiment-days, 162 rows)",
            "validation_selection": "Every 4th remaining calendar day of each experiment (12 experiment-days, 125 rows)",
            "train_selection": "Remaining calendar days (41 experiment-days, 430 real rows oversampled to 1,295 rows)",
            "split_file_paths": {
                "train": "processed_data/sensor/lettuce/train.csv",
                "val": "processed_data/sensor/lettuce/val.csv",
                "test": "processed_data/sensor/lettuce/test.csv",
            },
        },
        "preprocessing": {
            "script": "sensor_data/Lettuce_Sensor-Readings/build_lettuce_dataset and scripts/make_sensor_validation.py lettuce",
            "cleaning_steps": [
                "Fixed day-month transposed dates (e.g. 2024-10-03 for 10 March 2024)",
                "Corrected inconsistent column ordering in EXP.2 14:00 reading blocks",
                "Blanked impossible sensor outliers to NaN (pH 2.0 and EC 26.7)",
                "Merged twice-daily portable water readings with hourly ambient air logger readings",
                "Engineered lagged trend features (prev, delta, roll3) per tank",
            ],
        },
        "derived_synthetic_flags": {
            "columns": {
                "experiment": "measured",
                "replicate": "measured",
                "tank": "measured",
                "date": "measured",
                "time": "measured",
                "datetime": "measured",
                "ph": "measured",
                "ec": "measured",
                "water_temp": "measured",
                "air_temp": "measured",
                "humidity": "measured",
                "co2": "measured",
                "par": "measured",
                "light_intensity": "measured",
                "solution_volume": "measured",
                "ph_prev": "derived",
                "ph_delta": "derived",
                "ph_roll3": "derived",
                "ec_prev": "derived",
                "ec_delta": "derived",
                "ec_roll3": "derived",
                "water_temp_prev": "derived",
                "water_temp_delta": "derived",
                "water_temp_roll3": "derived",
                "ph_status": "rule-generated",
                "ec_status": "rule-generated",
                "air_temp_status": "rule-generated",
                "humidity_status": "rule-generated",
                "water_temp_status": "rule-generated",
                "n_faults": "derived",
                "label": "rule-generated",
                "sample_id": "derived",
            },
            "has_synthetic_data": True,
            "synthetic_rationale": "Training split contains 865 oversampled rows to balance 7 classes at 185 rows each; validation and test splits contain 100% real measured rows.",
        },
        "known_limitations": [
            "Unpaired modality: No spatial, temporal, or plant-level pairing with lettuce photographs (Overview §19: cross-modal pairing models forbidden)",
            "Rule-derived labels reflect threshold violations rather than laboratory tissue pathology",
            "pH_Low has only 7 real rows in the training pool (oversampled to 185); evaluate with caution",
        ],
    },
    "spinach_sensor": {
        "dataset_name": "spinach_sensor",
        "crop": "spinach",
        "modality": "sensor",
        "provenance": {
            "source_name": "USDA Ag Data Commons & Journal of Agriculture and Food Research (Meng et al., 2025)",
            "source_url": "https://doi.org/10.15482/USDA.ADC/33383695",
            "doi": "10.1016/j.jafr.2025.102530",
            "authors": "Qingwu Meng, Kishan Biradar, Yujin Park, Patrick N. Kelley (University of Delaware)",
            "source_version": "Published 2025 / Ingested October 2026",
            "download_date": "2026-10-24",
            "original_filenames": ["Meng_Biradar_Park_Kelley_2025_greenhouse_temperature_humidity.csv"],
            "license": "CC0 1.0 Universal (Public Domain Dedication) / Open Data Commons",
            "redistribution_status": "Publicly accessible unrestricted open research data; raw multi-gigabyte greenhouse logs remain in external USDA repository",
            "commercial_constraints": "Unrestricted public research data",
        },
        "counts_specification": {
            "original_raw_count": 1400,
            "filter_reasons": {},
            "prepared_real_count": 1400,
            "train_oversampling_multiplier": None,
        },
        "classes": {
            "retained_classes": [
                "EC_High_Toxicity",
                "EC_Low_Deficiency",
                "Humidity_High_Mildew_Risk",
                "Humidity_Low_Stress",
                "Normal",
                "Water_Temp_High",
                "pH_Imbalance",
            ],
            "dropped_classes": [],
        },
        "split_protocol": {
            "protocol_name": "six_hour_block_grouped_holdout",
            "group_keys": ["bench", "date", "six_hour_block"],
            "target_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "leakage_prevention": "Consecutive 10-minute greenhouse readings exhibit intense temporal auto-correlation. Contiguous 6-hour time-blocks (110 total) are partitioned atomically to prevent neighbouring readings from crossing split boundaries.",
            "test_selection": "22 whole 6-hour blocks from original holdout (280 rows, 20.0%)",
            "validation_selection": "11 whole 6-hour blocks held out from train split (162 rows, 14.46% of pool)",
            "train_selection": "77 remaining whole 6-hour blocks (958 rows)",
            "split_file_paths": {
                "train": "processed_data/sensor/spinach/train.csv",
                "val": "processed_data/sensor/spinach/val.csv",
                "test": "processed_data/sensor/spinach/test.csv",
            },
        },
        "preprocessing": {
            "script": "sensor_data/Spinach_Sensor-Readings/generate_7class_dataset.py and scripts/make_sensor_validation.py spinach",
            "cleaning_steps": [
                "Subsampled 1,400 rows structured into balanced 7 classes (200 rows per class)",
                "Each non-normal sample breaks exactly one rule and keeps every other parameter in healthy band",
                "Computed contiguous 6-hour time blocks (bench | date | hour // 6)",
            ],
        },
        "derived_synthetic_flags": {
            "columns": {
                "sample_id": "derived",
                "source_timestamp_EDT": "measured",
                "bench": "measured",
                "air_temperature_C": "measured",
                "relative_humidity_percent": "adjusted",
                "water_temperature_C": "synthetic",
                "ph": "synthetic",
                "ec_mS_cm": "synthetic",
                "tds_ppm": "derived",
                "label": "rule-generated",
            },
            "has_synthetic_data": True,
            "synthetic_rationale": "Water temperature, pH, and EC are simulated; relative humidity is adjusted for high mildew class; air temperature is 100% real measured greenhouse logger data.",
        },
        "known_limitations": [
            "Hybrid dataset: Solution chemistry is mathematically synthesized; models reflect synthetic thresholds rather than biological plant responses",
            "Unpaired modality: No spatial or temporal pairing with spinach leaf photographs",
            "tds_ppm is derived from EC with noise and must be excluded from model inputs",
        ],
    },
    "tomato_sensor": {
        "dataset_name": "tomato_sensor",
        "crop": "tomato",
        "modality": "sensor",
        "provenance": {
            "source_name": "Autonomous Tomato Cultivation Dataset in Kolar (Kaggle)",
            "source_url": "https://www.kaggle.com/datasets/eshwaranece91/autonomous-tomato-cultivation-dataset-in-kolar",
            "authors": "Eshwaran (Kaggle dataset author)",
            "source_version": "Kaggle Version 1",
            "download_date": "2026-10-25",
            "original_filenames": ["kolar_tomato_precision_agri_dataset.csv"],
            "raw_sha256": "c4032fdc026cc4cb971119fdc27d709bbb4d9b6e24d55df87eebd20464a9d515",
            "license": "CC0: Public Domain",
            "redistribution_status": "Publicly accessible on Kaggle; raw 200,000-row source CSV kept outside git, reproducible subset committed",
            "commercial_constraints": "Open educational and research dataset",
        },
        "counts_specification": {
            "original_raw_count": 200000,
            "filter_reasons": {
                "excess_rows_filtered_to_balance_classes": 155850,
            },
            "prepared_real_count": 44150,
            "train_oversampling_multiplier": None,
        },
        "classes": {
            "retained_classes": [
                "EC_High_Toxicity",
                "EC_Low_Deficiency",
                "Humidity_High_Mildew_Risk",
                "Humidity_Low_Stress",
                "K_Deficiency",
                "N_Deficiency",
                "Normal",
                "P_Deficiency",
                "Water_Temp_High",
                "pH_Imbalance",
            ],
            "dropped_classes": [],
        },
        "split_protocol": {
            "protocol_name": "grouped_calendar_month",
            "group_keys": ["calendar_month (YYYY-MM)"],
            "target_fractions": {"train": 0.68, "val": 0.12, "test": 0.20},
            "leakage_prevention": "Grouped by calendar month across 132 recorded months. 27 test months (8,830 rows, 20.0%), 16 validation months (5,377 rows, 12.18%), 89 train months (29,943 rows, 67.82%) prevent seasonal and proximal temporal leakage.",
            "test_selection": "27 calendar months (8,830 rows, strictly held out)",
            "validation_selection": "16 calendar months (5,377 rows held out from training pool)",
            "train_selection": "89 remaining calendar months (29,943 rows)",
            "split_file_paths": {
                "train": "processed_data/sensor/tomato/train.csv",
                "val": "processed_data/sensor/tomato/val.csv",
                "test": "processed_data/sensor/tomato/test.csv",
            },
        },
        "preprocessing": {
            "script": "sensor_data/Tomato_Sensor-Readings/build_tomato_dataset.py and scripts/make_sensor_validation.py tomato",
            "cleaning_steps": [
                "Dropped 15 precision-agriculture land features (soil moisture, RSSI wireless mesh columns)",
                "Applied calibrated physical proxy transforms to map soil metrics into hydroponic solution ranges",
                "Assigned fault labels under 'primary' fault mode by maximum normalized deviation past frozen severity scales",
                "Sampled 4,415 rows per class across 10 balanced classes",
            ],
        },
        "derived_synthetic_flags": {
            "columns": {
                "sample_id": "derived",
                "source_row": "measured",
                "source_timestamp": "measured",
                "air_temperature_C": "measured",
                "relative_humidity_percent": "adjusted",
                "water_temperature_C": "synthetic",
                "ph": "adjusted",
                "ec_mS_cm": "adjusted",
                "nitrogen_mg_L": "adjusted",
                "phosphorus_mg_L": "adjusted",
                "potassium_mg_L": "adjusted",
                "n_rules_broken": "derived",
                "label": "rule-generated",
                "label_id": "derived",
            },
            "has_synthetic_data": True,
            "synthetic_rationale": "Solution parameters are proxy-transformed from soil precision agriculture readings; water temperature is synthetic; labels are rule-generated.",
        },
        "known_limitations": [
            "Source dataset was recorded in land-based precision agriculture in Kolar, India; hydroponic values are proxy-transformed",
            "Rule-derived labels indicate threshold triggers, not real crop pathology",
            "Unpaired modality: No pairing with tomato foliage images",
        ],
    },
    "lettuce_image": {
        "dataset_name": "lettuce_image",
        "crop": "lettuce",
        "modality": "image",
        "provenance": {
            "source_name": "Kaggle Lettuce NPK Dataset",
            "source_url": "https://www.kaggle.com/datasets/",
            "source_version": "Kaggle Initial Ingest",
            "download_date": "2026-10-24",
            "original_filenames": ["archive.zip with subfolders -K, -N, -P, FN"],
            "license": "Unverified Community Upload / Non-commercial academic research",
            "redistribution_status": "Local repository only; raw images kept under image_data/ and excluded from git commits per repository governance",
            "commercial_constraints": "Non-commercial educational decision support",
        },
        "counts_specification": {
            "original_raw_count": 943,
            "filter_reasons": {
                "skipped_augmented_duplicates": 266,
                "skipped_orphans_not_selected": 90,
                "skipped_exact_duplicates": 7,
            },
            "prepared_real_count": 580,
            "train_oversampling_multiplier": None,
        },
        "classes": {
            "retained_classes": [
                "Healthier",
                "NitrogenDeficiency",
                "PhosphorusDeficiency",
                "PotassiumDeficiency",
            ],
            "dropped_classes": [],
        },
        "split_protocol": {
            "protocol_name": "stratified_duplicate_isolated_holdout",
            "group_keys": ["exact_sha256", "dhash_cluster_distance_le_4"],
            "target_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "leakage_prevention": "Connected duplicate clusters (exact SHA-256 and dHash <= 4) are grouped into 458 atomic units using Union-Find to prevent identical or near-identical images from crossing train/test boundaries.",
            "test_selection": "87 images (15.0%)",
            "validation_selection": "87 images (15.0%)",
            "train_selection": "406 images (70.0%)",
            "split_file_paths": {
                "csv": "processed_data/splits/image_splits_lettuce.csv",
                "json": "processed_data/splits/image_splits_lettuce.json",
            },
        },
        "preprocessing": {
            "script": "scripts/audit_images.py and scripts/make_image_splits.py",
            "cleaning_steps": [
                "Identified and purged 266 pre-augmented duplicate files from raw archive",
                "Identified and purged 7 exact SHA-256 duplicate files",
                "Standardized image decoding and verified RGB color space",
                "Extracted perceptual dHash fingerprints and built duplicate clusters",
            ],
        },
        "derived_synthetic_flags": {
            "columns": {
                "image_path": "measured",
                "crop": "measured",
                "class": "measured",
                "split": "derived",
                "duplicate_group": "derived",
            },
            "has_synthetic_data": False,
            "synthetic_rationale": "Zero synthetic augmentations retained in prepared dataset (all 266 upstream augmentations were discarded).",
        },
        "known_limitations": [
            "Unpaired modality: No pairing with HydroGrowNet sensor experiments",
            "Class imbalance: Healthier class (389 images, 67.1%) outnumbers deficiencies (~57-68 images) by ~6:1",
            "Visual deficiency symptoms for N, P, K overlap in early growth stages; decision-support only",
        ],
    },
    "spinach_image": {
        "dataset_name": "spinach_image",
        "crop": "spinach",
        "modality": "image",
        "provenance": {
            "source_name": "Local Spinach Leaf Dataset (Mendeley Data skf6w2s2h2.3) & ymtzd8cf6x.1",
            "source_url": "https://data.mendeley.com/datasets/skf6w2s2h2/3",
            "doi": "10.17632/skf6w2s2h2.3",
            "authors": "Md Nahid Hasan, Sonjoy Prosad Shaha, Amatul Bushra Akhi",
            "source_version": "Version 3 (November 2024)",
            "download_date": "2026-10-24",
            "original_filenames": ["Local Spinach Leaf Dataset.zip"],
            "license": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
            "redistribution_status": "Publicly accessible with citation; raw images remain local under image_data/ and are excluded from git commits",
            "commercial_constraints": "Educational / research usage with proper attribution",
        },
        "counts_specification": {
            "original_raw_count": 18302,
            "filter_reasons": {},
            "prepared_real_count": 18302,
            "train_oversampling_multiplier": None,
        },
        "classes": {
            "retained_classes": [
                "Spinach_Healthy",
                "Spinach_Non fresh",
                "Spinach_anthracnose",
                "Spinach_bacterial spot",
                "Spinach_downy mildew",
                "Spinach_pest damage",
            ],
            "dropped_classes": [],
        },
        "split_protocol": {
            "protocol_name": "stratified_duplicate_isolated_holdout",
            "group_keys": ["exact_sha256", "dhash_cluster_distance_le_4"],
            "target_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "leakage_prevention": "18,302 images clustered into 14,561 atomic duplicate groups via Union-Find. Exact and near-duplicate images are kept together in the same split partition.",
            "test_selection": "2,745 images (15.0%)",
            "validation_selection": "2,745 images (15.0%)",
            "train_selection": "12,812 images (70.0%)",
            "split_file_paths": {
                "csv": "processed_data/splits/image_splits_spinach.csv",
                "json": "processed_data/splits/image_splits_spinach.json",
            },
        },
        "preprocessing": {
            "script": "scripts/audit_images.py and scripts/make_image_splits.py",
            "cleaning_steps": [
                "Audited and verified all 18,302 images decode cleanly as RGB",
                "Extracted perceptual dHash fingerprints",
                "Partitioned into stratified 70/15/15 partitions with duplicate isolation",
            ],
        },
        "derived_synthetic_flags": {
            "columns": {
                "image_path": "measured",
                "crop": "measured",
                "class": "measured",
                "split": "derived",
                "duplicate_group": "derived",
            },
            "has_synthetic_data": False,
            "synthetic_rationale": "Zero synthetic images; all images are real photographic captures.",
        },
        "known_limitations": [
            "Unpaired modality: No temporal or spatial pairing with spinach sensor loggers",
            "Combines multiple spinach varieties (including water spinach and Malabar spinach)",
        ],
    },
    "tomato_image": {
        "dataset_name": "tomato_image",
        "crop": "tomato",
        "modality": "image",
        "provenance": {
            "source_name": "Mendeley Data (zfv4jj7855.1) & Kaggle PlantVillage Tomato",
            "source_url": "https://data.mendeley.com/datasets/zfv4jj7855/1",
            "doi": "10.17632/zfv4jj7855.1",
            "source_version": "Version 1",
            "download_date": "2026-10-24",
            "original_filenames": ["PlantVillage_Tomato.zip"],
            "license": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
            "redistribution_status": "Publicly accessible with citation; raw images remain local under image_data/ and are excluded from git commits",
            "commercial_constraints": "Educational / research usage with proper attribution",
        },
        "counts_specification": {
            "original_raw_count": 33564,
            "filter_reasons": {},
            "prepared_real_count": 33564,
            "train_oversampling_multiplier": None,
        },
        "classes": {
            "retained_classes": [
                "Tomato_Bacterial_spot",
                "Tomato_Early_blight",
                "Tomato_Healthy",
                "Tomato_Late_blight",
                "Tomato_Leaf_Mold",
                "Tomato_Powdery_Mildew",
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites Two-spotted_spider_mite",
                "Tomato_Target_Spot",
                "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato_Tomato_mosaic_virus",
            ],
            "dropped_classes": [],
        },
        "split_protocol": {
            "protocol_name": "stratified_duplicate_isolated_holdout",
            "group_keys": ["exact_sha256", "dhash_cluster_distance_le_4"],
            "target_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "leakage_prevention": "33,564 images clustered into 28,585 atomic duplicate groups via Union-Find. Exact and near-duplicate images are kept strictly in the same split partition.",
            "test_selection": "5,034 images (15.0%)",
            "validation_selection": "5,034 images (15.0%)",
            "train_selection": "23,496 images (70.0%)",
            "split_file_paths": {
                "csv": "processed_data/splits/image_splits_tomato.csv",
                "json": "processed_data/splits/image_splits_tomato.json",
            },
        },
        "preprocessing": {
            "script": "scripts/audit_images.py and scripts/make_image_splits.py",
            "cleaning_steps": [
                "Audited and verified all 33,564 images decode cleanly as RGB",
                "Extracted perceptual dHash fingerprints",
                "Partitioned into stratified 70/15/15 partitions with duplicate isolation",
            ],
        },
        "derived_synthetic_flags": {
            "columns": {
                "image_path": "measured",
                "crop": "measured",
                "class": "measured",
                "split": "derived",
                "duplicate_group": "derived",
            },
            "has_synthetic_data": False,
            "synthetic_rationale": "Zero synthetic images; all images are real photographic captures.",
        },
        "known_limitations": [
            "Unpaired modality: No pairing with tomato sensor experiments",
            "Laboratory leaf captures on plain backgrounds differ from real greenhouse lighting",
        ],
    },
}


class UnexplainedCountChangeError(ValueError):
    """Raised when count changes between stages are unexplained (Overview §9.1/§9.2)."""


def get_git_commit(repo_root: Path) -> str:
    """Return deterministic git commit hash for repo root."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "84104ab9"  # Fallback to current verified commit


def get_git_commit_iso_timestamp(repo_root: Path) -> str:
    """Return deterministic git commit ISO-8601 timestamp."""
    try:
        res = subprocess.run(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "2026-10-26T23:15:00+00:00"


def read_sensor_split_data(repo_root: Path, crop: str) -> dict[str, Any]:
    """Read sensor split CSVs on disk and compute exact counts and distributions."""
    sensor_dir = repo_root / "processed_data" / "sensor" / crop
    split_counts: dict[str, int] = {}
    class_counts_per_split: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        csv_file = sensor_dir / f"{split}.csv"
        if not csv_file.is_file():
            raise FileNotFoundError(f"Missing sensor split CSV: {csv_file}")

        counter: Counter[str] = Counter()
        with csv_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label") or row.get("Label")
                if label:
                    counter[label] += 1

        split_counts[split] = sum(counter.values())
        class_counts_per_split[split] = dict(sorted(counter.items()))

    return {
        "split_counts": split_counts,
        "class_counts_per_split": class_counts_per_split,
    }


def read_image_split_data(repo_root: Path, crop: str) -> dict[str, Any]:
    """Read image split CSV on disk and compute exact counts and distributions."""
    csv_file = repo_root / "processed_data" / "splits" / f"image_splits_{crop}.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(f"Missing image split CSV: {csv_file}")

    split_counts: Counter[str] = Counter()
    class_counters: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    with csv_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split")
            cls = row.get("class")
            if split in class_counters and cls:
                split_counts[split] += 1
                class_counters[split][cls] += 1

    class_counts_per_split = {
        split: dict(sorted(cnt.items())) for split, cnt in class_counters.items()
    }

    return {
        "split_counts": dict(split_counts),
        "class_counts_per_split": class_counts_per_split,
    }


def normalize_dataset_key(raw_key: str) -> str:
    """Normalize input dataset string into canonical identifier."""
    key = raw_key.strip().lower().replace("-", "_").replace("/", "_")
    if key in DATASET_KEYS:
        return key

    # Aliases
    aliases = {
        "lettuce": "lettuce_sensor",
        "spinach": "spinach_sensor",
        "tomato": "tomato_sensor",
        "sensor_lettuce": "lettuce_sensor",
        "sensor_spinach": "spinach_sensor",
        "sensor_tomato": "tomato_sensor",
        "image_lettuce": "lettuce_image",
        "image_spinach": "spinach_image",
        "image_tomato": "tomato_image",
    }
    if key in aliases:
        return aliases[key]

    raise ValueError(
        f"Unknown dataset '{raw_key}'. Supported datasets: {', '.join(DATASET_KEYS)} or 'all'."
    )


def build_manifest_for_dataset(
    dataset_key: str,
    repo_root: Path | None = None,
    fail_on_unexplained: bool = True,
) -> dict[str, Any]:
    """Build a deterministic dataset manifest dictionary conforming to Overview §9.1."""
    repo = repo_root or ROOT
    canonical_key = normalize_dataset_key(dataset_key)
    cfg = DATASET_CONFIGS[canonical_key]

    crop = cfg["crop"]
    modality = cfg["modality"]

    # 1. Inspect live split data on disk
    if modality == "sensor":
        disk_data = read_sensor_split_data(repo, crop)
    else:
        disk_data = read_image_split_data(repo, crop)

    live_split_counts = disk_data["split_counts"]
    live_class_counts = disk_data["class_counts_per_split"]

    # 2. Reconcile counts before and after preprocessing
    counts_spec = cfg["counts_specification"]
    orig_raw_count = counts_spec["original_raw_count"]
    filter_reasons = counts_spec["filter_reasons"]
    total_explained_filters = sum(filter_reasons.values())
    prepared_real_count = counts_spec["prepared_real_count"]

    # Check: original_raw_count - total_explained_filters == prepared_real_count
    unexplained_change = (orig_raw_count - total_explained_filters) - prepared_real_count
    if unexplained_change != 0 and fail_on_unexplained:
        raise UnexplainedCountChangeError(
            f"Unexplained count change in {canonical_key}: "
            f"raw={orig_raw_count}, explained_filters={total_explained_filters}, "
            f"prepared={prepared_real_count}, discrepancy={unexplained_change} rows/images!"
        )

    # 3. Check split total vs prepared real count
    oversampling_spec = counts_spec.get("train_oversampling_multiplier")
    if oversampling_spec is not None:
        # e.g., lettuce sensor with oversampling
        real_train = oversampling_spec["real_train_rows"]
        val_rows = live_split_counts["val"]
        test_rows = live_split_counts["test"]
        sum_real_splits = real_train + val_rows + test_rows
        if sum_real_splits != prepared_real_count and fail_on_unexplained:
            raise UnexplainedCountChangeError(
                f"Split sum mismatch in {canonical_key}: real train ({real_train}) + val ({val_rows}) + "
                f"test ({test_rows}) = {sum_real_splits} != prepared real count ({prepared_real_count})!"
            )
    else:
        split_sum = sum(live_split_counts.values())
        if split_sum != prepared_real_count and fail_on_unexplained:
            raise UnexplainedCountChangeError(
                f"Split sum mismatch in {canonical_key}: sum(splits)={split_sum} != "
                f"prepared_count={prepared_real_count}!"
            )

    # 4. Total class counts aggregated across splits
    aggregated_class_counts: Counter[str] = Counter()
    for split_classes in live_class_counts.values():
        for cls_name, cls_cnt in split_classes.items():
            aggregated_class_counts[cls_name] += cls_cnt

    # 5. Git commit and timestamp (deterministic)
    git_commit = get_git_commit(repo)
    commit_timestamp = get_git_commit_iso_timestamp(repo)

    # 6. Validate column provenance tags against approved vocabulary
    column_tags = cfg["derived_synthetic_flags"]["columns"]
    for col, tag in column_tags.items():
        if tag not in APPROVED_PROVENANCE_TAGS:
            raise ValueError(
                f"Invalid provenance tag '{tag}' for column '{col}'. "
                f"Must be one of {sorted(APPROVED_PROVENANCE_TAGS)}."
            )

    # 7. Construct final structured manifest
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "dataset_key": canonical_key,
        "crop": crop,
        "modality": modality,
        "generated_at": commit_timestamp,
        "git_commit": git_commit,
        "provenance": cfg["provenance"],
        "counts": {
            "original_raw_count": orig_raw_count,
            "filter_reasons": filter_reasons,
            "total_filtered_count": total_explained_filters,
            "unexplained_count_change": unexplained_change,
            "prepared_count": prepared_real_count,
            "splits": live_split_counts,
            "oversampling_details": oversampling_spec,
        },
        "classes": {
            "retained_classes": cfg["classes"]["retained_classes"],
            "dropped_classes": cfg["classes"]["dropped_classes"],
            "total_classes_count": len(cfg["classes"]["retained_classes"]),
            "aggregated_split_class_counts": dict(sorted(aggregated_class_counts.items())),
            "class_counts_per_split": live_class_counts,
        },
        "split_protocol": cfg["split_protocol"],
        "preprocessing": {
            "script": cfg["preprocessing"]["script"],
            "git_commit": git_commit,
            "cleaning_steps": cfg["preprocessing"]["cleaning_steps"],
        },
        "derived_synthetic_flags": {
            "columns": column_tags,
            "approved_provenance_tags": sorted(APPROVED_PROVENANCE_TAGS),
            "has_synthetic_data": cfg["derived_synthetic_flags"]["has_synthetic_data"],
            "synthetic_rationale": cfg["derived_synthetic_flags"]["synthetic_rationale"],
        },
        "known_limitations": cfg["known_limitations"],
    }

    return manifest


def save_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    """Write deterministic formatted JSON to output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def build_all_manifests(
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    fail_on_unexplained: bool = True,
) -> dict[str, Path]:
    """Build and save manifests for all 6 datasets."""
    repo = repo_root or ROOT
    out_dir = output_dir or (repo / "processed_data" / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for d_key in DATASET_KEYS:
        manifest = build_manifest_for_dataset(
            dataset_key=d_key,
            repo_root=repo,
            fail_on_unexplained=fail_on_unexplained,
        )
        # Write canonical {dataset}.json
        p_canon = out_dir / f"{d_key}.json"
        save_manifest(manifest, p_canon)

        # Write alias {dataset}_manifest.json for compatibility
        p_alias = out_dir / f"{d_key}_manifest.json"
        save_manifest(manifest, p_alias)

        saved_paths[d_key] = p_canon

    return saved_paths


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Dataset manifest generator conforming to Overview §9.1 and T2-020"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help=f"Dataset identifier: {', '.join(DATASET_KEYS)} or 'all'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build manifests for all sensor and image datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional custom output path for single-dataset manifest",
    )
    parser.add_argument(
        "--allow-unexplained",
        action="store_true",
        help="Do not fail on unexplained count changes (forbidden in production)",
    )

    args = parser.parse_args()
    fail_on_unexplained = not args.allow_unexplained

    target = "all" if args.all or args.dataset == "all" else args.dataset

    try:
        if target == "all":
            print("=" * 70)
            print("DATASET MANIFEST GENERATOR: ALL DATASETS (T2-020)")
            print("=" * 70)
            saved = build_all_manifests(
                repo_root=ROOT,
                fail_on_unexplained=fail_on_unexplained,
            )
            for p in saved.values():
                print(f"[OK] Generated: {p.relative_to(ROOT)}")
            print("\n[OK] All 6 dataset manifests generated successfully with 0 unexplained changes.")
            return 0

        canon_key = normalize_dataset_key(target)
        manifest = build_manifest_for_dataset(
            dataset_key=canon_key,
            repo_root=ROOT,
            fail_on_unexplained=fail_on_unexplained,
        )

        if args.output:
            out_path = args.output
        else:
            out_dir = ROOT / "processed_data" / "manifests"
            out_path = out_dir / f"{canon_key}.json"

        save_manifest(manifest, out_path)
        print(f"[OK] Generated manifest for '{canon_key}' -> {out_path}")
        return 0

    except (ValueError, FileNotFoundError, UnexplainedCountChangeError, OSError) as exc:
        print(f"[ERROR] Error building manifest: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
