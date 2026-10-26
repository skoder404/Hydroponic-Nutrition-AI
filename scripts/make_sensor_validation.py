#!/usr/bin/env python3
"""Generate partitioned train, validation, and test sensor datasets (v2).

Supports grouped holdout validation splits to prevent temporal and replicate autocorrelation leakage.

Usage:
    python scripts/make_sensor_validation.py lettuce
    python scripts/make_sensor_validation.py <crop>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Lettuce Setup (T2-010)
# ---------------------------------------------------------------------------
LETTUCE_DATA_DIR = ROOT / "sensor_data" / "Lettuce_Sensor-Readings"
LETTUCE_OUTPUT_DIR = ROOT / "processed_data" / "sensor" / "lettuce"

LETTUCE_KEPT_CLASSES = [
    "EC_High",
    "Humidity_Low",
    "Multiple_Stress",
    "Normal",
    "WaterTemp_High",
    "pH_High",
    "pH_Low",
]
LETTUCE_DROPPED_CLASSES = ["EC_Low", "WaterTemp_Low"]


def balance_training_set(
    train_df: pd.DataFrame,
    seed: int = 42,
    min_keep: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    """Oversample minority classes to the majority count in the training set only."""
    counts = train_df["label"].value_counts()
    counts = counts[counts >= min_keep]
    target = int(counts.max())
    rng = np.random.RandomState(seed)
    parts = []
    for cls, n in counts.items():
        sub = train_df[train_df["label"] == cls]
        if n < target:
            extra = sub.sample(target - n, replace=True, random_state=rng)
            sub = pd.concat([sub, extra])
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out, sorted(counts.index)


def make_lettuce_validation(output_dir: Path | None = None) -> tuple[int, dict]:
    """Create grouped holdout train, val, and test partitions for lettuce (v2)."""
    out_dir = output_dir or LETTUCE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = LETTUCE_DATA_DIR / "hydroponic_lettuce_full.csv"
    orig_test_csv = LETTUCE_DATA_DIR / "hydroponic_lettuce_test.csv"

    if not full_csv.exists() or not orig_test_csv.exists():
        print(f"Error: Missing input files in {LETTUCE_DATA_DIR}")
        return 1, {}

    full = pd.read_csv(full_csv)
    orig_test = pd.read_csv(orig_test_csv)

    # 1. Reconstruct test set: every 4th calendar day of each experiment
    test_parts, train_raw_parts = [], []
    for exp, sub in full.groupby("experiment"):
        days = sorted(sub["date"].unique())
        test_days = set(days[3::4])
        test_parts.append(sub[sub["date"].isin(test_days)])
        train_raw_parts.append(sub[~sub["date"].isin(test_days)])

    test_raw = pd.concat(test_parts).reset_index(drop=True)
    train_raw_initial = pd.concat(train_raw_parts).reset_index(drop=True)

    # Filter to kept classes
    train_pool = train_raw_initial[train_raw_initial["label"].isin(LETTUCE_KEPT_CLASSES)].reset_index(drop=True)
    test = test_raw[test_raw["label"].isin(LETTUCE_KEPT_CLASSES)].reset_index(drop=True)

    # Verify test set exactly matches original test file
    if not test.equals(orig_test):
        print("Error: Reconstructed test split does not exactly match original test set!")
        return 1, {}

    # 2. Hold out every 4th remaining day of each experiment as validation
    val_parts, new_train_parts = [], []
    for exp, sub in train_pool.groupby("experiment"):
        days = sorted(sub["date"].unique())
        val_days = set(days[3::4])
        val_parts.append(sub[sub["date"].isin(val_days)])
        new_train_parts.append(sub[~sub["date"].isin(val_days)])

    val = pd.concat(val_parts).reset_index(drop=True)
    new_train_raw = pd.concat(new_train_parts).reset_index(drop=True)

    # Check that (experiment, date) pairs are strictly disjoint across partitions
    train_keys = set(zip(new_train_raw["experiment"], new_train_raw["date"]))
    val_keys = set(zip(val["experiment"], val["date"]))
    test_keys = set(zip(test["experiment"], test["date"]))

    assert len(train_keys & val_keys) == 0, "Train and Val share (experiment, date) keys!"
    assert len(train_keys & test_keys) == 0, "Train and Test share (experiment, date) keys!"
    assert len(val_keys & test_keys) == 0, "Val and Test share (experiment, date) keys!"

    # 3. Apply class balancing (oversampling to majority count) to the new training pool only
    train_balanced, _kept = balance_training_set(new_train_raw, seed=42, min_keep=7)

    # 4. Write CSV files (v2)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_balanced.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    shutil.copyfile(orig_test_csv, test_path)

    print(f"Wrote {train_path} ({len(train_balanced)} rows, balanced)")
    print(f"Wrote {val_path} ({len(val)} rows, real un-augmented)")
    print(f"Wrote {test_path} ({len(test)} rows, identical to v1 test)")

    # 5. Build Manifest
    real_counts_train = new_train_raw["label"].value_counts().to_dict()
    real_counts_val = val["label"].value_counts().to_dict()
    real_counts_test = test["label"].value_counts().to_dict()

    class_coverage: dict[str, dict] = {}
    flagged_under_5: list[str] = []

    for cls in LETTUCE_KEPT_CLASSES:
        n_train = real_counts_train.get(cls, 0)
        n_val = real_counts_val.get(cls, 0)
        n_test = real_counts_test.get(cls, 0)
        is_low = n_train < 5
        if is_low:
            flagged_under_5.append(cls)
        class_coverage[cls] = {
            "real_train_rows": n_train,
            "balanced_train_rows": int(train_balanced["label"].value_counts().get(cls, 0)),
            "real_val_rows": n_val,
            "real_test_rows": n_test,
            "total_real_rows": n_train + n_val + n_test,
            "flagged_under_5_real_train_rows": is_low,
        }

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "dataset_version": "v2",
        "crop": "lettuce",
        "modality": "sensor",
        "generated_at": now_iso,
        "script": "scripts/make_sensor_validation.py lettuce",
        "partition_strategy": {
            "strategy": "day_grouped_holdout",
            "group_keys": ["experiment", "date"],
            "test_selection": "every 4th calendar day of each experiment (16 experiment-days, 162 rows)",
            "validation_selection": "every 4th remaining calendar day of each experiment (12 experiment-days, 125 rows)",
            "train_selection": "remaining calendar days (41 experiment-days, 430 real rows, oversampled to 1,295 rows)",
            "validation_fraction_of_pool": round(len(val) / (len(val) + len(new_train_raw)), 4),
            "no_leakage_verified": True,
        },
        "classes_kept": LETTUCE_KEPT_CLASSES,
        "classes_dropped": LETTUCE_DROPPED_CLASSES,
        "row_counts": {
            "train_balanced": len(train_balanced),
            "train_real": len(new_train_raw),
            "val_real": len(val),
            "test_real": len(test),
            "full_v1": len(full),
        },
        "experiment_date_counts": {
            "train_days": len(train_keys),
            "val_days": len(val_keys),
            "test_days": len(test_keys),
            "keys_overlap": 0,
        },
        "class_coverage": class_coverage,
        "flagged_classes_under_5_real_train_rows": flagged_under_5,
        "special_class_notes": {
            "pH_Low": "pH_Low has 10 total real rows in v1 (7 in train, 1 in val, 2 in test). As confirmed by team decision, it is retained as a distinct class with 7 real training rows oversampled to 185."
        },
        "limitations": [
            "Neighbouring days of the same tank are correlated; day-grouped validation is more realistic than random row splits, but remains somewhat optimistic compared with contiguous whole-replicate or multi-week block holdout.",
            "pH_Low has only 7 real rows in the training pool (oversampled to 185). Models evaluating pH_Low must be interpreted with caution regarding sample diversity."
        ],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")

    return 0, manifest


# ---------------------------------------------------------------------------
# Spinach Setup (T2-011)
# ---------------------------------------------------------------------------
SPINACH_DATA_DIR = ROOT / "sensor_data" / "Spinach_Sensor-Readings"
SPINACH_OUTPUT_DIR = ROOT / "processed_data" / "sensor" / "spinach"

SPINACH_CLASSES = [
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "Normal",
    "Water_Temp_High",
    "pH_Imbalance",
]


def compute_spinach_blocks(df: pd.DataFrame) -> pd.Series:
    """Build 6-hour temporal block key from bench, date, and hour//6 of source_timestamp_EDT."""
    ts = pd.to_datetime(df["source_timestamp_EDT"])
    return df["bench"] + "|" + ts.dt.strftime("%Y-%m-%d") + "|" + (ts.dt.hour // 6).astype(str)


def make_spinach_validation(
    output_dir: Path | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[int, dict]:
    """Create grouped 6-hour block holdout train, val, and test partitions for spinach (v2)."""
    out_dir = output_dir or SPINACH_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_train_csv = SPINACH_DATA_DIR / "spinach_7class_train.csv"
    orig_test_csv = SPINACH_DATA_DIR / "spinach_7class_test.csv"
    orig_full_csv = SPINACH_DATA_DIR / "spinach_7class_full.csv"

    if not orig_train_csv.exists() or not orig_test_csv.exists():
        print(f"Error: Missing input files in {SPINACH_DATA_DIR}")
        return 1, {}

    train_raw = pd.read_csv(orig_train_csv)
    test_raw = pd.read_csv(orig_test_csv)
    full_len = len(pd.read_csv(orig_full_csv)) if orig_full_csv.exists() else 1400

    # Derive block keys
    train_raw["block"] = compute_spinach_blocks(train_raw)
    test_raw["block"] = compute_spinach_blocks(test_raw)

    test_blocks = set(test_raw["block"])
    all_train_blocks = sorted(train_raw["block"].unique())

    # Verify no block overlap between original train and test
    orig_overlap = set(all_train_blocks).intersection(test_blocks)
    if orig_overlap:
        print(f"Error: Original train and test share blocks: {orig_overlap}")
        return 1, {}

    # Hold out whole blocks (~15% of training rows) as validation
    target_rows = round(val_fraction * len(train_raw))
    block_sizes = train_raw["block"].value_counts().to_dict()

    rng = np.random.RandomState(seed)
    shuffled_blocks = list(all_train_blocks)
    rng.shuffle(shuffled_blocks)

    val_blocks: list[str] = []
    acc = 0
    best_diff = float("inf")
    best_blocks: list[str] = []

    for b in shuffled_blocks:
        val_blocks.append(b)
        acc += block_sizes[b]
        diff = abs(acc - target_rows)
        if diff < best_diff:
            best_diff = diff
            best_blocks = list(val_blocks)
        if acc >= target_rows and diff > best_diff:
            break

    selected_val_blocks = set(best_blocks)
    val = train_raw[train_raw["block"].isin(selected_val_blocks)].drop(columns=["block"]).reset_index(drop=True)
    new_train = train_raw[~train_raw["block"].isin(selected_val_blocks)].drop(columns=["block"]).reset_index(drop=True)
    test = test_raw.drop(columns=["block"]).reset_index(drop=True)

    # Re-verify disjoint block keys
    new_train_blocks = set(train_raw[~train_raw["block"].isin(selected_val_blocks)]["block"])
    assert len(new_train_blocks & selected_val_blocks) == 0, "Train and Val share block keys!"
    assert len(new_train_blocks & test_blocks) == 0, "Train and Test share block keys!"
    assert len(selected_val_blocks & test_blocks) == 0, "Val and Test share block keys!"

    # Re-verify sample ID disjointness
    train_ids = set(new_train["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])
    assert len(train_ids & val_ids) == 0, "Train and Val share sample IDs!"
    assert len(train_ids & test_ids) == 0, "Train and Test share sample IDs!"
    assert len(val_ids & test_ids) == 0, "Val and Test share sample IDs!"

    # Write CSV files (v2)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    new_train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    shutil.copyfile(orig_test_csv, test_path)

    print(f"Wrote {train_path} ({len(new_train)} rows, {len(new_train_blocks)} blocks, unresampled)")
    print(f"Wrote {val_path} ({len(val)} rows, {len(selected_val_blocks)} blocks, {len(val)/len(train_raw):.2%})")
    print(f"Wrote {test_path} ({len(test)} rows, {len(test_blocks)} blocks, identical to v1 test)")

    # Build manifest
    train_counts = new_train["label"].value_counts().to_dict()
    val_counts = val["label"].value_counts().to_dict()
    test_counts = test["label"].value_counts().to_dict()

    class_coverage: dict[str, dict] = {}
    for cls in SPINACH_CLASSES:
        n_train = int(train_counts.get(cls, 0))
        n_val = int(val_counts.get(cls, 0))
        n_test = int(test_counts.get(cls, 0))
        class_coverage[cls] = {
            "train_rows": n_train,
            "val_rows": n_val,
            "test_rows": n_test,
            "total_rows": n_train + n_val + n_test,
            "train_fraction": round(n_train / len(new_train), 4),
            "val_fraction": round(n_val / len(val), 4),
            "test_fraction": round(n_test / len(test), 4),
        }

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "dataset_version": "v2",
        "crop": "spinach",
        "modality": "sensor",
        "generated_at": now_iso,
        "script": "scripts/make_sensor_validation.py spinach",
        "partition_strategy": {
            "strategy": "six_hour_block_grouped_holdout",
            "block_definition": "bench + '|' + source_timestamp_EDT (YYYY-MM-DD) + '|' + (hour // 6)",
            "test_selection": f"22 whole 6-hour blocks from original 80/20 split ({len(test)} rows, untouched)",
            "validation_selection": f"{len(selected_val_blocks)} whole 6-hour blocks held out from original train split ({len(val)} rows, {len(val)/len(train_raw):.2%} of training pool)",
            "train_selection": f"{len(new_train_blocks)} remaining whole 6-hour blocks ({len(new_train)} rows)",
            "seed": seed,
            "target_validation_fraction": val_fraction,
            "actual_validation_fraction": round(len(val) / len(train_raw), 4),
            "no_leakage_verified": True,
            "resampling_applied": False,
        },
        "classes": SPINACH_CLASSES,
        "row_counts": {
            "train": len(new_train),
            "val": len(val),
            "test": len(test),
            "total_v2": len(new_train) + len(val) + len(test),
            "total_v1_full": full_len,
        },
        "block_counts": {
            "train_blocks": len(new_train_blocks),
            "val_blocks": len(selected_val_blocks),
            "test_blocks": len(test_blocks),
            "keys_overlap": 0,
        },
        "class_coverage": class_coverage,
        "limitations": [
            "Chemistry values (pH, EC, water temperature) are simulated based on real greenhouse air temperature and relative humidity logs.",
            "Labels are derived from deterministic threshold rules; model performance on this dataset may be artificially high compared to real-world nutrient solution dynamics.",
            "Data was originally balanced by physical drawing, not oversampling; training row counts per class drop below 160 (ranging between 127 and 146) because blocks contain heterogeneous classes and are held out whole without resampling.",
        ],
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")

    return 0, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create sensor validation split.")
    parser.add_argument("crop", choices=["lettuce", "spinach", "tomato"], help="Target crop")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory")
    args = parser.parse_args()

    if args.crop == "lettuce":
        code, _ = make_lettuce_validation(args.output_dir)
        return code
    elif args.crop == "spinach":
        code, _ = make_spinach_validation(args.output_dir)
        return code
    else:
        print(f"Crop '{args.crop}' validation generator not yet implemented.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
