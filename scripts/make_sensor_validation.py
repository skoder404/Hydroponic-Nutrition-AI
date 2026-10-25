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
    import shutil
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Create sensor validation split.")
    parser.add_argument("crop", choices=["lettuce", "spinach", "tomato"], help="Target crop")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory")
    args = parser.parse_args()

    if args.crop == "lettuce":
        code, _ = make_lettuce_validation(args.output_dir)
        return code
    else:
        print(f"Crop '{args.crop}' validation generator not yet implemented.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
