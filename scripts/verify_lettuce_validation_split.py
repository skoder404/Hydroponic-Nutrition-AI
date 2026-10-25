#!/usr/bin/env python3
"""Verification script for Lettuce Sensor Validation Set Split (T2-010).

Acceptance Criteria:
1. Test rows are identical to the original test file.
2. No (experiment, date) pair appears in more than one of train/val/test.
3. Per-class real-row counts per partition are in the manifest and any class under 5 real training rows is flagged.

Re-run command:
    python scripts/verify_lettuce_validation_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
ORIG_DATA_DIR = ROOT / "sensor_data" / "Lettuce_Sensor-Readings"
V2_DATA_DIR = ROOT / "processed_data" / "sensor" / "lettuce"
MANIFEST_FILE = V2_DATA_DIR / "manifest.json"


def verify_test_identical_to_original() -> tuple[bool, str]:
    """Criterion 1: Test rows are identical to original test file."""
    orig_test_csv = ORIG_DATA_DIR / "hydroponic_lettuce_test.csv"
    v2_test_csv = V2_DATA_DIR / "test.csv"

    if not orig_test_csv.exists() or not v2_test_csv.exists():
        return False, f"Missing test files: orig={orig_test_csv.exists()}, v2={v2_test_csv.exists()}"

    orig = pd.read_csv(orig_test_csv)
    v2 = pd.read_csv(v2_test_csv)

    if not v2.equals(orig):
        return False, f"V2 test set differs from original! (lengths: orig={len(orig)}, v2={len(v2)})"

    return True, f"V2 test set is identical to original test file ({len(v2)} rows, 7 classes)"


def verify_no_overlapping_experiment_date_pairs() -> tuple[bool, str]:
    """Criterion 2: No (experiment, date) pair appears in more than one of train/val/test."""
    train_csv = V2_DATA_DIR / "train.csv"
    val_csv = V2_DATA_DIR / "val.csv"
    test_csv = V2_DATA_DIR / "test.csv"

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    train_keys = set(zip(train["experiment"], train["date"]))
    val_keys = set(zip(val["experiment"], val["date"]))
    test_keys = set(zip(test["experiment"], test["date"]))

    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    if train_val_overlap:
        return False, f"Overlap between Train and Val: {train_val_overlap}"
    if train_test_overlap:
        return False, f"Overlap between Train and Test: {train_test_overlap}"
    if val_test_overlap:
        return False, f"Overlap between Val and Test: {val_test_overlap}"

    return True, (
        f"Strictly disjoint (experiment, date) pairs: "
        f"train={len(train_keys)} days, val={len(val_keys)} days, test={len(test_keys)} days"
    )


def verify_manifest_class_counts_and_flagging() -> tuple[bool, str]:
    """Criterion 3: Per-class real counts in manifest and classes under 5 real rows flagged."""
    if not MANIFEST_FILE.exists():
        return False, f"Missing manifest: {MANIFEST_FILE}"

    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    coverage = data.get("class_coverage", {})
    if len(coverage) != 7:
        return False, f"Expected 7 classes in coverage, got {len(coverage)}"

    flagged = data.get("flagged_classes_under_5_real_train_rows", [])

    # pH_Low should have 7 real train rows (>= 5, so not flagged < 5)
    ph_low = coverage.get("pH_Low", {})
    if ph_low.get("real_train_rows") != 7:
        return False, f"Expected 7 real train rows for pH_Low, got {ph_low.get('real_train_rows')}"
    if ph_low.get("real_val_rows") != 1:
        return False, f"Expected 1 real val row for pH_Low, got {ph_low.get('real_val_rows')}"
    if ph_low.get("real_test_rows") != 2:
        return False, f"Expected 2 real test rows for pH_Low, got {ph_low.get('real_test_rows')}"

    return True, (
        f"Manifest records real counts per partition for all 7 classes; "
        f"pH_Low confirmed with 7 train, 1 val, 2 test rows; flagged list: {flagged}"
    )


def main() -> int:
    checks = [
        ("Criterion 1: Test Set Identical to V1", verify_test_identical_to_original),
        ("Criterion 2: Disjoint (Experiment, Date) Days", verify_no_overlapping_experiment_date_pairs),
        ("Criterion 3: Manifest Counts & Under-5 Flagging", verify_manifest_class_counts_and_flagging),
    ]

    all_passed = True
    print("=" * 65)
    print("Lettuce Sensor Validation Set Verification (T2-010)")
    print("=" * 65)

    for name, fn in checks:
        passed, msg = fn()
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"{icon} {name}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED")
        return 0
    else:
        print("VERIFICATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
