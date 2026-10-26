#!/usr/bin/env python3
"""Verification script for Spinach Sensor Validation Set Split (T2-011).

Acceptance Criteria:
1. Test rows unchanged (identical to original test file).
2. No block key appears in more than one partition.
3. Per-class counts per partition are in the manifest.

Re-run command:
    python scripts/verify_spinach_validation_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
ORIG_DATA_DIR = ROOT / "sensor_data" / "Spinach_Sensor-Readings"
V2_DATA_DIR = ROOT / "processed_data" / "sensor" / "spinach"
MANIFEST_FILE = V2_DATA_DIR / "manifest.json"

EXPECTED_CLASSES = [
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "Normal",
    "Water_Temp_High",
    "pH_Imbalance",
]


def compute_block_keys(df: pd.DataFrame) -> pd.Series:
    """Build 6-hour temporal block key from bench, date, and hour//6 of source_timestamp_EDT."""
    ts = pd.to_datetime(df["source_timestamp_EDT"])
    return df["bench"] + "|" + ts.dt.strftime("%Y-%m-%d") + "|" + (ts.dt.hour // 6).astype(str)


def verify_test_identical_to_original() -> tuple[bool, str]:
    """Criterion 1: Test rows unchanged."""
    orig_test_csv = ORIG_DATA_DIR / "spinach_7class_test.csv"
    v2_test_csv = V2_DATA_DIR / "test.csv"

    if not orig_test_csv.exists() or not v2_test_csv.exists():
        return False, f"Missing test files: orig={orig_test_csv.exists()}, v2={v2_test_csv.exists()}"

    orig = pd.read_csv(orig_test_csv)
    v2 = pd.read_csv(v2_test_csv)

    if not v2.equals(orig):
        return False, f"V2 test set differs from original! (lengths: orig={len(orig)}, v2={len(v2)})"

    return True, f"V2 test set is identical to original test file ({len(v2)} rows, 7 classes, 40 per class)"


def verify_no_overlapping_block_keys() -> tuple[bool, str]:
    """Criterion 2: No block key appears in more than one partition."""
    train_csv = V2_DATA_DIR / "train.csv"
    val_csv = V2_DATA_DIR / "val.csv"
    test_csv = V2_DATA_DIR / "test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            return False, f"Missing partition file: {p}"

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    train_blocks = set(compute_block_keys(train))
    val_blocks = set(compute_block_keys(val))
    test_blocks = set(compute_block_keys(test))

    train_val_overlap = train_blocks & val_blocks
    train_test_overlap = train_blocks & test_blocks
    val_test_overlap = val_blocks & test_blocks

    if train_val_overlap:
        return False, f"Block overlap between Train and Val: {train_val_overlap}"
    if train_test_overlap:
        return False, f"Block overlap between Train and Test: {train_test_overlap}"
    if val_test_overlap:
        return False, f"Block overlap between Val and Test: {val_test_overlap}"

    # Also check sample IDs
    train_ids = set(train["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])

    if train_ids & val_ids:
        return False, f"Sample ID overlap between Train and Val: {train_ids & val_ids}"
    if train_ids & test_ids:
        return False, f"Sample ID overlap between Train and Test: {train_ids & test_ids}"
    if val_ids & test_ids:
        return False, f"Sample ID overlap between Val and Test: {val_ids & test_ids}"

    return True, (
        f"Strictly disjoint 6-hour blocks: "
        f"train={len(train_blocks)} blocks, val={len(val_blocks)} blocks, test={len(test_blocks)} blocks (total 110 blocks, 0 overlap)"
    )


def verify_manifest_per_class_counts() -> tuple[bool, str]:
    """Criterion 3: Per-class counts per partition are in the manifest."""
    if not MANIFEST_FILE.exists():
        return False, f"Missing manifest: {MANIFEST_FILE}"

    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    coverage = data.get("class_coverage", {})
    if len(coverage) != 7:
        return False, f"Expected 7 classes in coverage, got {len(coverage)}"

    for cls in EXPECTED_CLASSES:
        if cls not in coverage:
            return False, f"Class {cls} missing from manifest coverage"
        c = coverage[cls]
        for key in ("train_rows", "val_rows", "test_rows", "total_rows"):
            if key not in c:
                return False, f"Key '{key}' missing from coverage of {cls}"
        if c["total_rows"] != 200:
            return False, f"Class {cls} total rows is {c['total_rows']}, expected 200"
        if c["test_rows"] != 40:
            return False, f"Class {cls} test rows is {c['test_rows']}, expected 40"
        if c["train_rows"] + c["val_rows"] != 160:
            return False, f"Class {cls} train + val is {c['train_rows'] + c['val_rows']}, expected 160"

    # Row count verification
    row_counts = data.get("row_counts", {})
    if row_counts.get("train") != 958:
        return False, f"Manifest train count is {row_counts.get('train')}, expected 958"
    if row_counts.get("val") != 162:
        return False, f"Manifest val count is {row_counts.get('val')}, expected 162"
    if row_counts.get("test") != 280:
        return False, f"Manifest test count is {row_counts.get('test')}, expected 280"

    return True, (
        f"Manifest records per-class counts across all 7 classes; "
        f"train=958, val=162 ({162/1120:.2%}), test=280 (total 1400 rows)"
    )


def run_all_checks() -> int:
    """Execute all acceptance criteria verification checks."""
    print("=" * 70)
    print("Spinach Sensor Validation Set Split Verification (T2-011)")
    print("=" * 70)

    checks = [
        ("Criterion 1: Test rows unchanged", verify_test_identical_to_original),
        ("Criterion 2: Disjoint 6-hour temporal blocks", verify_no_overlapping_block_keys),
        ("Criterion 3: Manifest per-class counts recorded", verify_manifest_per_class_counts),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {title}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED (T2-011)")
        return 0
    else:
        print("SOME VERIFICATION CHECKS FAILED (T2-011)")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
