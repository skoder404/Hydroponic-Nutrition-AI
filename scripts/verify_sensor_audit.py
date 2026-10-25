#!/usr/bin/env python3
"""Verification script for Lettuce Sensor Dataset Audit (T2-005).

Acceptance Criteria:
1. Every count from the task description is confirmed or listed as a discrepancy.
2. The test rows' (experiment, date) pairs follow the every-4th-day rule and never appear in train.
3. The number of duplicated rows created by oversampling is quantified.
4. Report exists at reports/data/lettuce_sensor_audit.md with all required findings.

Re-run command:
    python scripts/verify_sensor_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "sensor_data" / "Lettuce_Sensor-Readings"
REPORT_FILE = PROJECT_ROOT / "reports" / "data" / "lettuce_sensor_audit.md"


def verify_counts_and_discrepancies() -> tuple[bool, str]:
    """Criterion 1: Every count confirmed or listed as a discrepancy."""
    full_csv = DATA_DIR / "hydroponic_lettuce_full.csv"
    train_csv = DATA_DIR / "hydroponic_lettuce_train.csv"
    test_csv = DATA_DIR / "hydroponic_lettuce_test.csv"

    for p in (full_csv, train_csv, test_csv):
        if not p.exists():
            return False, f"Missing dataset file: {p}"

    full = pd.read_csv(full_csv)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Full rows: 726
    if len(full) != 726:
        return False, f"Expected 726 full rows, got {len(full)}"

    # Test rows: 162
    if len(test) != 162:
        return False, f"Expected 162 test rows, got {len(test)}"

    # Train rows: 1680 (240 * 7 classes)
    if len(train) != 1680:
        return False, f"Expected 1680 train rows, got {len(train)}"

    # 7 classes in train, 240 each
    train_counts = train["label"].value_counts()
    if len(train_counts) != 7:
        return False, f"Expected 7 classes in train, got {len(train_counts)}"
    for cls, count in train_counts.items():
        if count != 240:
            return False, f"Expected 240 rows for train class '{cls}', got {count}"

    # Dropped classes: WaterTemp_Low (8 in full) and EC_Low (1 in full)
    if "WaterTemp_Low" in train["label"].values or "WaterTemp_Low" in test["label"].values:
        return False, "Dropped class WaterTemp_Low found in train or test"
    if "EC_Low" in train["label"].values or "EC_Low" in test["label"].values:
        return False, "Dropped class EC_Low found in train or test"

    return True, "All dataset row counts, classes, and dropped classes confirmed."


def verify_every_4th_day_and_no_leakage() -> tuple[bool, str]:
    """Criterion 2: Test rows follow every-4th-day rule and never appear in train."""
    full = pd.read_csv(DATA_DIR / "hydroponic_lettuce_full.csv")
    train = pd.read_csv(DATA_DIR / "hydroponic_lettuce_train.csv")
    test = pd.read_csv(DATA_DIR / "hydroponic_lettuce_test.csv")

    full["date"] = pd.to_datetime(full["date"])
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    # Check every-4th-day rule per experiment
    for exp in sorted(full["experiment"].unique()):
        exp_full = full[full["experiment"] == exp]
        exp_test = test[test["experiment"] == exp]

        days = sorted(exp_full["date"].unique())
        expected_test_days = set(days[3::4])
        actual_test_days = set(exp_test["date"].unique())

        if expected_test_days != actual_test_days:
            return False, f"Experiment {exp} test days do not match every-4th-day rule"

    # Check zero overlap (no leakage)
    train_pairs = set(zip(train["experiment"], train["date"]))
    test_pairs = set(zip(test["experiment"], test["date"]))
    overlap = train_pairs & test_pairs
    if overlap:
        return False, f"Leakage detected: {len(overlap)} (experiment, date) pairs in both train and test"

    return True, "Every-4th-day rule verified for all experiments; zero train/test leakage."


def verify_oversampling_quantification() -> tuple[bool, str]:
    """Criterion 3: The number of duplicated rows created by oversampling is quantified."""
    full = pd.read_csv(DATA_DIR / "hydroponic_lettuce_full.csv")
    train = pd.read_csv(DATA_DIR / "hydroponic_lettuce_train.csv")
    test = pd.read_csv(DATA_DIR / "hydroponic_lettuce_test.csv")

    test_dates = set(zip(test["experiment"], test["date"]))
    is_train_day = full.apply(
        lambda r: (r["experiment"], r["date"]) not in test_dates,
        axis=1,
    )
    raw_train = full[is_train_day]
    # Dropped classes < 8 rows in raw training split
    kept_raw_train = raw_train[~raw_train["label"].isin(["WaterTemp_Low", "EC_Low"])]

    # 558 raw train rows, 3 dropped (WaterTemp_Low), 555 real rows kept
    if len(raw_train) != 558:
        return False, f"Expected 558 raw train rows, got {len(raw_train)}"
    if len(kept_raw_train) != 555:
        return False, f"Expected 555 kept raw train rows, got {len(kept_raw_train)}"

    # Oversampling: 1680 - 555 = 1125 duplicate rows created
    duplicates_created = len(train) - len(kept_raw_train)
    if duplicates_created != 1125:
        return False, f"Expected 1125 duplicated rows, got {duplicates_created}"

    return True, f"Oversampling quantified: {duplicates_created} duplicated rows added (555 real -> 1680 balanced)."


def verify_report_file() -> tuple[bool, str]:
    """Criterion 4: Report exists at reports/data/lettuce_sensor_audit.md."""
    if not REPORT_FILE.exists():
        return False, f"Report file missing: {REPORT_FILE}"

    text = REPORT_FILE.read_text(encoding="utf-8")
    required_phrases = [
        "## 1. Row Counts",
        "## 2. Class Distribution",
        "## 3. Action Bands & Label Assignment Rules",
        "## 4. Every-4th-Day Split Rule",
        "## 5. Train/Test Separation (No Data Leakage)",
        "## 6. Oversampling Quantification",
        "## 7. Known Discrepancies",
        "✅ ALL CHECKS PASSED",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            return False, f"Report missing required section: '{phrase}'"

    return True, "Audit report exists and contains all required sections."


def main() -> int:
    print("================================================================================")
    print("VERIFYING LETTUCE SENSOR DATASET AUDIT (T2-005)")
    print("================================================================================\n")

    checks = [
        ("Criterion 1: Row counts, classes, and dropped classes", verify_counts_and_discrepancies),
        ("Criterion 2: Every-4th-day holdout and zero leakage", verify_every_4th_day_and_no_leakage),
        ("Criterion 3: Oversampling quantification", verify_oversampling_quantification),
        ("Criterion 4: Comprehensive audit report on disk", verify_report_file),
    ]

    all_passed = True
    for name, fn in checks:
        passed, msg = fn()
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{status}] {name}")
        print(f"       {msg}\n")
        if not passed:
            all_passed = False

    print("--------------------------------------------------------------------------------")
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY.")
        return 0
    print("VERIFICATION FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
