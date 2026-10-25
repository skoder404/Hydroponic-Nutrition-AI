#!/usr/bin/env python3
"""Verification script for Spinach Sensor Dataset Audit (T2-006).

Acceptance Criteria:
1. Counts per class per split match the README or discrepancies are listed.
2. No six-hour block appears in both train and test.
3. Real vs adjusted vs synthetic columns are recorded and tds_ppm is marked derived.
4. Report exists at reports/data/spinach_sensor_audit.md with all required findings.

Re-run command:
    python scripts/verify_spinach_sensor_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "sensor_data" / "Spinach_Sensor-Readings"
REPORT_FILE = PROJECT_ROOT / "reports" / "data" / "spinach_sensor_audit.md"

EXPECTED_CLASSES = [
    "Normal",
    "Water_Temp_High",
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "pH_Imbalance",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
]


def verify_counts_and_classes() -> tuple[bool, str]:
    """Criterion 1: Counts per class per split match the README."""
    full_csv = DATA_DIR / "spinach_7class_full.csv"
    train_csv = DATA_DIR / "spinach_7class_train.csv"
    test_csv = DATA_DIR / "spinach_7class_test.csv"

    for p in (full_csv, train_csv, test_csv):
        if not p.exists():
            return False, f"Missing dataset file: {p}"

    full = pd.read_csv(full_csv)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    if len(full) != 1400:
        return False, f"Expected 1400 full rows, got {len(full)}"
    if len(train) != 1120:
        return False, f"Expected 1120 train rows, got {len(train)}"
    if len(test) != 280:
        return False, f"Expected 280 test rows, got {len(test)}"

    full_counts = full["label"].value_counts()
    train_counts = train["label"].value_counts()
    test_counts = test["label"].value_counts()

    if len(full_counts) != 7 or len(train_counts) != 7 or len(test_counts) != 7:
        return False, "Expected 7 distinct classes in full, train, and test"

    for cls in EXPECTED_CLASSES:
        if full_counts.get(cls, 0) != 200:
            return False, f"Full class '{cls}' count is {full_counts.get(cls)}, expected 200"
        if train_counts.get(cls, 0) != 160:
            return False, f"Train class '{cls}' count is {train_counts.get(cls)}, expected 160"
        if test_counts.get(cls, 0) != 40:
            return False, f"Test class '{cls}' count is {test_counts.get(cls)}, expected 40"

    return True, "Counts per class per split match README (1400 full = 200*7, 1120 train = 160*7, 280 test = 40*7)"


def verify_block_split_no_leakage() -> tuple[bool, str]:
    """Criterion 2: No six-hour block appears in both train and test."""
    train_csv = DATA_DIR / "spinach_7class_train.csv"
    test_csv = DATA_DIR / "spinach_7class_test.csv"

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    for df in (train, test):
        ts = pd.to_datetime(df["source_timestamp_EDT"])
        df["block"] = df["bench"] + "|" + ts.dt.strftime("%Y-%m-%d") + "|" + (ts.dt.hour // 6).astype(str)

    train_blocks = set(train["block"])
    test_blocks = set(test["block"])
    overlap = train_blocks.intersection(test_blocks)

    if overlap:
        return False, f"6-hour block leakage detected: {len(overlap)} blocks overlap between train and test"

    # Sample ID check
    overlap_ids = set(train["sample_id"]).intersection(set(test["sample_id"]))
    if overlap_ids:
        return False, f"Sample ID leakage: {len(overlap_ids)} samples shared between train and test"

    return True, f"Zero 6-hour temporal block overlap (train: {len(train_blocks)} blocks, test: {len(test_blocks)} blocks)"


def verify_column_provenance_and_derived_tds() -> tuple[bool, str]:
    """Criterion 3: Real vs adjusted vs synthetic columns recorded, tds_ppm marked derived."""
    full_csv = DATA_DIR / "spinach_7class_full.csv"
    full = pd.read_csv(full_csv)

    # Check humidity_source
    hum_counts = full["humidity_source"].value_counts().to_dict()
    if hum_counts.get("real") != 1200 or hum_counts.get("adjusted") != 200:
        return False, f"Expected 1200 real and 200 adjusted rows, got {hum_counts}"

    # Verify TDS derived formula: tds ~= ec * 700 (+- 2% noise)
    ratio = full["tds_ppm"] / (full["ec_mS_cm"] * 700)
    mean_ratio = float(ratio.mean())
    if not (0.98 <= mean_ratio <= 1.02):
        return False, f"TDS derived check failed: mean ratio {mean_ratio:.4f} outside [0.98, 1.02]"

    return True, f"Real (1200) vs adjusted (200) verified; TDS confirmed derived from EC*700 (mean ratio {mean_ratio:.4f})"


def verify_report_exists() -> tuple[bool, str]:
    """Criterion 4: Report exists with required sections."""
    if not REPORT_FILE.exists():
        return False, f"Report file not found: {REPORT_FILE}"

    text = REPORT_FILE.read_text(encoding="utf-8")
    required_phrases = [
        "Spinach Sensor Dataset Audit Report",
        "ALL CHECKS PASSED",
        "1400 rows",
        "1120 rows",
        "280 rows",
        "6-hour temporal block overlap",
        "tds_ppm",
        "Derived (Excluded)",
        "Meng et al. (2025)",
        "Humidity_High_Mildew_Risk",
        "Single-rule violation constraint",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        return False, f"Report missing required sections: {missing}"

    return True, f"Report exists at {REPORT_FILE.relative_to(PROJECT_ROOT)} with all required sections"


def main() -> int:
    checks = [
        ("Criterion 1: Counts & Class Distribution", verify_counts_and_classes),
        ("Criterion 2: 6-Hour Block Separation (No Leakage)", verify_block_split_no_leakage),
        ("Criterion 3: Provenance & Derived TDS", verify_column_provenance_and_derived_tds),
        ("Criterion 4: Audit Report", verify_report_exists),
    ]

    all_passed = True
    print("=" * 65)
    print("Spinach Sensor Dataset Audit Verification (T2-006)")
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
