#!/usr/bin/env python3
"""Verification script for Tomato Sensor Dataset Audit (T2-007).

Acceptance Criteria:
1. Every item from the task description is confirmed from the CSVs or listed as a discrepancy.
2. Manifest contains the ordered feature list, row counts per class and per month.
3. Mode used (primary or single) and the test months are recorded.
4. Contract documentation exists at docs/datasets/tomato_sensor_contract.md stating simulated origin.

Re-run command:
    python scripts/verify_tomato_sensor_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "sensor_data" / "Tomato_Sensor-Readings"
MANIFEST_PATH = ROOT / "processed_data" / "sensor" / "tomato" / "manifest.json"
CONTRACT_PATH = ROOT / "docs" / "datasets" / "tomato_sensor_contract.md"

EXPECTED_FEATURES = [
    "air_temperature_C",
    "relative_humidity_percent",
    "water_temperature_C",
    "ph",
    "ec_mS_cm",
    "nitrogen_mg_L",
    "phosphorus_mg_L",
    "potassium_mg_L",
]

EXPECTED_CLASSES = [
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
]


def verify_csv_counts_and_classes() -> tuple[bool, str]:
    """Criterion 1: Every item confirmed from the CSVs or listed as a discrepancy."""
    full_csv = DATA_DIR / "tomato_hydroponic_full.csv"
    train_csv = DATA_DIR / "tomato_hydroponic_train.csv"
    test_csv = DATA_DIR / "tomato_hydroponic_test.csv"

    for p in (full_csv, train_csv, test_csv):
        if not p.exists():
            return False, f"Missing CSV: {p}"

    full = pd.read_csv(full_csv)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Full rows: 44,150 (not 44,141 as overview stated)
    if len(full) != 44150:
        return False, f"Expected 44,150 full rows, got {len(full)}"
    if len(train) != 35320:
        return False, f"Expected 35,320 train rows, got {len(train)}"
    if len(test) != 8830:
        return False, f"Expected 8,830 test rows, got {len(test)}"

    # 10 classes in all splits
    for name, df in [("full", full), ("train", train), ("test", test)]:
        classes = sorted(df["label"].unique().tolist())
        if classes != EXPECTED_CLASSES:
            return False, f"Split '{name}' classes mismatch: {classes} != {EXPECTED_CLASSES}"

    # Features present in correct order
    cols = list(full.columns)
    for idx, feat in enumerate(EXPECTED_FEATURES, start=1):
        if cols[idx] != feat:
            return False, f"Feature at index {idx} expected '{feat}', got '{cols[idx]}'"

    # Mode confirmation: Multiple_Faults absent -> primary mode
    if "Multiple_Faults" in full["label"].values:
        return False, "Multiple_Faults found; unexpected single mode"

    return True, "CSV row counts (44,150/35,320/8,830), 10 classes, feature order, and primary mode confirmed."


def verify_manifest() -> tuple[bool, str]:
    """Criterion 2 & 3: Manifest contains ordered features, row counts per class and month, mode, and test months."""
    if not MANIFEST_PATH.exists():
        return False, f"Manifest missing: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Check ordered features
    if data.get("ordered_features") != EXPECTED_FEATURES:
        return False, f"Manifest ordered_features mismatch: {data.get('ordered_features')}"

    # Check mode
    if data.get("labeling_mode") != "primary":
        return False, f"Manifest labeling_mode expected 'primary', got: {data.get('labeling_mode')}"

    # Check test months
    test_months = data.get("splits", {}).get("test_months")
    if not test_months or len(test_months) != 27:
        return False, f"Expected 27 test months in manifest, got {len(test_months) if test_months else 0}"

    # Check row counts per class
    rc_class = data.get("row_counts_per_class", {})
    if len(rc_class.get("full", {})) != 10:
        return False, "Manifest missing 10 classes in row_counts_per_class.full"

    # Check row counts per month
    rc_month = data.get("row_counts_per_month", {})
    if len(rc_month.get("full", {})) != 132:
        return False, f"Expected 132 months in manifest, got {len(rc_month.get('full', {}))}"

    return True, "Manifest contains ordered features, class counts, monthly counts, mode ('primary'), and 27 test months."


def verify_contract_documentation() -> tuple[bool, str]:
    """Criterion 4: Contract documentation exists stating simulated origin of Kolar data."""
    if not CONTRACT_PATH.exists():
        return False, f"Contract file missing: {CONTRACT_PATH}"

    text = CONTRACT_PATH.read_text(encoding="utf-8")

    # Verify mention that Kolar looks simulated
    if "simulated" not in text.lower():
        return False, "Contract does not state that the Kolar data looks simulated"

    # Verify mention of synthetic water temperature
    if "synthetic" not in text.lower() or "water_temperature_c" not in text.lower():
        return False, "Contract missing synthetic water temperature documentation"

    # Verify mention of primary mode and proxy transforms
    if "primary" not in text.lower() or "proxy" not in text.lower():
        return False, "Contract missing primary mode or proxy transform documentation"

    return True, "Contract documentation verified (states simulated origin, proxy transforms, synthetic water temp)."


def main() -> int:
    print("================================================================================")
    print("VERIFYING TOMATO SENSOR DATASET AUDIT (T2-007)")
    print("================================================================================\n")

    checks = [
        ("Criterion 1: CSV counts, classes, feature ordering, and mode", verify_csv_counts_and_classes),
        ("Criterion 2 & 3: Manifest schema, features, counts, mode, and test months", verify_manifest),
        ("Criterion 4: Technical contract states simulated origin and proxy transforms", verify_contract_documentation),
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
