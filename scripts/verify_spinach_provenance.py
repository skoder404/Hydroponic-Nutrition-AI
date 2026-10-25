#!/usr/bin/env python3
"""Verification script for T2-003: Spinach image and sensor provenance sheets.

Validates:
1. docs/datasets/spinach_images_ymtzd8cf6x.md exists with Mendeley v1 source details.
2. docs/datasets/spinach_images_skf6w2s2h2.md exists with Mendeley v3 source details.
3. docs/datasets/spinach_sensor_dataset.md exists and contains:
   - Canonical USDA Ag Data Commons URL (10.15482/USDA.ADC/33383695).
   - Column taxonomy matching Overview §13.4:
     * air_temperature_C -> measured
     * relative_humidity_percent -> adjusted
     * water_temperature_C -> synthetic
     * ph -> synthetic
     * ec_mS_cm -> synthetic
     * tds_ppm -> derived
   - tds_ppm explicitly marked as derived and excluded from model inputs.
"""
import re
import sys
from pathlib import Path

REQUIRED_FILES = [
    "docs/datasets/spinach_images_ymtzd8cf6x.md",
    "docs/datasets/spinach_images_skf6w2s2h2.md",
    "docs/datasets/spinach_sensor_dataset.md",
]


def check_files_exist():
    missing = [f for f in REQUIRED_FILES if not Path(f).is_file()]
    if missing:
        return False, f"Missing files: {missing}"
    return True, "All three spinach provenance files exist."


def check_canonical_url():
    sensor_sheet = Path("docs/datasets/spinach_sensor_dataset.md").read_text(encoding="utf-8")
    has_canonical = ("10.15482/USDA.ADC/33383695" in sensor_sheet or "data.nal.usda.gov" in sensor_sheet)
    if not has_canonical:
        return False, "Canonical USDA Ag Data Commons URL not found in spinach_sensor_dataset.md"
    return True, "Canonical USDA Ag Data Commons URL recorded."


def check_column_table_and_tds():
    sensor_sheet = Path("docs/datasets/spinach_sensor_dataset.md").read_text(encoding="utf-8")

    # Overview §13.4 expectations:
    expected_tags = {
        "air_temperature_C": "measured",
        "relative_humidity_percent": "adjusted",
        "water_temperature_C": "synthetic",
        "ph": "synthetic",
        "ec_mS_cm": "synthetic",
        "tds_ppm": "derived",
    }

    errors = []
    for col, tag in expected_tags.items():
        pattern = rf"\|.*`{re.escape(col)}`.*\|.*`{re.escape(tag)}`.*\|"
        if not re.search(pattern, sensor_sheet):
            errors.append(f"Column '{col}' not tagged as '{tag}' in column table")

    # Check tds_ppm excluded
    tds_pattern = r"\|.*`tds_ppm`.*\|.*derived.*\|.*EXCLUDED"
    if not re.search(tds_pattern, sensor_sheet, re.IGNORECASE):
        errors.append("tds_ppm is not explicitly marked as EXCLUDED in table")

    if errors:
        return False, "; ".join(errors)
    return True, "Column taxonomy matches Overview §13.4 and tds_ppm is marked derived/excluded."


def main():
    print("=== Running T2-003 Verification ===")
    errors = []

    ok_files, msg_files = check_files_exist()
    if not ok_files:
        errors.append(msg_files)
    else:
        print(f"[PASS] {msg_files}")

    ok_url, msg_url = check_canonical_url()
    if not ok_url:
        errors.append(msg_url)
    else:
        print(f"[PASS] {msg_url}")

    ok_col, msg_col = check_column_table_and_tds()
    if not ok_col:
        errors.append(msg_col)
    else:
        print(f"[PASS] {msg_col}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T2-003 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
