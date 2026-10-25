#!/usr/bin/env python3
"""Verification script for Tomato Kolar Provenance & Severity Scales (T2-008).

Acceptance Criteria:
1. Download date, source version, SHA-256 and counts are recorded in tomato provenance sheet.
2. The nine severity scales are stored as constants with the script version that produced them.
3. The raw file is not committed to git.

Re-run command:
    python scripts/verify_tomato_kolar_provenance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_DOC = ROOT / "docs" / "datasets" / "tomato_sensor_kolar.md"
CONTRACT_DOC = ROOT / "docs" / "datasets" / "tomato_sensor_contract.md"
MANIFEST_FILE = ROOT / "processed_data" / "sensor" / "tomato" / "manifest.json"

EXPECTED_SHA256 = "c4032fdc026cc4cb971119fdc27d709bbb4d9b6e24d55df87eebd20464a9d515"
EXPECTED_SEVERITY_SCALES = {
    "Water_Temp_High": 6.58,
    "EC_High_Toxicity": 0.20,
    "EC_Low_Deficiency": 1.30,
    "pH_Imbalance": 1.00,
    "Humidity_High_Mildew_Risk": 5.00,
    "Humidity_Low_Stress": 5.00,
    "N_Deficiency": 50.00,
    "P_Deficiency": 20.00,
    "K_Deficiency": 54.00,
}


def verify_provenance_records() -> tuple[bool, str]:
    """Criterion 1: Download date, source version, SHA-256 and counts are recorded."""
    if not PROVENANCE_DOC.exists():
        return False, f"Missing provenance document: {PROVENANCE_DOC}"

    text = PROVENANCE_DOC.read_text(encoding="utf-8")

    checks = [
        (EXPECTED_SHA256, "SHA-256 hash"),
        ("2026-10-25", "Download date"),
        ("Version 1", "Source version"),
        ("200,000 rows", "Row count"),
        ("29 columns", "Column count"),
        ("109,402,834 bytes", "File size"),
        ("build_tomato_dataset.py", "Build script reference"),
        ("--mode primary", "Build mode"),
    ]

    for val, label in checks:
        if val not in text:
            return False, f"Provenance doc missing {label} ('{val}')"

    return True, "Download date, version, SHA-256, row/col counts, and build command confirmed in provenance sheet"


def verify_severity_scales_constants() -> tuple[bool, str]:
    """Criterion 2: Nine severity scales stored as constants with script version."""
    if not MANIFEST_FILE.exists():
        return False, f"Missing manifest: {MANIFEST_FILE}"
    if not CONTRACT_DOC.exists():
        return False, f"Missing contract: {CONTRACT_DOC}"

    manifest_data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    contract_text = CONTRACT_DOC.read_text(encoding="utf-8")

    # Check manifest severity_scales
    scales = manifest_data.get("severity_scales", {})
    if len(scales) != 9:
        return False, f"Manifest severity_scales expected 9 rules, got {len(scales)}"

    for rule, exp_val in EXPECTED_SEVERITY_SCALES.items():
        val = scales.get(rule)
        if val is None or abs(float(val) - exp_val) > 0.001:
            return False, f"Manifest rule '{rule}' expected scale {exp_val}, got {val}"
        if rule not in contract_text or f"{exp_val:.2f}" not in contract_text:
            return False, f"Contract doc missing severity scale for '{rule}': {exp_val:.2f}"

    # Check script version
    scales_meta = manifest_data.get("severity_scales_metadata", {})
    if scales_meta.get("script_version") != "1.0":
        return False, f"Missing script version in manifest: {scales_meta}"
    if "build_tomato_dataset.py" not in contract_text or "v1.0" not in contract_text:
        return False, "Contract missing script version reference"

    return True, "Nine severity scale constants and script version confirmed in manifest and contract"


def verify_raw_file_not_in_git() -> tuple[bool, str]:
    """Criterion 3: Raw file is not committed to git."""
    # Check git tracked files
    result = subprocess.run(
        ["git", "ls-files", "*kolar*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    # Only markdown or metadata should be tracked
    csv_tracked = [f for f in tracked_files if f.endswith(".csv")]
    if csv_tracked:
        return False, f"Raw Kolar CSV tracked in git: {csv_tracked}"

    # Check git status for untracked big csv files
    status_res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "kolar_tomato_precision_agri_dataset.csv" in status_res.stdout:
        return False, "Raw Kolar CSV present in working directory git status"

    return True, "Confirmed raw Kolar CSV is not tracked or staged in git"


def main() -> int:
    checks = [
        ("Criterion 1: Provenance Records & Checksum", verify_provenance_records),
        ("Criterion 2: Nine Severity Scale Constants", verify_severity_scales_constants),
        ("Criterion 3: Raw File Excluded from Git", verify_raw_file_not_in_git),
    ]

    all_passed = True
    print("=" * 65)
    print("Tomato Kolar Provenance & Severity Scales Verification (T2-008)")
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
