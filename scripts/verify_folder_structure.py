#!/usr/bin/env python3
"""Verification script for T1-002: Monorepo folder structure and rules.

Validates:
1. All required directories exist.
2. Git tracking: every leaf directory contains either files or .gitkeep.
3. No duplicate or parallel folders (e.g. data duplicates, parallel app dirs).
4. ml/fusion/README.md exists and contains the strict no-fusion rule (Overview §19, TECH_RULES §15).
5. Deviations from TECH_RULES layout are logged in docs/decisions/0001-data-inventory.md.
"""
import sys
from pathlib import Path

REQUIRED_DIRS = [
    "frontend",
    "backend/app/api",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/database",
    "backend/app/core",
    "ml/image_models",
    "ml/sensor_models",
    "ml/segmentation",
    "ml/features",
    "ml/fusion",
    "ml/recommendations",
    "ml/evaluation",
    "ml/registry",
    "processed_data/splits",
    "processed_data/features",
    "processed_data/sensor",
    "processed_data/manifests",
    "models/image",
    "models/sensor",
    "models/manifests",
    "reports",
    "notebooks",
    "scripts",
    "tests/unit",
    "tests/integration",
    "tests/contracts",
    "tests/data",
    "docs/learning",
    "docs/datasets",
    "docs/decisions",
]

DISALLOWED_DUPLICATES = [
    "data",
    "src",
    "app",
    "raw_data",
    "datasets",
    "ml_models",
]


def check_directories():
    missing = []
    untracked = []
    for d in REQUIRED_DIRS:
        p = Path(d)
        if not p.is_dir():
            missing.append(d)
        else:
            # Check if directory has at least one file or subfolder
            contents = list(p.iterdir())
            if not contents:
                untracked.append(d)
    return missing, untracked


def check_duplicates():
    found = []
    for d in DISALLOWED_DUPLICATES:
        p = Path(d)
        if p.exists() and p.is_dir():
            found.append(d)
    return found


def check_fusion_readme():
    path = Path("ml/fusion/README.md")
    if not path.is_file():
        return False, "ml/fusion/README.md does not exist."
    content = path.read_text(encoding="utf-8")
    if "No fusion model may be trained on unpaired" not in content:
        return False, "ml/fusion/README.md missing no-fusion rule statement."
    if "19" not in content:
        return False, "ml/fusion/README.md does not reference Overview §19."
    return True, "ml/fusion/README.md satisfies all criteria."


def check_deviations_logged():
    adr_path = Path("docs/decisions/0001-data-inventory.md")
    if not adr_path.is_file():
        return False, "docs/decisions/0001-data-inventory.md missing."
    content = adr_path.read_text(encoding="utf-8")
    if "Monorepo Layout & Deviations Log" not in content:
        return False, "ADR 0001 does not contain Monorepo Layout & Deviations Log."
    return True, "Deviations logged in ADR 0001."


def main():
    print("=== Running T1-002 Verification ===")
    errors = []

    # 1. Required dirs
    missing, untracked = check_directories()
    if missing:
        errors.append(f"Missing required directories: {missing}")
    else:
        print(f"[PASS] All {len(REQUIRED_DIRS)} required directories exist.")

    if untracked:
        errors.append(f"Empty directories lacking .gitkeep or contents: {untracked}")
    else:
        print("[PASS] All required directories contain files or .gitkeep for git tracking.")

    # 2. Duplicate dirs
    dupes = check_duplicates()
    if dupes:
        errors.append(f"Found disallowed duplicate directories: {dupes}")
    else:
        print("[PASS] No parallel or duplicate directories found.")

    # 3. Fusion README
    fusion_ok, fusion_msg = check_fusion_readme()
    if not fusion_ok:
        errors.append(fusion_msg)
    else:
        print(f"[PASS] {fusion_msg}")

    # 4. Deviations logged
    dev_ok, dev_msg = check_deviations_logged()
    if not dev_ok:
        errors.append(dev_msg)
    else:
        print(f"[PASS] {dev_msg}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T1-002 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
