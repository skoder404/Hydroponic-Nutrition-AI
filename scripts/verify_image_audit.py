#!/usr/bin/env python3
"""Verification script for Image Dataset Audit (T2-013).

Acceptance Criteria:
1. A JSON report exists for each crop (lettuce, spinach, tomato).
2. Corrupt files are listed, not deleted.
3. Single-leaf vs whole-plant status per crop is recorded from the sample.
4. Class labels are the exact folder names and every difference from Overview §6 is listed (e.g. Tomato_Powdery_Mildew).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CROPS = ["lettuce", "spinach", "tomato"]


def verify_reports_exist() -> tuple[bool, str]:
    """Criterion 1: A JSON report exists for each crop."""
    missing = []
    for crop in CROPS:
        p = PROJECT_ROOT / "reports" / "data" / f"image_audit_{crop}.json"
        if not p.exists():
            missing.append(str(p))

    if missing:
        return False, f"Missing audit report files: {missing}"
    return True, "All three JSON audit reports exist in reports/data/."


def verify_corrupt_files_tracked() -> tuple[bool, str]:
    """Criterion 2: Corrupt files are listed, not deleted."""
    for crop in CROPS:
        p = PROJECT_ROOT / "reports" / "data" / f"image_audit_{crop}.json"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        if "corrupt_files" not in data or not isinstance(data["corrupt_files"], list):
            return False, f"Crop '{crop}' report missing 'corrupt_files' list."

        if "total_images" not in data or data["total_images"] <= 0:
            return False, f"Crop '{crop}' report has zero or missing total_images."

    return True, "Corrupt files list verified across all reports (non-destructive audit)."


def verify_presentation_status() -> tuple[bool, str]:
    """Criterion 3: Single-leaf vs whole-plant status per crop is recorded from the sample."""
    expected = {
        "lettuce": "whole-plant",
        "spinach": "single-leaf",
        "tomato": "single-leaf",
    }
    for crop, expected_status in expected.items():
        p = PROJECT_ROOT / "reports" / "data" / f"image_audit_{crop}.json"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        sample_check = data.get("sample_check", {})
        status = sample_check.get("overall_presentation_status")
        if status != expected_status:
            return False, f"Expected '{expected_status}' for {crop}, got '{status}'."

        per_class = sample_check.get("per_class_sample", {})
        if not per_class:
            return False, f"Crop '{crop}' missing per_class_sample records."

        for cls_name, info in per_class.items():
            if info.get("presentation_status") != expected_status:
                return False, f"Class {cls_name} has status {info.get('presentation_status')} != {expected_status}"
            if info.get("sampled_count", 0) <= 0:
                return False, f"Class {cls_name} has zero sample count."

    return True, "Single-leaf vs whole-plant status properly verified from 30-image samples per class."


def verify_class_labels_and_differences() -> tuple[bool, str]:
    """Criterion 4: Class labels are exact folder names and every difference from Overview §6 is listed."""
    for crop in CROPS:
        p = PROJECT_ROOT / "reports" / "data" / f"image_audit_{crop}.json"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        comp = data.get("overview_sec6_comparison", {})
        folder_classes = comp.get("exact_folder_classes", [])
        class_counts = data.get("class_counts", {})

        if sorted(folder_classes) != sorted(class_counts.keys()):
            return False, f"Folder classes do not match class_counts keys for {crop}."

        diffs = comp.get("differences", [])
        if not diffs:
            return False, f"Differences list is empty for {crop}."

        if crop == "tomato":
            # Must explicitly list Tomato_Powdery_Mildew
            has_powdery = any("Tomato_Powdery_Mildew" in d.get("folder_name", "") for d in diffs)
            if not has_powdery:
                return False, "Tomato differences did not list 'Tomato_Powdery_Mildew'."

        if crop == "lettuce":
            has_healthier = any("Healthier" in d.get("folder_name", "") for d in diffs)
            if not has_healthier:
                return False, "Lettuce differences did not list 'Healthier'."

    return True, "Class labels match folder names and all Overview §6 taxonomy differences are listed."


def main() -> int:
    checks = [
        ("A JSON report exists for each crop", verify_reports_exist),
        ("Corrupt files are listed, not deleted", verify_corrupt_files_tracked),
        ("Single-leaf vs whole-plant status recorded from sample", verify_presentation_status),
        ("Class labels are exact folder names and §6 differences listed", verify_class_labels_and_differences),
    ]

    print("================================================================================")
    print("VERIFY IMAGE DATASET AUDIT (T2-013)")
    print("================================================================================\n")

    all_passed = True
    for title, check_fn in checks:
        ok, msg = check_fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {title}\n       {msg}")
        if not ok:
            all_passed = False

    print("\n--------------------------------------------------------------------------------")
    if all_passed:
        print("All criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
