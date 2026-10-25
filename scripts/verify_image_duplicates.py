#!/usr/bin/env python3
"""Verification script for Image Duplicate Detection (T2-014).

Acceptance Criteria:
1. Duplicate groups are reported per crop (lettuce, spinach, tomato).
2. Threshold and library are documented.
3. No file is modified or deleted in image_data/.

Re-run command:
    python scripts/verify_image_duplicates.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "data"
CROPS = ["lettuce", "spinach", "tomato"]


def verify_per_crop_reports() -> tuple[bool, str]:
    """Criterion 1: Duplicate groups are reported per crop."""
    for crop in CROPS:
        p = REPORT_DIR / f"image_duplicates_{crop}.json"
        if not p.exists():
            return False, f"Missing duplicate report for crop '{crop}': {p}"

        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        if "exact_duplicates" not in data or "groups" not in data["exact_duplicates"]:
            return False, f"Crop '{crop}' report missing exact_duplicates.groups"

        if "near_duplicates" not in data or "clusters" not in data["near_duplicates"]:
            return False, f"Crop '{crop}' report missing near_duplicates.clusters"

        if "total_images_scanned" not in data or data["total_images_scanned"] <= 0:
            return False, f"Crop '{crop}' report has zero or missing total_images_scanned"

    summary_md = REPORT_DIR / "image_duplicates_summary.md"
    if not summary_md.exists():
        return False, f"Missing consolidated summary report: {summary_md}"

    return True, "All three crop JSON reports and summary markdown exist with duplicate groups."


def verify_threshold_and_library_documented() -> tuple[bool, str]:
    """Criterion 2: Threshold and library are documented."""
    for crop in CROPS:
        p = REPORT_DIR / f"image_duplicates_{crop}.json"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        method = data.get("methodology", {})
        lib = method.get("library")
        thresh = method.get("hamming_distance_threshold")
        algo = method.get("perceptual_hash_algorithm")

        if not lib or "pillow" not in lib.lower():
            return False, f"Crop '{crop}' missing documented library (expected Pillow, got {lib})"

        if thresh is None or thresh != 4:
            return False, f"Crop '{crop}' missing documented threshold (expected 4, got {thresh})"

        if not algo or "dhash" not in algo.lower():
            return False, f"Crop '{crop}' missing documented perceptual hash algorithm"

    # Check summary markdown
    summary_md = REPORT_DIR / "image_duplicates_summary.md"
    text = summary_md.read_text(encoding="utf-8")
    if "pillow" not in text.lower() or "dhash" not in text.lower() or "4" not in text:
        return False, "Summary markdown does not document Pillow, dHash, and threshold <= 4"

    return True, "Library (Pillow), algorithm (dHash), and conservative threshold (<= 4) clearly documented."


def verify_non_destructive_execution() -> tuple[bool, str]:
    """Criterion 3: No file is modified or deleted."""
    res = subprocess.run(
        ["git", "status", "--porcelain", "image_data"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return False, f"git status failed: {res.stderr}"

    changed = res.stdout.strip()
    if changed:
        return False, f"image_data/ has modified or deleted files:\n{changed}"

    return True, "git status confirms zero files in image_data/ were modified or deleted."


def main() -> int:
    print("================================================================================")
    print("VERIFYING IMAGE DUPLICATE DETECTION (T2-014)")
    print("================================================================================\n")

    checks = [
        ("Criterion 1: Duplicate groups reported per crop", verify_per_crop_reports),
        ("Criterion 2: Threshold and library documented", verify_threshold_and_library_documented),
        ("Criterion 3: Non-destructive execution (no image modified/deleted)", verify_non_destructive_execution),
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
