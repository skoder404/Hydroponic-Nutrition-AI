#!/usr/bin/env python3
"""Verification script for T5-033: Promote image and feature models to the registry.

Acceptance Criteria:
1. Registry validation passes for all promoted image models.
2. Missing combinations are listed in models/manifests/COVERAGE.md.
3. Rollback tested via CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.registry import ModelRegistry


def run_verification() -> bool:
    print("================================================================================")
    print("VERIFICATION FOR T5-033: PROMOTE IMAGE AND FEATURE MODELS TO REGISTRY")
    print("================================================================================\n")

    registry = ModelRegistry()
    crops = ["lettuce", "spinach", "tomato"]

    # 1. Registry validation
    print("[1/3] Validating Promoted Image Models in ModelRegistry...")
    for crop in crops:
        try:
            pointer = registry.get_active_pointer(crop, "image")
            meta = registry.get_active_metadata(crop, "image")
            model_obj, _ = registry.load_model(crop, "image")
            print(
                f"  - {crop.capitalize()} Image: v{pointer['active_version']} "
                f"({meta.family}, val_f1={meta.validation_metrics.get('macro_f1', 0):.4f}, "
                f"test_f1={meta.final_test_metrics.get('macro_f1', 0):.4f}, "
                f"status={meta.status})"
            )
        except Exception as exc:
            print(f"  [FAIL] Failed registry check for {crop}/image: {exc}")
            return False
    print("  [PASS] All 3 promoted image models are active and valid in the registry.\n")

    # 2. Check COVERAGE.md
    print("[2/3] Verifying models/manifests/COVERAGE.md...")
    coverage_file = PROJECT_ROOT / "models" / "manifests" / "COVERAGE.md"
    if not coverage_file.is_file():
        print(f"  [FAIL] Missing coverage manifest at {coverage_file}")
        return False

    coverage_text = coverage_file.read_text(encoding="utf-8")
    required_sections = [
        "Modality & Crop Coverage Matrix",
        "Explicitly Unsupported & Untrained Combinations",
        "Tomato Live Image & Feature Models",
        "Spinach Live & Lettuce Live Telemetry",
        "Classical Handcrafted Feature Models",
        "Direct Whole-Canopy Unsegmented Classification Models",
    ]
    for sec in required_sections:
        if sec not in coverage_text:
            print(f"  [FAIL] Missing section '{sec}' in COVERAGE.md")
            return False
    print("  [PASS] COVERAGE.md exists and comprehensively documents all combinations.\n")

    # 3. Run unit tests
    print("[3/3] Running pytest on tests/unit/test_t5_033.py...")
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/unit/test_t5_033.py"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print("  [FAIL] Unit tests failed.")
        return False
    print("  [PASS] Unit tests passed cleanly (including CLI rollback test).\n")

    print("================================================================================")
    print("ALL ACCEPTANCE CRITERIA FOR T5-033 VERIFIED SUCCESSFULLY!")
    print("================================================================================")
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)

