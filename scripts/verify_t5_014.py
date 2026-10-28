#!/usr/bin/env python3
"""Verification script for T5-014: Train and report lettuce OpenCV-feature classifiers.

Acceptance Criteria:
1. Validation-only report written
2. Feature order saved in metadata
3. Class support shown
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.lettuce import LETTUCE_FEATURE_NAMES


def run_verification() -> bool:
    print("[1/3] Verifying validation-only report written...")
    reports_dir = PROJECT_ROOT / "reports/lettuce"
    comp_md = reports_dir / "lettuce_feature_experiments.md"
    comp_json = reports_dir / "lettuce_feature_experiments.json"

    if not comp_md.exists() or not comp_json.exists():
        print(f"FAILED: Comparison report missing at {comp_md}")
        return False

    with open(comp_json, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    # Invariant: split_evaluated is strictly 'validation'
    assert comp_data["evaluation_split"] == "validation"

    rf_rep_md = reports_dir / "lettuce_feature_random_forest_v1.0.0_validation.md"
    rf_rep_json = reports_dir / "lettuce_feature_random_forest_v1.0.0_validation.json"
    svm_rep_md = reports_dir / "lettuce_feature_svm_v1.0.0_validation.md"
    svm_rep_json = reports_dir / "lettuce_feature_svm_v1.0.0_validation.json"

    assert rf_rep_md.exists() and rf_rep_json.exists(), "Random Forest validation report missing"
    assert svm_rep_md.exists() and svm_rep_json.exists(), "SVM validation report missing"

    for r_json in (rf_rep_json, svm_rep_json):
        with open(r_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["split_evaluated"] == "validation"
            assert "test" not in data["model_name"]
            assert len(data["limitations"]) > 0

    print("  AC 1 PASS: Structured Section 18.2 reports written exclusively on validation partition.")

    print("[2/3] Verifying feature order saved in metadata...")
    rf_meta_file = PROJECT_ROOT / "models/image/lettuce/random_forest/v1.0.0/metadata.json"
    svm_meta_file = PROJECT_ROOT / "models/image/lettuce/svm/v1.0.0/metadata.json"

    assert rf_meta_file.exists(), f"Metadata missing at {rf_meta_file}"
    assert svm_meta_file.exists(), f"Metadata missing at {svm_meta_file}"

    for meta_file in (rf_meta_file, svm_meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert "feature_order" in meta, f"Missing feature_order in {meta_file}"
            assert tuple(meta["feature_order"]) == LETTUCE_FEATURE_NAMES, (
                f"Feature order mismatch in {meta_file}: expected {LETTUCE_FEATURE_NAMES}, got {meta['feature_order']}"
            )
            assert meta["num_features"] == len(LETTUCE_FEATURE_NAMES)

    print(f"  AC 2 PASS: Canonical feature order {LETTUCE_FEATURE_NAMES} strictly verified in model metadata.")

    print("[3/3] Verifying class support shown across all classes...")
    expected_classes = {"Healthier", "NitrogenDeficiency", "PhosphorusDeficiency", "PotassiumDeficiency"}
    for r in comp_data["models"]:
        per_class = r["per_class"]
        assert set(per_class.keys()) == expected_classes, f"Class mismatch: {set(per_class.keys())}"
        for cname, metrics in per_class.items():
            assert "support" in metrics, f"Missing support for class {cname}"
            assert metrics["support"] > 0, f"Support must be > 0 for {cname}"
            print(f"    [{r['family']}] {cname}: Support={metrics['support']}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    # Check Markdown report displays support column
    md_content = comp_md.read_text(encoding="utf-8")
    assert "Support (Validation Instances)" in md_content
    for cname in expected_classes:
        assert f"`{cname}`" in md_content

    print("  AC 3 PASS: Support explicitly shown for all 4 classes in both JSON and Markdown reports.")

    print("\nAll T5-014 acceptance criteria successfully verified!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
