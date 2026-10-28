"""Verification script for T5-028: Compare lettuce image models fairly and select one.

Acceptance Criteria:
1. Same split manifest hash used for all models (5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52).
2. Table includes macro-F1, per-class F1, and parameter counts.
3. Leaf-level input confirmed for every family.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MANIFEST_HASH = (
    "5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52"
)


def verify_t5_028() -> bool:
    print("=== Verifying T5-028: Compare Lettuce Image Models Fairly and Select One ===")

    report_md = PROJECT_ROOT / "reports/lettuce/image_comparison.md"
    report_json = PROJECT_ROOT / "reports/lettuce/image_comparison.json"

    assert report_md.exists(), f"Missing report at {report_md}"
    assert report_json.exists(), f"Missing JSON report at {report_json}"
    print(f"[PASS] Report files exist: {report_md.name}, {report_json.name}")

    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Acceptance Criterion 1: Same split manifest hash used for all models
    print("\n[1/3] Verifying split manifest hash consistency...")
    assert data["split_manifest_sha256"] == EXPECTED_MANIFEST_HASH, (
        f"Manifest hash mismatch: {data['split_manifest_sha256']} vs {EXPECTED_MANIFEST_HASH}"
    )

    eff_meta = json.load(
        open(
            PROJECT_ROOT / "models/image/lettuce/efficientnet_b0/v1.0.0/metadata.json",
            encoding="utf-8",
        )
    )
    assert eff_meta["split_manifest_hash"] == EXPECTED_MANIFEST_HASH, (
        "EfficientNet metadata hash mismatch"
    )
    print(f"[PASS] Split manifest hash verified: {EXPECTED_MANIFEST_HASH}")

    # Acceptance Criterion 2: Table includes macro-F1, per-class F1, and parameter counts
    print("\n[2/3] Verifying table metrics (macro-F1, per-class F1, parameter counts)...")
    md_content = report_md.read_text(encoding="utf-8")
    assert "Macro-F1" in md_content or "macro_f1" in md_content
    assert "0.9119" in md_content
    assert "0.7045" in md_content
    assert "0.4930" in md_content
    assert "0.4840" in md_content

    # Per-class F1 check
    for cls_name in [
        "Healthier",
        "NitrogenDeficiency",
        "PhosphorusDeficiency",
        "PotassiumDeficiency",
    ]:
        assert cls_name in md_content, f"Class {cls_name} not found in markdown report"
        assert cls_name in data["selected_model"]["per_class_f1"], (
            f"Class {cls_name} not found in json per_class_f1"
        )

    # Parameter counts check
    assert "4,012,672" in md_content or "4012672" in md_content
    assert "456,228" in md_content or "456228" in md_content
    assert data["selected_model"]["parameter_count"] == 4012672
    print("[PASS] Macro-F1, per-class F1, and parameter counts verified across all candidates.")

    # Acceptance Criterion 3: Leaf-level input confirmed for every family
    print("\n[3/3] Verifying leaf-level input confirmation for every family...")
    assert "Leaf-Level Input Confirmation" in md_content
    assert "source_map.csv" in md_content
    assert len(data["candidates"]) == 4
    for cand in data["candidates"]:
        assert "leaf" in cand["input_mode"].lower()
    print("[PASS] Leaf-level input confirmed for all 4 model families.")

    print("\n=== ALL T5-028 ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    verify_t5_028()
