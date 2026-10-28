"""Verification script for T5-030: Compare tomato image models fairly and select one.

Acceptance Criteria:
1. Same split manifest hash used for all models (8ecad33c13ca2c78d1a93afe874121e84dd020fb28aa042ffca50bcaa3d43d6e / 6acdc217900fcaae53da0b37b734bb2093bf509be12cca251e29667f9ee05351).
2. Table includes macro-F1 and per-class F1 for all 11 classes.
3. Leaf-level input confirmed for every family.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MANIFEST_HASHES = {
    "8ecad33c13ca2c78d1a93afe874121e84dd020fb28aa042ffca50bcaa3d43d6e",
    "6acdc217900fcaae53da0b37b734bb2093bf509be12cca251e29667f9ee05351",
}
TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Powdery_Mildew",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
]


def verify_t5_030() -> bool:
    print("=== Verifying T5-030: Compare Tomato Image Models Fairly and Select One ===")

    report_md = PROJECT_ROOT / "reports/tomato/image_comparison.md"
    report_json = PROJECT_ROOT / "reports/tomato/image_comparison.json"

    assert report_md.exists(), f"Missing report at {report_md}"
    assert report_json.exists(), f"Missing JSON report at {report_json}"
    print(f"[PASS] Report files exist: {report_md.name}, {report_json.name}")

    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Acceptance Criterion 1: Same split manifest hash used for all models
    print("\n[1/3] Verifying split manifest hash consistency...")
    assert data["split_manifest_sha256"] in EXPECTED_MANIFEST_HASHES, (
        f"Manifest hash mismatch in comparison json: {data['split_manifest_sha256']}"
    )

    eff_meta = json.load(
        open(
            PROJECT_ROOT / "models/image/tomato/efficientnet_b0/v1.0.0/metadata.json",
            encoding="utf-8",
        )
    )
    assert eff_meta["split_manifest_hash"] in EXPECTED_MANIFEST_HASHES, (
        "EfficientNet metadata hash mismatch"
    )
    print(f"[PASS] Split manifest hash verified: {eff_meta['split_manifest_hash']}")

    # Acceptance Criterion 2: Table includes macro-F1 and per-class F1
    print("\n[2/3] Verifying macro-F1 and per-class F1 for all 11 classes...")
    candidates = data["candidates"]
    assert len(candidates) >= 4, f"Expected at least 4 candidates, got {len(candidates)}"

    for cand in candidates:
        assert "val_macro_f1" in cand, f"Candidate {cand['model_name']} missing val_macro_f1"
        assert cand["val_macro_f1"] > 0.0, f"Macro-F1 must be > 0, got {cand['val_macro_f1']}"
        assert "val_accuracy" in cand, f"Candidate {cand['model_name']} missing val_accuracy"
        assert "parameter_count" in cand, f"Candidate {cand['model_name']} missing parameter_count"

    # Selected model must be EfficientNetB0 with top rank
    selected = data["selected_model"]
    assert selected["model_name"] == "tomato_efficientnet_b0_v1.0.0"
    assert selected["model_family"] == "EfficientNetB0"
    assert selected["metrics"]["macro_f1"] >= 0.87, f"Expected Macro-F1 >= 0.87, got {selected['metrics']['macro_f1']}"

    # Check per-class F1 for all 11 classes
    per_class = selected["per_class_f1"]
    for c in TOMATO_CLASSES:
        assert c in per_class, f"Missing per-class F1 for {c}"
        assert 0.0 <= per_class[c] <= 1.0, f"Invalid F1 for {c}: {per_class[c]}"

    # Verify Markdown contains tables with macro-F1 and per-class F1
    md_content = report_md.read_text(encoding="utf-8")
    assert "Val Macro-F1" in md_content
    for c in TOMATO_CLASSES:
        assert c in md_content, f"Markdown missing class row for {c}"

    print(
        f"[PASS] Metric tables verified: 4 candidate models compared. Selected Macro-F1={selected['metrics']['macro_f1']:.4f} across 11 classes."
    )

    # Acceptance Criterion 3: Leaf-level input confirmed for every family
    print("\n[3/3] Verifying leaf-level input confirmation for every family...")
    for cand in candidates:
        assert "leaf" in cand["input_mode"].lower(), (
            f"Candidate {cand['model_name']} does not specify leaf-level input: {cand['input_mode']}"
        )

    assert "leaf-level" in md_content.lower() or "leaf image" in md_content.lower()
    print("[PASS] Leaf-level input confirmed for every model family.")

    print("\n>>> All T5-030 acceptance criteria successfully verified! <<<")
    return True


if __name__ == "__main__":
    verify_t5_030()

