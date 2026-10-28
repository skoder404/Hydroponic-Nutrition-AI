"""Unit tests for T5-030: Compare tomato image models fairly and select one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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


def test_tomato_comparison_reports_exist() -> None:
    """Verify that both markdown and JSON comparison reports exist."""
    report_md = PROJECT_ROOT / "reports/tomato/image_comparison.md"
    report_json = PROJECT_ROOT / "reports/tomato/image_comparison.json"
    assert report_md.exists(), f"Missing markdown report at {report_md}"
    assert report_json.exists(), f"Missing JSON report at {report_json}"


def test_tomato_comparison_manifest_hash_and_model_selection() -> None:
    """Verify split manifest hash and model selection in JSON report."""
    report_json = PROJECT_ROOT / "reports/tomato/image_comparison.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["crop"] == "tomato"
    assert data["split_manifest_sha256"] in EXPECTED_MANIFEST_HASHES
    assert data["selection_metric"] == "validation_macro_f1"

    selected = data["selected_model"]
    assert selected["model_name"] == "tomato_efficientnet_b0_v1.0.0"
    assert selected["model_family"] == "EfficientNetB0"
    assert selected["metrics"]["macro_f1"] > 0.87
    assert selected["parameter_count"] == 4021639

    # Check 11 per-class F1 scores
    for c in TOMATO_CLASSES:
        assert c in selected["per_class_f1"]
        assert 0.0 < selected["per_class_f1"][c] <= 1.0


def test_tomato_comparison_candidates_and_leaf_input() -> None:
    """Verify candidates ranking, metrics, and leaf-level input confirmation."""
    report_json = PROJECT_ROOT / "reports/tomato/image_comparison.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]
    assert len(candidates) == 4

    families = {c["model_family"] for c in candidates}
    assert families == {"EfficientNetB0", "CustomCNN", "RandomForest", "SVM"}

    for cand in candidates:
        assert "leaf" in cand["input_mode"].lower()
        assert cand["val_macro_f1"] > 0.0
        assert cand["val_accuracy"] > 0.0

    # Ensure rank ordering by val_macro_f1 descending
    f1s = [c["val_macro_f1"] for c in candidates]
    assert f1s == sorted(f1s, reverse=True)
