"""Unit test for T5-028: Lettuce image model comparison and selection."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_t5_028_comparison_report_exists():
    report_md = PROJECT_ROOT / "reports/lettuce/image_comparison.md"
    report_json = PROJECT_ROOT / "reports/lettuce/image_comparison.json"
    assert report_md.exists()
    assert report_json.exists()


def test_t5_028_manifest_hash_consistency():
    report_json = PROJECT_ROOT / "reports/lettuce/image_comparison.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    expected_hash = "5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52"
    assert data["split_manifest_sha256"] == expected_hash


def test_t5_028_selected_model_metrics():
    report_json = PROJECT_ROOT / "reports/lettuce/image_comparison.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected = data["selected_model"]
    assert selected["model_name"] == "lettuce_efficientnet_b0_v1.0.0"
    assert selected["model_family"] == "EfficientNetB0"
    assert selected["metrics"]["macro_f1"] > 0.90
    assert selected["metrics"]["accuracy"] > 0.94
    assert len(selected["per_class_f1"]) == 4


def test_t5_028_leaf_level_input_all_families():
    report_json = PROJECT_ROOT / "reports/lettuce/image_comparison.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["candidates"]) == 4
    for candidate in data["candidates"]:
        assert "leaf" in candidate["input_mode"].lower()

