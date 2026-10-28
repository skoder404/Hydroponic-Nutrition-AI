"""Verification script for T5-026: Train the custom CNN on tomato leaf images."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t5_026_tomato_cnn() -> None:
    model_dir = PROJECT_ROOT / "models/image/tomato/custom_cnn/v1.0.0"
    checkpoint_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    report_md_path = PROJECT_ROOT / "reports/tomato/tomato_custom_cnn_v1.0.0_validation.md"
    report_json_path = PROJECT_ROOT / "reports/tomato/tomato_custom_cnn_v1.0.0_validation.json"

    print("=== Verifying T5-026: Tomato Custom CNN Model & Report ===")

    # 1. Check artifact existence
    assert checkpoint_path.exists(), f"Model checkpoint missing at {checkpoint_path}"
    assert checkpoint_path.stat().st_size > 100_000, "Checkpoint file appears empty or incomplete"
    assert metadata_path.exists(), f"Metadata missing at {metadata_path}"
    assert report_md_path.exists(), f"Report markdown missing at {report_md_path}"
    assert report_json_path.exists(), f"Report json missing at {report_json_path}"
    print("[PASS] Checkpoint, metadata, and reports exist.")

    # 2. Check metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["crop"] == "tomato", f"Expected tomato, got {meta.get('crop')}"
    assert meta["family"] == "CustomCNN"
    assert meta["num_classes"] == 11, f"Expected 11 classes for tomato, got {meta.get('num_classes')}"

    # Criterion 3: Input type recorded
    assert "input_type" in meta, "Input type missing in metadata"
    assert "leaf image as provided" in meta["input_type"].lower(), (
        f"Expected 'leaf image as provided' in input_type, got {meta['input_type']}"
    )
    print(f"[PASS] Input type recorded: '{meta['input_type']}'")

    # Device actually used
    assert "device_actually_used" in meta and meta["device_actually_used"] in ("cuda:0", "cpu"), (
        f"Invalid device_actually_used: {meta.get('device_actually_used')}"
    )
    print(f"[PASS] Device actually used recorded: '{meta['device_actually_used']}'")

    # 3. Check report validation split isolation, per-class F1 for all 11 classes, and class support
    with open(report_json_path, "r", encoding="utf-8") as f:
        rep_json = json.load(f)

    assert rep_json["split_evaluated"] == "validation", (
        f"Report must evaluate validation split only, got {rep_json['split_evaluated']}"
    )
    assert rep_json["split_counts"]["test"] == 0, "Test set must not be accessed or reported"

    # Criterion 1: Validation report with per-class F1 for all 11 classes
    per_class = rep_json["metrics"].get("per_class", {})
    assert len(per_class) == 11, f"Expected 11 per-class entries, got {len(per_class)}"
    assert "Tomato_Powdery_Mildew" in per_class, "Tomato_Powdery_Mildew missing in per-class metrics"

    for cls_name, cls_metrics in per_class.items():
        assert "f1" in cls_metrics, f"F1 score missing for class {cls_name}"
        # Criterion 2: Class support shown
        assert "support" in cls_metrics and cls_metrics["support"] > 0, (
            f"Class support missing or 0 for {cls_name}: {cls_metrics.get('support')}"
        )

    print("[PASS] Validation report has per-class F1 for all 11 classes.")
    print("[PASS] Class support shown for each of the 11 classes.")

    with open(report_md_path, "r", encoding="utf-8") as f:
        rep_md = f.read()

    assert "Evaluated Split:** `validation`" in rep_md
    assert "Test (Held-Out) | 0" in rep_md
    assert "Tomato_Powdery_Mildew" in rep_md
    assert "Support" in rep_md
    assert "Per-Class Performance Breakdown" in rep_md

    print(f"\n>>> All T5-026 acceptance criteria verified successfully! (Best Val Macro-F1: {meta['val_macro_f1']:.4f}) <<<")


if __name__ == "__main__":
    test_t5_026_tomato_cnn()

