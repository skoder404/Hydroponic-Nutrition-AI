"""Verification script for T5-024: Train the custom CNN on spinach leaf images."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t5_024_spinach_cnn() -> None:
    model_dir = PROJECT_ROOT / "models/image/spinach/custom_cnn/v1.0.0"
    checkpoint_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    report_md_path = PROJECT_ROOT / "reports/spinach/spinach_custom_cnn_v1.0.0_validation.md"
    report_json_path = PROJECT_ROOT / "reports/spinach/spinach_custom_cnn_v1.0.0_validation.json"

    print("=== Verifying T5-024: Spinach Custom CNN Model & Report ===")

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

    assert meta["crop"] == "spinach", f"Expected spinach, got {meta.get('crop')}"
    assert meta["family"] == "CustomCNN"
    assert meta["num_classes"] == 6, f"Expected 6 classes for spinach, got {meta.get('num_classes')}"

    # Criterion 2: Input type (leaf image as provided) recorded in metadata
    assert "input_type" in meta, "Input type missing in metadata"
    assert "leaf image as provided" in meta["input_type"].lower(), (
        f"Expected 'leaf image as provided' in input_type, got {meta['input_type']}"
    )
    print(f"[PASS] Input type recorded in metadata: '{meta['input_type']}'")

    # Criterion 3: Device actually used recorded
    assert "device_actually_used" in meta and meta["device_actually_used"] in ("cuda:0", "cpu"), (
        f"Invalid device_actually_used: {meta.get('device_actually_used')}"
    )
    print(f"[PASS] Device actually used recorded: '{meta['device_actually_used']}'")

    # 3. Check report validation split isolation and per-class metrics
    with open(report_json_path, "r", encoding="utf-8") as f:
        rep_json = json.load(f)

    # Criterion 1: Validation report produced with per-class metrics
    assert rep_json["split_evaluated"] == "validation", (
        f"Report must evaluate validation split only, got {rep_json['split_evaluated']}"
    )
    assert rep_json["split_counts"]["test"] == 0, "Test set must not be accessed or reported"
    assert "per_class" in rep_json["metrics"], "per_class metrics missing in report"
    assert len(rep_json["metrics"]["per_class"]) == 6, "Expected 6 per-class metrics entries"
    assert "macro_f1" in rep_json["metrics"]
    assert "confusion_matrix" in rep_json["metrics"]

    with open(report_md_path, "r", encoding="utf-8") as f:
        rep_md = f.read()

    assert "Evaluated Split:** `validation`" in rep_md
    assert "Test (Held-Out) | 0" in rep_md
    assert "Macro-F1" in rep_md
    assert "Per-Class Performance Breakdown" in rep_md
    print("[PASS] Validation report produced with per-class metrics (holdout validation only).")

    print(f"\n>>> All T5-024 acceptance criteria verified successfully! (Best Val Macro-F1: {meta['val_macro_f1']:.4f}) <<<")


if __name__ == "__main__":
    test_t5_024_spinach_cnn()

