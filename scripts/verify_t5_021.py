"""Verification script for T5-021: Train the custom CNN on lettuce leaf crops."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t5_021_lettuce_cnn() -> None:
    model_dir = PROJECT_ROOT / "models/image/lettuce/custom_cnn/v1.0.0"
    checkpoint_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    report_md_path = PROJECT_ROOT / "reports/lettuce/lettuce_custom_cnn_v1.0.0_validation.md"
    report_json_path = PROJECT_ROOT / "reports/lettuce/lettuce_custom_cnn_v1.0.0_validation.json"

    print("=== Verifying T5-021: Lettuce Custom CNN Model & Report ===")

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

    assert meta["crop"] == "lettuce", f"Expected lettuce, got {meta.get('crop')}"
    assert meta["family"] == "CustomCNN"
    assert "device_actually_used" in meta and meta["device_actually_used"] in ("cuda:0", "cpu")
    assert "hyperparameters" in meta
    for hp in ("batch_size", "initial_lr", "optimizer", "early_stopping_patience", "seed"):
        assert hp in meta["hyperparameters"], f"Missing hyperparameter {hp}"
    assert meta["val_macro_f1"] > 0.60, f"Expected Macro-F1 > 0.60, got {meta['val_macro_f1']}"
    print(f"[PASS] Metadata valid: device_actually_used={meta['device_actually_used']}, val_macro_f1={meta['val_macro_f1']:.4f}")

    # 3. Check report validation split isolation
    with open(report_json_path, "r", encoding="utf-8") as f:
        rep_json = json.load(f)

    assert rep_json["split_evaluated"] == "validation", f"Report must evaluate validation split only, got {rep_json['split_evaluated']}"
    assert rep_json["split_counts"]["test"] == 0, "Test set must not be accessed or reported"
    assert "macro_f1" in rep_json["metrics"]
    assert "confusion_matrix" in rep_json["metrics"]
    assert "per_class" in rep_json["metrics"]

    with open(report_md_path, "r", encoding="utf-8") as f:
        rep_md = f.read()

    assert "Evaluated Split:** `validation`" in rep_md
    assert "Test (Held-Out) | 0" in rep_md
    assert "Macro-F1" in rep_md
    assert "Confusion Matrix" in rep_md
    print("[PASS] Report validates holdout validation split only (test remains untouched).")

    print("\n>>> All T5-021 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t5_021_lettuce_cnn()
