"""Unit tests for trained Spinach EfficientNetB0 model and validation reports (T5-025)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.image_models.efficientnet import build_efficientnet_b0  # noqa: E402
from ml.labels import IMAGE_CLASSES  # noqa: E402

EXPECTED_MANIFEST_HASH = (
    "4c271d42de9b12de59b32ac5bb2ba4cb47821d7fc8dc9eebbc50cc41ad494f68"
)


def test_spinach_efficientnet_metadata_and_stages() -> None:
    """Verify spinach EfficientNetB0 metadata adheres to T5-025 acceptance criteria."""
    meta_path = (
        PROJECT_ROOT / "models/image/spinach/efficientnet_b0/v1.0.0/metadata.json"
    )
    if not meta_path.exists():
        pytest.skip("Model not yet trained; skipping artifact test.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["crop"] == "spinach"
    assert meta["family"] == "EfficientNetB0"
    assert meta["architecture"] == "efficientnet_b0"
    assert meta["classes"] == list(IMAGE_CLASSES["spinach"])
    assert meta["num_classes"] == 6
    assert meta["device_actually_used"] in ("cpu", "cuda:0")

    # Acceptance criterion 1: Both stages' settings are logged
    assert "stages" in meta
    stages = meta["stages"]
    assert "stage_1_frozen_backbone" in stages
    assert "stage_2_finetuning" in stages
    assert (
        stages["stage_1_frozen_backbone"]["trainable_parameters"]
        < stages["stage_2_finetuning"]["trainable_parameters"]
    )
    assert stages["stage_1_frozen_backbone"]["frozen_parameters"] > 0

    # Acceptance criterion 2: Same split manifest hash
    assert meta["split_manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert meta["split_counts"]["train"] == 12812
    assert meta["split_counts"]["val"] == 2745


def test_spinach_efficientnet_validation_report() -> None:
    """Verify spinach EfficientNetB0 validation report adheres to Section 18.2."""
    rep_path = (
        PROJECT_ROOT / "reports/spinach/spinach_efficientnet_b0_v1.0.0_validation.json"
    )
    md_path = (
        PROJECT_ROOT / "reports/spinach/spinach_efficientnet_b0_v1.0.0_validation.md"
    )
    if not rep_path.exists() or not md_path.exists():
        pytest.skip("Validation report not yet generated; skipping report test.")

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["crop"] == "spinach"
    assert rep["model_name"] == "spinach_efficientnet_b0_v1.0.0"
    assert rep["split_evaluated"] == "validation"
    assert rep["split_counts"]["test"] == 0

    metrics = rep["metrics"]
    assert metrics["macro_f1"] > 0.0
    assert metrics["accuracy"] > 0.0
    assert len(metrics["confusion_matrix_list"]) == 6

    per_class = metrics["per_class"]
    for cls_name in IMAGE_CLASSES["spinach"]:
        assert cls_name in per_class
        assert "f1" in per_class[cls_name]
        assert "precision" in per_class[cls_name]
        assert "recall" in per_class[cls_name]
        assert per_class[cls_name]["support"] > 0

    md_text = md_path.read_text(encoding="utf-8")
    assert "Evaluated Split:** `validation`" in md_text
    assert "Macro-F1" in md_text


def test_spinach_checkpoint_load_and_forward() -> None:
    """Verify saved checkpoint can be loaded and perform forward inference."""
    ckpt_path = PROJECT_ROOT / "models/image/spinach/efficientnet_b0/v1.0.0/model.pt"
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not yet saved; skipping forward test.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint

    model = build_efficientnet_b0(crop="spinach", weights="none")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 6)
