"""Unit tests for EfficientNetB0 transfer learning module (T5-022)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.image_models.efficientnet import (  # noqa: E402
    DEFAULT_INPUT_SIZE,
    DEFAULT_WEIGHTS_IDENTIFIER,
    EfficientNetB0,
    EfficientNetConfig,
    FreezeScheduleConfig,
    build_efficientnet_b0,
)
from ml.labels import IMAGE_CLASSES  # noqa: E402


@pytest.mark.parametrize("crop", ["lettuce", "spinach", "tomato"])
def test_head_output_size_matches_crop_class_count(crop: str) -> None:
    """Acceptance criterion 1: Head output size matches crop class count."""
    expected_classes = len(IMAGE_CLASSES[crop])
    model = build_efficientnet_b0(crop=crop, weights="none")

    assert model.num_classes == expected_classes
    assert model.classifier[-1].out_features == expected_classes

    # Test forward output shape
    x = torch.randn(2, 3, *DEFAULT_INPUT_SIZE)
    out = model(x)
    assert out.shape == (2, expected_classes)


def test_custom_num_classes() -> None:
    """Verify initialization with custom class count."""
    model = build_efficientnet_b0(num_classes=7, weights="none")
    assert model.num_classes == 7
    assert model.classifier[-1].out_features == 7


def test_freeze_unfreeze_schedule_configurable_and_logged() -> None:
    """Acceptance criterion 2: Freeze/unfreeze schedule is configurable and logged."""
    schedule = FreezeScheduleConfig(
        initial_freeze=True,
        unfreeze_at_epoch=4,
        unfreeze_mode="all",
    )
    model = build_efficientnet_b0(
        crop="spinach", freeze_schedule=schedule, weights="none"
    )

    # Initial state: backbone is frozen, classifier head is trainable
    counts_init = model.count_parameters()
    head_params = sum(p.numel() for p in model.classifier.parameters())
    assert counts_init["trainable"] == head_params
    assert counts_init["frozen"] > 0
    assert len(model.schedule_history) >= 1
    assert model.schedule_history[0]["action"] == "freeze_backbone"

    # Step before threshold
    assert not model.step_schedule(epoch=1)
    assert not model.step_schedule(epoch=3)
    assert model.count_parameters()["trainable"] == head_params

    # Step at threshold: transitions to unfreeze
    assert model.step_schedule(epoch=4)
    counts_unfrozen = model.count_parameters()
    assert counts_unfrozen["frozen"] == 0
    assert counts_unfrozen["trainable"] == counts_unfrozen["total"]
    assert len(model.schedule_history) >= 2
    assert model.schedule_history[1]["action"] == "unfreeze_backbone"
    assert model.schedule_history[1]["epoch"] == 4


def test_unfreeze_top_blocks() -> None:
    """Verify partial unfreezing of top blocks."""
    schedule = FreezeScheduleConfig(
        initial_freeze=True,
        unfreeze_at_epoch=2,
        unfreeze_mode="top_blocks",
        num_unfrozen_blocks=2,
    )
    model = build_efficientnet_b0(
        crop="lettuce", freeze_schedule=schedule, weights="none"
    )

    head_params = sum(p.numel() for p in model.classifier.parameters())
    assert model.count_parameters()["trainable"] == head_params

    model.step_schedule(epoch=2)
    counts = model.count_parameters()
    assert counts["trainable"] > head_params
    assert counts["frozen"] > 0


def test_weights_identifier_and_metadata() -> None:
    """Acceptance criterion 3: Weights identifier, input size, and normalization saved in metadata."""
    model = build_efficientnet_b0(crop="lettuce", weights="DEFAULT")
    meta = model.get_metadata()

    assert meta["model_family"] == "EfficientNetB0"
    assert meta["architecture"] == "efficientnet_b0"
    assert meta["weights_identifier"] == DEFAULT_WEIGHTS_IDENTIFIER
    assert "ImageNet" in meta["weights_license"]
    assert meta["input_size"] == [3, 224, 224]
    assert meta["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert meta["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert meta["num_classes"] == 4
    assert meta["crop"] == "lettuce"
    assert meta["total_parameters"] > 0
    assert "freeze_schedule" in meta
    assert "schedule_history" in meta


def test_shape_table_activation_shapes() -> None:
    """Verify layer-by-layer shape table activation tracking."""
    model = build_efficientnet_b0(crop="tomato", weights="none")
    table = model.get_shape_table(input_shape=(1, 3, 224, 224))

    assert len(table) > 5
    assert table[-1]["layer"] == "classifier_head"
    assert table[-1]["output_shape"] == [1, 11]


def test_backward_gradient_flow() -> None:
    """Verify gradient flow through trainable parameters."""
    model = build_efficientnet_b0(crop="spinach", weights="none")
    model.train()
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1])

    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()

    # Head parameters must have gradients
    for p in model.classifier.parameters():
        assert p.grad is not None


def test_invalid_crop_raises_error() -> None:
    """Verify unknown crop string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown crop 'apple'"):
        build_efficientnet_b0(crop="apple")


def test_missing_crop_and_num_classes_raises_error() -> None:
    """Verify missing both crop and num_classes raises ValueError."""
    with pytest.raises(
        ValueError, match="Either 'num_classes' or 'crop' must be provided"
    ):
        build_efficientnet_b0()


def test_direct_instantiation_with_config() -> None:
    """Verify direct initialization of EfficientNetB0 using EfficientNetConfig."""
    cfg = EfficientNetConfig(crop="lettuce", dropout_rate=0.25, weights="none")
    model = EfficientNetB0(config=cfg)
    assert model.num_classes == 4
    assert model.dropout_rate == 0.25
    assert cfg.to_dict()["crop"] == "lettuce"
