"""Verification script for Task T5-022: EfficientNetB0 transfer-learning model.

Verifies:
1. Head output size strictly matches target crop class count (lettuce=4, spinach=6, tomato=11).
2. Freeze-then-unfreeze schedule is configurable, transitions as scheduled, and logs actions.
3. Weights identifier, input size (224x224), normalisation, and licence terms saved in metadata.
4. Forward pass and backpropagation verification on synthetic input tensor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.image_models.efficientnet import (  # noqa: E402
    DEFAULT_WEIGHTS_IDENTIFIER,
    FreezeScheduleConfig,
    build_efficientnet_b0,
)
from ml.labels import IMAGE_CLASSES  # noqa: E402


def verify_t5_022() -> None:
    print("=== Verifying T5-022: EfficientNetB0 Transfer-Learning Model ===")

    # 1. Acceptance Criterion 1: Head output size matches class count
    for crop_name, expected_classes in IMAGE_CLASSES.items():
        expected_count = len(expected_classes)
        model = build_efficientnet_b0(
            crop=crop_name, weights="none"
        )  # Use weights="none" for fast unit testing

        assert model.num_classes == expected_count, (
            f"Expected {expected_count} classes for {crop_name}, got {model.num_classes}"
        )
        assert model.classifier[-1].out_features == expected_count, (
            f"Head out_features {model.classifier[-1].out_features} does not match {expected_count}"
        )
        assert model.crop == crop_name
        assert model.classes == list(expected_classes)

        # Forward pass shape check
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, expected_count), (
            f"Expected output shape (2, {expected_count}), got {output.shape}"
        )
        print(
            f"[PASS] Head output size matches class count for '{crop_name}' ({expected_count} classes)."
        )

    # Custom num_classes
    custom_model = build_efficientnet_b0(num_classes=8, weights="none")
    assert custom_model.num_classes == 8
    assert custom_model.classifier[-1].out_features == 8
    print("[PASS] Head output size verified for custom num_classes (8 classes).")

    # 2. Acceptance Criterion 2: Freeze/unfreeze schedule is configurable and logged
    sched_cfg = FreezeScheduleConfig(
        initial_freeze=True,
        unfreeze_at_epoch=3,
        unfreeze_mode="all",
    )
    sched_model = build_efficientnet_b0(
        crop="spinach", freeze_schedule=sched_cfg, weights="none"
    )

    counts_frozen = sched_model.count_parameters()
    # When frozen, only classifier head is trainable
    assert counts_frozen["frozen"] > 0, "No frozen parameters after initial_freeze=True"
    head_params = sum(p.numel() for p in sched_model.classifier.parameters())
    assert counts_frozen["trainable"] == head_params, (
        f"Trainable params ({counts_frozen['trainable']}) should match head ({head_params})"
    )

    # Epoch 1 & 2: step_schedule returns False (no transition)
    assert not sched_model.step_schedule(epoch=1)
    assert not sched_model.step_schedule(epoch=2)
    assert sched_model.count_parameters()["trainable"] == head_params

    # Epoch 3: step_schedule returns True and unfreezes all parameters
    assert sched_model.step_schedule(epoch=3)
    counts_unfrozen = sched_model.count_parameters()
    assert counts_unfrozen["frozen"] == 0
    assert counts_unfrozen["trainable"] == counts_unfrozen["total"]
    assert len(sched_model.schedule_history) >= 2
    print(
        "[PASS] Freeze/unfreeze schedule verified: initial freeze and unfreeze at epoch 3 logged."
    )

    # Test top_blocks unfreeze mode
    top_sched_cfg = FreezeScheduleConfig(
        initial_freeze=True,
        unfreeze_at_epoch=2,
        unfreeze_mode="top_blocks",
        num_unfrozen_blocks=2,
    )
    top_model = build_efficientnet_b0(
        crop="lettuce", freeze_schedule=top_sched_cfg, weights="none"
    )
    top_model.step_schedule(epoch=2)
    top_counts = top_model.count_parameters()
    assert top_counts["trainable"] > head_params, (
        "Trainable params did not increase after unfreezing top blocks"
    )
    assert top_counts["frozen"] > 0, (
        "Some early blocks should remain frozen in top_blocks mode"
    )
    print(
        f"[PASS] Top blocks unfreezing verified: {top_counts['trainable']} trainable, "
        f"{top_counts['frozen']} frozen."
    )

    # 3. Acceptance Criterion 3: Weights identifier saved in metadata
    pretrained_model = build_efficientnet_b0(crop="tomato", weights="DEFAULT")
    meta = pretrained_model.get_metadata()

    assert meta["model_family"] == "EfficientNetB0"
    assert meta["architecture"] == "efficientnet_b0"
    assert meta["weights_identifier"] == DEFAULT_WEIGHTS_IDENTIFIER, (
        f"Expected weights_identifier '{DEFAULT_WEIGHTS_IDENTIFIER}', got '{meta['weights_identifier']}'"
    )
    assert "weights_license" in meta and "ImageNet" in meta["weights_license"]
    assert meta["input_size"] == [3, 224, 224]
    assert meta["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert meta["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert meta["crop"] == "tomato"
    assert meta["num_classes"] == 11
    assert "total_parameters" in meta
    assert "trainable_parameters" in meta
    assert "freeze_schedule" in meta
    assert "schedule_history" in meta
    print(
        f"[PASS] Metadata verified: weights_identifier='{meta['weights_identifier']}', "
        f"input_size={meta['input_size']}, licence documented."
    )

    # 4. Verify forward & backward gradient flow
    model_grad = build_efficientnet_b0(crop="lettuce", weights="none")
    model_grad.train()
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 2])
    out = model_grad(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()

    # Head parameters must receive gradients
    has_grad = any(
        p.grad is not None and torch.norm(p.grad) > 0
        for p in model_grad.classifier.parameters()
    )
    assert has_grad, "Head parameters did not receive gradients during backpropagation"
    print("[PASS] Gradient flow through head verified during backpropagation.")

    # 5. Verify shape table profiling
    shape_table = pretrained_model.get_shape_table()
    assert len(shape_table) > 5, "Shape table contains too few layers"
    assert shape_table[-1]["layer"] == "classifier_head"
    assert shape_table[-1]["output_shape"] == [1, 11]
    print(
        f"[PASS] Shape table verified ({len(shape_table)} layers profiled, final shape [1, 11])."
    )

    print("=== All T5-022 Acceptance Criteria Successfully Verified ===")


if __name__ == "__main__":
    verify_t5_022()
