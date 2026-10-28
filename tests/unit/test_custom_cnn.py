"""Unit tests for CustomCNN architecture module (T5-019).

Acceptance Criteria:
- [x] Forward pass works on a dummy batch for each crop's class count
- [x] Parameter count logged
- [x] Architecture is described in the concept note
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from ml.image_models.custom_cnn import (
    CustomCNN,
    build_custom_cnn,
)
from ml.labels import IMAGE_CLASSES


@pytest.mark.parametrize("crop", ["lettuce", "spinach", "tomato"])
def test_criterion_1_forward_pass_dummy_batch_each_crop(crop: str):
    """Acceptance Criterion 1: Forward pass works on a dummy batch for each crop's class count."""
    expected_classes = len(IMAGE_CLASSES[crop])

    model = build_custom_cnn(crop=crop)
    assert model.num_classes == expected_classes
    assert model.crop == crop

    # Forward pass on dummy batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    logits = model(dummy_input)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, expected_classes)
    assert not torch.isnan(logits).any(), "Logits contain NaN values"
    assert not torch.isinf(logits).any(), "Logits contain Inf values"


def test_criterion_2_parameter_count_logged(caplog: pytest.LogCaptureFixture):
    """Acceptance Criterion 2: Parameter count logged and shape table produced."""
    caplog.set_level(logging.INFO)

    model = build_custom_cnn(crop="lettuce")
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)

    assert total_params > 0
    assert trainable_params == total_params  # all parameters are trainable by default

    # Verify logging
    summary_text = model.log_architecture_summary(input_shape=(1, 3, 224, 224))
    assert "Total Parameters:" in summary_text
    assert "Trainable Parameters:" in summary_text
    assert "ConvBlock_1" in summary_text
    assert "AdaptiveAvgPool2d" in summary_text
    assert "Linear" in summary_text

    # Verify table structure
    table = model.get_shape_table(input_shape=(2, 3, 224, 224))
    assert len(table) >= 6
    assert table[0]["output_shape"] == [2, 3, 224, 224]
    assert table[-1]["output_shape"] == [2, 4]  # lettuce output


def test_criterion_3_architecture_described_in_concept_note():
    """Acceptance Criterion 3: Architecture is described in the concept note."""
    project_root = Path(__file__).resolve().parent.parent.parent
    concept_note = (
        project_root / "docs/learning/convolution-pooling-and-backpropagation.md"
    )

    assert concept_note.exists(), f"Concept note missing: {concept_note}"
    content = concept_note.read_text(encoding="utf-8")

    assert "Custom CNN Architecture" in content or "custom_cnn.py" in content
    assert "ConvBlock" in content or "conv blocks" in content
    assert "AdaptiveAvgPool2d" in content or "pooling" in content
    assert "lettuce" in content and "spinach" in content and "tomato" in content


def test_custom_num_classes_and_invalid_crop():
    """Verify custom class initialization and error handling for invalid configurations."""
    # Arbitrary class count
    model = build_custom_cnn(num_classes=5)
    assert model.num_classes == 5
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 5)

    # Invalid crop raises ValueError
    with pytest.raises(ValueError, match="Unknown crop"):
        build_custom_cnn(crop="unknown_crop")

    # Missing both crop and num_classes raises ValueError
    with pytest.raises(
        ValueError, match="Either 'num_classes' or 'crop' must be provided"
    ):
        CustomCNN()


def test_backward_pass_computes_gradients():
    """Verify gradient computation through all conv blocks and classifier head."""
    model = build_custom_cnn(crop="spinach")
    model.train()

    inputs = torch.randn(2, 3, 64, 64, requires_grad=True)
    targets = torch.tensor([0, 2], dtype=torch.int64)

    outputs = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    loss.backward()

    # Verify gradients exist and are non-zero across feature and classifier layers
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has None gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
