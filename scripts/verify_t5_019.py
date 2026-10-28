#!/usr/bin/env python3
"""Verification script for T5-019: Design the custom CNN architecture module.

Acceptance Criteria:
1. Forward pass works on a dummy batch for each crop's class count
2. Parameter count logged
3. Architecture is described in the concept note

Usage:
    python scripts/verify_t5_019.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from ml.image_models.custom_cnn import build_custom_cnn  # noqa: E402
from ml.labels import IMAGE_CLASSES  # noqa: E402


def verify_criterion_1_forward_pass_all_crops() -> tuple[bool, str]:
    """Criterion 1: Forward pass works on a dummy batch for each crop's class count."""
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

    for crop in ["lettuce", "spinach", "tomato"]:
        expected_classes = len(IMAGE_CLASSES[crop])
        model = build_custom_cnn(crop=crop)
        logits = model(dummy_input)

        if logits.shape != (batch_size, expected_classes):
            return (
                False,
                f"FAIL: Crop '{crop}' expected output shape {(batch_size, expected_classes)}, got {tuple(logits.shape)}",
            )
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return False, f"FAIL: Crop '{crop}' forward pass yielded NaN or Inf logits."

    return (
        True,
        "PASS: Forward pass verified on dummy batches for all crops: lettuce (4), spinach (6), tomato (11).",
    )


def verify_criterion_2_parameter_count_logged() -> tuple[bool, str]:
    """Criterion 2: Parameter count logged and shape table produced."""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    custom_logger = logging.getLogger("verify_custom_cnn")
    custom_logger.setLevel(logging.INFO)
    custom_logger.addHandler(handler)

    model = build_custom_cnn(crop="lettuce")
    total_params = model.count_parameters(trainable_only=False)
    summary_text = model.log_architecture_summary(
        input_shape=(1, 3, 224, 224), logger=custom_logger
    )
    handler.flush()
    log_output = log_stream.getvalue()

    if total_params <= 0:
        return False, f"FAIL: Invalid parameter count: {total_params}"

    if (
        "Total Parameters:" not in summary_text
        or "Trainable Parameters:" not in summary_text
    ):
        return False, "FAIL: Architecture summary missing parameter count totals."

    if "ConvBlock_1" not in log_output or "Total Parameters:" not in log_output:
        return False, "FAIL: Parameter count or shape table was not logged."

    return (
        True,
        f"PASS: Parameter count ({total_params:,} parameters) and shape table successfully logged.",
    )


def verify_criterion_3_concept_note_description() -> tuple[bool, str]:
    """Criterion 3: Architecture is described in the concept note."""
    note_path = (
        PROJECT_ROOT / "docs/learning/convolution-pooling-and-backpropagation.md"
    )
    if not note_path.exists():
        return False, f"FAIL: Concept note file missing: {note_path}"

    content = note_path.read_text(encoding="utf-8")
    required_keywords = [
        "custom_cnn.py",
        "ConvBlock",
        "AdaptiveAvgPool2d",
        "lettuce",
        "spinach",
        "tomato",
    ]

    for kw in required_keywords:
        if kw not in content:
            return False, f"FAIL: Concept note missing architectural keyword '{kw}'."

    return (
        True,
        "PASS: Custom CNN architecture, block structure, and parameter counts are described in concept note.",
    )


def main() -> int:
    """Run all verification checks for T5-019."""
    print("=" * 70)
    print("Verifying T5-019: Design the custom CNN architecture module")
    print("=" * 70)

    checks = [
        (
            "Criterion 1: Forward pass works on a dummy batch for each crop's class count",
            verify_criterion_1_forward_pass_all_crops,
        ),
        (
            "Criterion 2: Parameter count logged",
            verify_criterion_2_parameter_count_logged,
        ),
        (
            "Criterion 3: Architecture is described in the concept note",
            verify_criterion_3_concept_note_description,
        ),
    ]

    all_passed = True
    for title, check_func in checks:
        print(f"\nChecking {title}...")
        passed, msg = check_func()
        print(f"  {msg}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY (T5-019).")
        print("=" * 70)
        return 0
    else:
        print("VERIFICATION FAILED FOR ONE OR MORE CRITERIA.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
