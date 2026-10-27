"""Verification script for T5-009: Per-leaf mask validation.

Verifies acceptance criteria:
1. Empty, tiny, and whole-image masks are rejected.
2. Thresholds are config values with a documented origin.
3. Every verdict carries a reason string.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ml.segmentation.mask_validation import (
    MaskValidationConfig,
    MaskVerdict,
    validate_mask,
)


def verify_thresholds_and_docstrings() -> None:
    config = MaskValidationConfig()
    doc = MaskValidationConfig.__doc__
    assert doc is not None, (
        "MaskValidationConfig must have a docstring documenting threshold origins"
    )

    required_thresholds = [
        ("min_area_ratio", config.min_area_ratio),
        ("max_area_ratio", config.max_area_ratio),
        ("max_border_contact_ratio", config.max_border_contact_ratio),
        ("min_component_ratio", config.min_component_ratio),
    ]

    for name, val in required_thresholds:
        assert name in doc, (
            f"Origin of threshold '{name}' not documented in config docstring"
        )
        assert isinstance(val, (int, float)), f"Threshold {name} must be numeric"
    print("MaskValidationConfig thresholds and documented origins: VERIFIED.")


def verify_rejection_rules() -> None:
    h, w = 300, 300

    # 1. Empty mask
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    res_empty = validate_mask(empty_mask)
    assert res_empty.verdict == MaskVerdict.REJECT, "Empty mask must be REJECTED"
    assert not res_empty.is_valid
    assert len(res_empty.reason) > 0, "Verdict must carry a reason string"
    print(f"Empty mask rejection: VERIFIED ('{res_empty.reason}')")

    # 2. Tiny mask (5x5 = 25 px, < 1%)
    tiny_mask = np.zeros((h, w), dtype=np.uint8)
    tiny_mask[50:55, 50:55] = 1
    res_tiny = validate_mask(tiny_mask)
    assert res_tiny.verdict == MaskVerdict.REJECT, "Tiny mask must be REJECTED"
    assert not res_tiny.is_valid
    assert len(res_tiny.reason) > 0, "Verdict must carry a reason string"
    print(f"Tiny mask rejection: VERIFIED ('{res_tiny.reason}')")

    # 3. Whole-image mask (95% coverage)
    whole_mask = np.ones((h, w), dtype=np.uint8)
    whole_mask[0:20, 0:20] = 0
    res_whole = validate_mask(whole_mask)
    assert res_whole.verdict == MaskVerdict.REJECT, "Whole-image mask must be REJECTED"
    assert not res_whole.is_valid
    assert len(res_whole.reason) > 0, "Verdict must carry a reason string"
    print(f"Whole-image mask rejection: VERIFIED ('{res_whole.reason}')")

    # 4. Valid mask
    valid_mask = np.zeros((h, w), dtype=np.uint8)
    valid_mask[50:200, 50:200] = 1  # 150x150 = 22500 px (25% area)
    res_valid = validate_mask(valid_mask)
    assert res_valid.verdict == MaskVerdict.OK, (
        "Centered coherent leaf must be ACCEPTED"
    )
    assert res_valid.is_valid
    assert len(res_valid.reason) > 0, "Verdict must carry a reason string"
    print(f"Valid mask acceptance: VERIFIED ('{res_valid.reason}')")


def main() -> int:
    print("=== Verifying T5-009: Per-Leaf Mask Validation ===")
    verify_thresholds_and_docstrings()
    verify_rejection_rules()
    print("=== All T5-009 acceptance criteria PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
