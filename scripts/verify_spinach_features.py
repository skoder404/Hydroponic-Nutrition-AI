#!/usr/bin/env python3
"""Verification script for T5-005: Implement spinach feature extractor (6 features).

Acceptance Criteria:
1. Order matches Overview §16.2
2. Denominator is documented and stored in metadata
3. Empty mask raises an error

Usage:
    python scripts/verify_spinach_features.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml.features.spinach as spinach_module
from ml.features.common import EmptyMaskError, FeatureVector
from ml.features.spinach import (
    DEFAULT_SPINACH_CONFIG,
    SPINACH_FEATURE_NAMES,
    SpinachFeatureConfig,
    extract_spinach_features,
)


def verify_order_matches_overview() -> tuple[bool, str]:
    """Criterion 1: Order matches Overview §16.2."""
    expected_order = (
        "exg_mean",
        "exr_mean",
        "yellow_pixel_ratio",
        "brown_pixel_ratio",
        "relative_leaf_area",
        "glcm_contrast",
    )

    if SPINACH_FEATURE_NAMES != expected_order:
        return False, f"Expected {expected_order}, got {SPINACH_FEATURE_NAMES}"

    # Generate synthetic image and circular leaf mask
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [30, 180, 50]  # Healthy green leaf

    mask = np.zeros((h, w), dtype=bool)
    mask[25:75, 25:75] = True  # 50x50 center square

    vec = extract_spinach_features(img, mask)

    if not isinstance(vec, FeatureVector):
        return False, f"Expected FeatureVector, got {type(vec).__name__}"

    if tuple(vec.feature_names) != expected_order:
        return False, f"FeatureVector feature_names mismatch: expected {expected_order}, got {tuple(vec.feature_names)}"

    if len(vec.values) != 6:
        return False, f"Expected 6 feature values, got {len(vec.values)}"

    # Check mapping in to_dict()
    as_dict = vec.to_dict()
    if list(as_dict.keys()) != list(expected_order):
        return False, f"FeatureVector.to_dict() keys mismatch: {list(as_dict.keys())}"

    return True, (
        f"Verified 6-feature ordering matches Overview §16.2 exactly: "
        f"{', '.join(expected_order)}"
    )


def verify_denominator_documented_and_stored() -> tuple[bool, str]:
    """Criterion 2: Denominator is documented and stored in metadata."""
    # 1. Check docstring documentation
    doc = spinach_module.__doc__
    if not doc:
        return False, "Module docstring missing in ml.features.spinach."

    if "relative_leaf_area" not in doc or "denominator" not in doc.lower():
        return False, "Module docstring does not document relative_leaf_area denominator."

    if "H * W" not in doc and "total_image_pixels" not in doc:
        return False, "Module docstring does not explicitly specify total image pixels (H * W) as denominator."

    func_doc = inspect.getdoc(extract_spinach_features)
    if not func_doc or "relative_leaf_area" not in func_doc:
        return False, "extract_spinach_features docstring missing relative_leaf_area explanation."

    # 2. Check metadata in extraction_config
    h, w = 120, 80
    total_pixels = h * w
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[10:50, 10:50] = True  # 40 x 40 = 1600 pixels
    fg_count = 40 * 40

    vec = extract_spinach_features(img, mask)
    config_dict = vec.extraction_config

    if "relative_leaf_area_denominator" not in config_dict:
        return False, "extraction_config missing key 'relative_leaf_area_denominator'."

    denom_str = str(config_dict["relative_leaf_area_denominator"])
    if "total_image_pixels" not in denom_str:
        return False, f"Unexpected denominator metadata value: '{denom_str}'"

    # 3. Check mathematical accuracy of relative_leaf_area
    expected_ratio = fg_count / float(total_pixels)
    actual_ratio = vec.values[SPINACH_FEATURE_NAMES.index("relative_leaf_area")]

    if abs(actual_ratio - expected_ratio) > 1e-7:
        return False, f"relative_leaf_area calculation mismatch: expected {expected_ratio}, got {actual_ratio}"

    return True, (
        f"Denominator is thoroughly documented in docstrings and stored in extraction metadata: "
        f"'{denom_str}'. Value verified: {fg_count}/{total_pixels} = {actual_ratio:.6f}."
    )


def verify_empty_mask_raises_error() -> tuple[bool, str]:
    """Criterion 3: Empty mask raises an error."""
    h, w = 64, 64
    img = np.full((h, w, 3), 100, dtype=np.uint8)

    # All false boolean mask
    mask_bool = np.zeros((h, w), dtype=bool)

    try:
        extract_spinach_features(img, mask_bool)
        return False, "Expected EmptyMaskError on all-False mask, but no error was raised."
    except EmptyMaskError as e:
        msg_bool = str(e)
    except Exception as e:
        return False, f"Expected EmptyMaskError, got {type(e).__name__}: {e}"

    # All zero uint8 mask
    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
    try:
        extract_spinach_features(img, mask_uint8)
        return False, "Expected EmptyMaskError on all-zero uint8 mask, but no error was raised."
    except EmptyMaskError as e:
        msg_uint8 = str(e)
    except Exception as e:
        return False, f"Expected EmptyMaskError, got {type(e).__name__}: {e}"

    return True, (
        f"Empty mask correctly raised typed EmptyMaskError for both boolean and uint8 inputs: "
        f"'{msg_bool}'."
    )


def run_verification() -> bool:
    """Run all verification checks for T5-005."""
    print("=" * 80)
    print("VERIFY SPINACH FEATURE EXTRACTOR (T5-005)")
    print("=" * 80)
    print()

    checks = [
        ("Order matches Overview §16.2", verify_order_matches_overview),
        ("Denominator is documented and stored in metadata", verify_denominator_documented_and_stored),
        ("Empty mask raises an error", verify_empty_mask_raises_error),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {title}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print()
    print("-" * 80)
    if all_passed:
        print("All T5-005 acceptance criteria successfully verified!")
    else:
        print("One or more T5-005 checks failed.")
    print()

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
