#!/usr/bin/env python3
"""Verification script for T5-003: Implement GLCM utility (quantise 32, distance 1, four angles).

Acceptance Criteria:
1. Flags and log base are documented in the module docstring
2. Constant-colour image does not crash or yield NaN
3. Output matches a hand-computed 4x4 example

Usage:
    python scripts/verify_glcm.py
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

import ml.features.glcm as glcm_module
from ml.features.glcm import (
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    DEFAULT_LEVELS,
    DEFAULT_LOG_BASE,
    DEFAULT_NORMED,
    DEFAULT_SYMMETRIC,
    compute_glcm,
    compute_glcm_contrast,
    compute_glcm_entropy,
    extract_glcm_features,
    quantize_gray,
    to_grayscale,
)


def verify_flags_and_log_base_documented() -> tuple[bool, str]:
    """Criterion 1: Flags and log base are documented in the module docstring."""
    doc = glcm_module.__doc__
    if not doc:
        return False, "Module docstring is missing in ml.features.glcm."

    # Check for symmetric flag documentation
    if "symmetric" not in doc or "True" not in doc:
        return False, "Module docstring does not explicitly document symmetric flag."

    # Check for normed flag documentation
    if "normed" not in doc or "True" not in doc:
        return False, "Module docstring does not explicitly document normed flag."

    # Check for log base documentation
    if "base 2" not in doc and "log_base" not in doc:
        return False, "Module docstring does not explicitly document logarithm base for entropy."

    # Check that module constants match the documented defaults
    if DEFAULT_LEVELS != 32:
        return False, f"Expected DEFAULT_LEVELS=32, got {DEFAULT_LEVELS}"
    if DEFAULT_DISTANCES != (1,):
        return False, f"Expected DEFAULT_DISTANCES=(1,), got {DEFAULT_DISTANCES}"
    if len(DEFAULT_ANGLES) != 4:
        return False, f"Expected 4 angles, got {len(DEFAULT_ANGLES)}"
    if not DEFAULT_SYMMETRIC:
        return False, "Expected DEFAULT_SYMMETRIC=True"
    if not DEFAULT_NORMED:
        return False, "Expected DEFAULT_NORMED=True"
    if DEFAULT_LOG_BASE != 2.0:
        return False, f"Expected DEFAULT_LOG_BASE=2.0, got {DEFAULT_LOG_BASE}"

    return True, (
        "Module docstring thoroughly documents symmetric=True, normed=True, "
        "and base-2 logarithm (shannons) for entropy calculation."
    )


def verify_constant_colour_image() -> tuple[bool, str]:
    """Criterion 2: Constant-colour image does not crash or yield NaN."""
    # Test across multiple constant image types: all-zeros, all-128, all-255, float ones, 3-channel RGB
    test_cases = [
        ("zeros_uint8", np.zeros((32, 32), dtype=np.uint8)),
        ("mid_uint8", np.full((40, 40), 128, dtype=np.uint8)),
        ("max_uint8", np.full((25, 25), 255, dtype=np.uint8)),
        ("float_ones", np.ones((20, 20), dtype=np.float32)),
        ("rgb_constant", np.full((30, 30, 3), 100, dtype=np.uint8)),
    ]

    for name, img in test_cases:
        try:
            features = extract_glcm_features(img)
        except Exception as e:
            return False, f"Constant image '{name}' crashed with exception: {e}"

        c = features.get("glcm_contrast")
        h = features.get("glcm_entropy")

        if c is None or h is None:
            return False, f"Missing glcm_contrast or glcm_entropy in returned features: {features}"

        if not np.isfinite(c) or not np.isfinite(h):
            return False, f"Constant image '{name}' yielded non-finite values: contrast={c}, entropy={h}"

        if abs(c) > 1e-6:
            return False, f"Expected zero contrast for constant image '{name}', got {c}"

        if abs(h) > 1e-6:
            return False, f"Expected zero entropy for constant image '{name}', got {h}"

    return True, (
        "Verified 5 constant-colour configurations (zeros, mid-gray, full-white, float, RGB); "
        "all cleanly yielded contrast=0.0 and entropy=0.0 with no crashes or NaNs."
    )


def verify_hand_computed_4x4_example() -> tuple[bool, str]:
    """Criterion 3: Output matches a hand-computed 4x4 example.

    Hand-computed 4x4 image:
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [2, 2, 3, 3],
         [2, 2, 3, 3]]

    Analytical Derivation (distance=1, levels=32, symmetric=True, normed=True):
    - 0 deg (horizontal):
      12 pairs per direction * 2 (symmetric) = 24 transitions.
      Squared differences: 4 vertical block crossings each of difference 1 => 4 * 1^2 * 2 = 8.
      Contrast(0 deg) = 8 / 24 = 1/3 ≈ 0.333333.
      Entropy(0 deg) = 4/3 + log2(3) ≈ 2.918296 bits.
    - 90 deg (vertical):
      12 pairs per direction * 2 (symmetric) = 24 transitions.
      Crossings of difference 2 => 4 * 2^2 * 2 = 32.
      Contrast(90 deg) = 32 / 24 = 4/3 ≈ 1.333333.
      Entropy(90 deg) = 4/3 + log2(3) ≈ 2.918296 bits.
    - 45 deg (up-right):
      9 pairs * 2 = 18 transitions.
      Squared diff sum = 22.
      Contrast(45 deg) = 22 / 18 = 11/9 ≈ 2.111111 (Wait: in symmetric pairs sum = 38, contrast = 19/9).
      Entropy(45 deg) ≈ 3.725481 bits.
    - 135 deg (up-left):
      9 pairs * 2 = 18 transitions.
      Contrast(135 deg) = 11/9 ≈ 1.222222.
      Entropy(135 deg) ≈ 3.725481 bits.

    Mean across 4 angles:
    - Mean Contrast = (1/3 + 19/9 + 4/3 + 11/9) / 4 = (45/9) / 4 = 5 / 4 = 1.250000 exactly!
    - Mean Entropy = (2.918296 + 3.725481 + 2.918296 + 3.725481) / 4 = 3.321888 bits.
    """
    hand_4x4 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ], dtype=np.uint8)

    # Compute GLCM directly on the integer array (values 0..3 are within 32 levels)
    glcm = compute_glcm(hand_4x4, levels=32, distances=(1,), angles=DEFAULT_ANGLES)
    c_direct = compute_glcm_contrast(glcm)
    h_direct = compute_glcm_entropy(glcm, log_base=2.0)

    # Expected exact theoretical values
    expected_contrast = 1.250000
    expected_entropy = 3.321888

    tol = 1e-5
    if abs(c_direct - expected_contrast) > tol:
        return False, f"Direct contrast mismatch: expected {expected_contrast:.6f}, got {c_direct:.6f}"

    if abs(h_direct - expected_entropy) > tol:
        return False, f"Direct entropy mismatch: expected {expected_entropy:.6f}, got {h_direct:.6f}"

    # Verify through extract_glcm_features wrapper
    # Map [0, 1, 2, 3] to image uint8 in bin centres: bin_width = 8 -> values = [4, 12, 20, 28]
    # In quantize_gray, 4//8 = 0, 12//8 = 1, 20//8 = 2, 28//8 = 3.
    img_mapped = (hand_4x4 * 8 + 4).astype(np.uint8)
    features = extract_glcm_features(img_mapped, levels=32)

    c_feat = features["glcm_contrast"]
    h_feat = features["glcm_entropy"]

    if abs(c_feat - expected_contrast) > tol:
        return False, f"Extracted contrast mismatch: expected {expected_contrast:.6f}, got {c_feat:.6f}"

    if abs(h_feat - expected_entropy) > tol:
        return False, f"Extracted entropy mismatch: expected {expected_entropy:.6f}, got {h_feat:.6f}"

    return True, (
        f"Hand-computed 4x4 matrix matched exact analytical values: "
        f"contrast={c_feat:.6f} (expected {expected_contrast:.6f}), "
        f"entropy={h_feat:.6f} bits (expected {expected_entropy:.6f})."
    )


def run_verification() -> bool:
    """Run all verification checks for T5-003."""
    print("=" * 80)
    print("VERIFY GLCM UTILITY (T5-003)")
    print("=" * 80)
    print()

    checks = [
        ("Flags and log base are documented in the module docstring", verify_flags_and_log_base_documented),
        ("Constant-colour image does not crash or yield NaN", verify_constant_colour_image),
        ("Output matches a hand-computed 4x4 example", verify_hand_computed_4x4_example),
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
        print("All T5-003 acceptance criteria successfully verified!")
    else:
        print("One or more T5-003 checks failed.")
    print()

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
