#!/usr/bin/env python3
"""Verification script for T8-002: Unit tests: OpenCV feature formulas and edge cases.

Acceptance Criteria:
1. Each feature has at least one hand-computed expected value:
   - exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, mean_hue, mean_saturation,
     contour_compactness, relative_leaf_area, bounding_aspect_ratio, solidity, droop_angle status.
2. Channel-order bug is caught by a test (RGB vs BGR inversion).
3. Degenerate inputs raise typed errors (EmptyMaskError, DimensionMismatchError,
   ZeroPerimeterError, DegenerateContourError, NonFiniteFeatureError).

Usage:
    python scripts/verify_opencv_features.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.common import (  # noqa: E402
    DegenerateContourError,
    DimensionMismatchError,
    EmptyMaskError,
    NonFiniteFeatureError,
    ZeroPerimeterError,
    assert_finite_features,
)
from ml.features.lettuce import (  # noqa: E402
    DEFAULT_LETTUCE_CONFIG,
    compute_contour_compactness,
    compute_mean_hue_and_saturation,
)
from ml.features.spinach import (  # noqa: E402
    DEFAULT_SPINACH_CONFIG,
    compute_excess_green,
    compute_excess_red,
    compute_hsv_ratio,
    extract_spinach_features,
)
from ml.features.tomato import (  # noqa: E402
    DEFAULT_TOMATO_CONFIG,
    TOMATO_FEATURE_NAMES,
    compute_shape_descriptors,
)


def verify_hand_computed_features() -> tuple[bool, str]:
    """Criterion 1: Each feature has at least one hand-computed expected value."""
    # 1. ExG hand-computed: (51, 153, 51) -> r=0.2, g=0.6, b=0.2 -> 2g - r - b = 0.8
    pixel = np.array([[51, 153, 51]], dtype=np.uint8)
    exg = compute_excess_green(pixel)[0]
    if abs(exg - 0.80) > 1e-5:
        return False, f"ExG mismatch: expected 0.80, got {exg}"

    # 2. ExR hand-computed: [0.8, 0.1, 0.1] -> 1.4r - g = 1.4(0.8) - 0.1 = 1.02
    pixel_flt = np.array([[0.8, 0.1, 0.1]], dtype=np.float64)
    exr = compute_excess_red(pixel_flt)[0]
    if abs(exr - 1.02) > 1e-5:
        return False, f"ExR mismatch: expected 1.02, got {exr}"

    # 3. Yellow ratio: 6 yellow + 4 blue -> 6/10 = 0.60
    y_pix = np.tile([255, 255, 0], (6, 1))
    b_pix = np.tile([0, 0, 255], (4, 1))
    yellow_mix = np.vstack([y_pix, b_pix]).astype(np.uint8)
    cfg_l = DEFAULT_LETTUCE_CONFIG
    y_ratio = compute_hsv_ratio(
        yellow_mix,
        hue_range=(cfg_l.yellow_hue_min, cfg_l.yellow_hue_max),
        sat_range=(cfg_l.yellow_sat_min, cfg_l.yellow_sat_max),
        val_range=(cfg_l.yellow_val_min, cfg_l.yellow_val_max),
    )
    if abs(y_ratio - 0.60) > 1e-5:
        return False, f"Yellow ratio mismatch: expected 0.60, got {y_ratio}"

    # 4. Brown ratio: 3 brown + 7 green -> 3/10 = 0.30
    br_pix = np.tile([139, 69, 19], (3, 1))
    gr_pix = np.tile([34, 139, 34], (7, 1))
    brown_mix = np.vstack([br_pix, gr_pix]).astype(np.uint8)
    cfg_s = DEFAULT_SPINACH_CONFIG
    br_ratio = compute_hsv_ratio(
        brown_mix,
        hue_range=(cfg_s.brown_hue_min, cfg_s.brown_hue_max),
        sat_range=(cfg_s.brown_sat_min, cfg_s.brown_sat_max),
        val_range=(cfg_s.brown_val_min, cfg_s.brown_val_max),
    )
    if abs(br_ratio - 0.30) > 1e-5:
        return False, f"Brown ratio mismatch: expected 0.30, got {br_ratio}"

    # 5. Mean Hue & Saturation: [255, 0, 0] (H=0, S=255) and [0, 255, 255] (H=90, S=255) -> Mean H=45, Mean S=255
    h_s_pixels = np.array([[255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    m_h, m_s = compute_mean_hue_and_saturation(h_s_pixels)
    if abs(m_h - 45.0) > 1e-5 or abs(m_s - 255.0) > 1e-5:
        return False, f"Mean HSV mismatch: expected (45.0, 255.0), got ({m_h}, {m_s})"

    # 6. Compactness: Square mask -> pi / 4 ~ 0.785398
    sq_mask = np.zeros((100, 100), dtype=bool)
    sq_mask[30:70, 30:70] = True
    c_val = compute_contour_compactness(sq_mask)
    if abs(c_val - (math.pi / 4.0)) > 1e-3:
        return (
            False,
            f"Compactness mismatch: expected {math.pi / 4:.5f}, got {c_val:.5f}",
        )

    # 7. Relative Leaf Area: 1250 fg / 5000 total = 0.25
    img = np.full((50, 100, 3), 100, dtype=np.uint8)
    rla_mask = np.zeros((50, 100), dtype=bool)
    rla_mask[10:35, 20:70] = True
    vec = extract_spinach_features(img, rla_mask)
    rla_val = vec.to_dict()["relative_leaf_area"]
    if abs(rla_val - 0.25) > 1e-5:
        return False, f"Relative leaf area mismatch: expected 0.25, got {rla_val}"

    # 8. Aspect Ratio & Solidity: 60x30 rectangle -> AR=2.0, Solidity=1.0
    rect_mask = np.zeros((100, 100), dtype=bool)
    rect_mask[20:50, 10:70] = True
    ar, sol = compute_shape_descriptors(rect_mask)
    if abs(ar - 2.00) > 0.05 or abs(sol - 1.00) > 0.05:
        return (
            False,
            f"AR/Solidity mismatch: expected (2.0, 1.0), got ({ar:.3f}, {sol:.3f})",
        )

    # 9. Droop angle verified removed per ADR 0007
    if "droop_angle" in TOMATO_FEATURE_NAMES:
        return False, "droop_angle was not removed from TOMATO_FEATURE_NAMES"
    if (
        DEFAULT_TOMATO_CONFIG.to_dict()["droop_angle_decision"]
        != "REMOVED per ADR 0007"
    ):
        return (
            False,
            "droop_angle removal decision not recorded in DEFAULT_TOMATO_CONFIG",
        )

    return (
        True,
        "Verified all 10 feature values and definitions match hand-computed ground truths.",
    )


def verify_channel_order_test() -> tuple[bool, str]:
    """Criterion 2: Channel-order bug is caught by a test."""
    # Red necrotic pixel R=200, G=50, B=30
    true_rgb = np.array([[200, 50, 30]], dtype=np.uint8)
    corrupted_bgr = np.array([[30, 50, 200]], dtype=np.uint8)

    exr_true = compute_excess_red(true_rgb)[0]
    exr_corrupted = compute_excess_red(corrupted_bgr)[0]

    discrepancy = abs(exr_true - exr_corrupted)
    if discrepancy < 0.50:
        return (
            False,
            f"Expected channel-order discrepancy > 0.50, but got only {discrepancy}",
        )

    return True, (
        f"Verified channel-order bug is caught: ExR changes from {exr_true:.3f} (RGB) "
        f"to {exr_corrupted:.3f} (BGR) with discrepancy {discrepancy:.3f}"
    )


def verify_degenerate_inputs_raise_errors() -> tuple[bool, str]:
    """Criterion 3: Degenerate inputs raise typed errors."""
    single_pixel = np.zeros((30, 30), dtype=bool)
    single_pixel[15, 15] = True

    # 1. Zero perimeter
    try:
        compute_contour_compactness(single_pixel)
        return False, "Expected ZeroPerimeterError on single-pixel mask."
    except ZeroPerimeterError:
        pass

    # 2. Degenerate contour / zero hull area
    try:
        compute_shape_descriptors(single_pixel)
        return False, "Expected DegenerateContourError on single-pixel mask."
    except DegenerateContourError:
        pass

    # 3. Empty mask
    empty_m = np.zeros((30, 30), dtype=bool)
    try:
        compute_contour_compactness(empty_m)
        return False, "Expected EmptyMaskError on empty mask."
    except EmptyMaskError:
        pass

    # 4. Dimension mismatch
    try:
        extract_spinach_features(
            np.zeros((30, 30, 3), dtype=np.uint8), np.zeros((40, 30), dtype=bool)
        )
        return False, "Expected DimensionMismatchError on size mismatch."
    except DimensionMismatchError:
        pass

    # 5. Non-finite values
    try:
        assert_finite_features({"f1": float("nan")})
        return False, "Expected NonFiniteFeatureError on NaN value."
    except NonFiniteFeatureError:
        pass

    return (
        True,
        "Verified typed errors: ZeroPerimeterError, DegenerateContourError, EmptyMaskError, DimensionMismatchError, NonFiniteFeatureError.",
    )


def run_verification() -> bool:
    checks = [
        (
            "Criterion 1: Hand-computed expected values for all features",
            verify_hand_computed_features,
        ),
        ("Criterion 2: Channel-order bug caught by test", verify_channel_order_test),
        (
            "Criterion 3: Degenerate inputs raise typed errors",
            verify_degenerate_inputs_raise_errors,
        ),
    ]

    all_passed = True
    print("=" * 70)
    print("Verification for T8-002: OpenCV feature formulas and edge cases")
    print("=" * 70)

    for name, check_fn in checks:
        passed, msg = check_fn()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
