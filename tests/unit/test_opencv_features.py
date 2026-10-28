"""Comprehensive unit tests for OpenCV feature formulas and edge cases (T8-002).

Verifies acceptance criteria:
1. Each feature has at least one hand-computed expected value:
   - exg_mean (Excess Green index)
   - exr_mean (Excess Red index)
   - yellow_pixel_ratio (Chlorotic HSV ratio)
   - brown_pixel_ratio (Necrotic HSV ratio)
   - mean_hue and mean_saturation
   - contour_compactness (Polsby-Popper / Isoperimetric quotient)
   - relative_leaf_area (Foreground pixels / Total image pixels)
   - bounding_aspect_ratio (Bounding box width / height)
   - solidity (Contour area / Convex hull area)
   - droop_angle (Verified removed per ADR 0007)
2. Channel-order bug is caught by a test (RGB vs BGR inversion).
3. Degenerate inputs raise typed errors:
   - EmptyMaskError on zero-foreground masks
   - DimensionMismatchError on image/mask size mismatch
   - ZeroPerimeterError on degenerate leaf perimeter
   - DegenerateContourError on zero convex hull area
   - NonFiniteFeatureError on NaN or +/-Inf
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from ml.features.common import (
    DegenerateContourError,
    DimensionMismatchError,
    EmptyMaskError,
    NonFiniteFeatureError,
    ZeroPerimeterError,
    assert_finite_features,
    load_image,
)
from ml.features.lettuce import (
    DEFAULT_LETTUCE_CONFIG,
    LETTUCE_FEATURE_NAMES,
    compute_contour_compactness,
    compute_mean_hue_and_saturation,
    extract_lettuce_features,
)
from ml.features.spinach import (
    DEFAULT_SPINACH_CONFIG,
    SPINACH_FEATURE_NAMES,
    compute_excess_green,
    compute_excess_red,
    compute_hsv_ratio,
    extract_spinach_features,
)
from ml.features.tomato import (
    DEFAULT_TOMATO_CONFIG,
    TOMATO_FEATURE_NAMES,
    compute_shape_descriptors,
    extract_tomato_features,
)

# ==============================================================================
# Criterion 1: Hand-Computed Expected Values
# ==============================================================================


def test_hand_computed_excess_green() -> None:
    """Hand-computed test for Excess Green: ExG = 2g - r - b.

    Given RGB pixel: (51, 153, 51)
    Sum = 51 + 153 + 51 = 255.
    Normalized chromaticities:
      r = 51 / 255 = 0.20
      g = 153 / 255 = 0.60
      b = 51 / 255 = 0.20
    ExG = 2*(0.60) - 0.20 - 0.20 = 1.20 - 0.40 = 0.8000000000.
    """
    pixel = np.array([[51, 153, 51]], dtype=np.uint8)
    exg = compute_excess_green(pixel)
    assert len(exg) == 1
    assert pytest.approx(exg[0], abs=1e-6) == 0.80


def test_hand_computed_excess_red() -> None:
    """Hand-computed test for Excess Red: ExR = 1.4r - g.

    Given RGB pixel: (204, 25.5, 25.5) -> scaled integer (204, 26, 25)
    Using exact float array: [0.80, 0.10, 0.10]
    Sum = 1.00
    r = 0.80, g = 0.10, b = 0.10
    ExR = 1.4 * 0.80 - 0.10 = 1.12 - 0.10 = 1.0200000000.
    """
    pixel_float = np.array([[0.80, 0.10, 0.10]], dtype=np.float64)
    exr = compute_excess_red(pixel_float)
    assert len(exr) == 1
    assert pytest.approx(exr[0], abs=1e-6) == 1.02


def test_hand_computed_hsv_yellow_ratio() -> None:
    """Hand-computed test for yellow_pixel_ratio.

    10 foreground pixels:
    - 6 pure chlorotic yellow pixels (RGB: [255, 255, 0] -> OpenCV HSV: [30, 255, 255])
      Hue 30 falls strictly inside default yellow hue range [18, 38].
    - 4 pure blue pixels (RGB: [0, 0, 255] -> OpenCV HSV: [120, 255, 255])
      Hue 120 is far outside yellow range.
    Expected yellow ratio = 6 / 10 = 0.600000.
    """
    yellow_pixels = np.tile([255, 255, 0], (6, 1))
    blue_pixels = np.tile([0, 0, 255], (4, 1))
    pixels = np.vstack([yellow_pixels, blue_pixels]).astype(np.uint8)

    cfg = DEFAULT_LETTUCE_CONFIG
    ratio = compute_hsv_ratio(
        pixels,
        hue_range=(cfg.yellow_hue_min, cfg.yellow_hue_max),
        sat_range=(cfg.yellow_sat_min, cfg.yellow_sat_max),
        val_range=(cfg.yellow_val_min, cfg.yellow_val_max),
    )
    assert pytest.approx(ratio, abs=1e-6) == 0.60


def test_hand_computed_hsv_brown_ratio() -> None:
    """Hand-computed test for brown_pixel_ratio (necrosis).

    10 foreground pixels:
    - 3 brown pixels (RGB: [139, 69, 19] -> OpenCV HSV ~ [12, 220, 139])
      Falls strictly inside default brown range: H in [5, 22], S in [30, 255], V in [20, 160].
    - 7 healthy green pixels (RGB: [34, 139, 34] -> OpenCV HSV ~ [60, 193, 139])
      Hue ~60 is outside brown range.
    Expected brown ratio = 3 / 10 = 0.300000.
    """
    brown_pixels = np.tile([139, 69, 19], (3, 1))
    green_pixels = np.tile([34, 139, 34], (7, 1))
    pixels = np.vstack([brown_pixels, green_pixels]).astype(np.uint8)

    cfg = DEFAULT_SPINACH_CONFIG
    ratio = compute_hsv_ratio(
        pixels,
        hue_range=(cfg.brown_hue_min, cfg.brown_hue_max),
        sat_range=(cfg.brown_sat_min, cfg.brown_sat_max),
        val_range=(cfg.brown_val_min, cfg.brown_val_max),
    )
    assert pytest.approx(ratio, abs=1e-6) == 0.30


def test_hand_computed_mean_hue_and_saturation() -> None:
    """Hand-computed test for mean_hue and mean_saturation.

    Two distinct color blocks:
    - Pixel 1: Pure Red (RGB: [255, 0, 0]) -> OpenCV HSV: H=0, S=255, V=255
    - Pixel 2: Pure Cyan (RGB: [0, 255, 255]) -> OpenCV HSV: H=90, S=255, V=255
    Expected Mean Hue = (0 + 90) / 2 = 45.0.
    Expected Mean Saturation = (255 + 255) / 2 = 255.0.
    """
    pixels = np.array([[255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    mean_h, mean_s = compute_mean_hue_and_saturation(pixels)
    assert pytest.approx(mean_h, abs=1e-6) == 45.0
    assert pytest.approx(mean_s, abs=1e-6) == 255.0


def test_hand_computed_contour_compactness() -> None:
    """Hand-computed test for contour compactness: 4 * pi * Area / Perimeter^2.

    For an axis-aligned square of width 40, height 40:
    Continuous Area = 40 * 40 = 1600.
    Continuous Perimeter = 4 * 40 = 160.
    Theoretical compactness = 4 * pi * 1600 / (160^2) = (6400 * pi) / 25600 = pi / 4 ~ 0.785398.

    With discrete pixel grid contour:
    OpenCV contour on 40x40 pixel block:
      cv2.contourArea(c) = (40-1)*(40-1) = 39*39 = 1521 (Green's theorem on pixel centers)
      cv2.arcLength(c, closed=True) = 4 * 39 = 156.
      compactness = 4 * pi * 1521 / (156^2) = 4 * pi * 1521 / 24336 = pi / 4 = 0.785398!
    The discrete ratio matches pi / 4 exactly!
    """
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True  # 40x40 square

    compactness = compute_contour_compactness(mask)
    theoretical_square_compactness = math.pi / 4.0
    assert pytest.approx(compactness, rel=1e-3) == theoretical_square_compactness


def test_hand_computed_relative_leaf_area() -> None:
    """Hand-computed test for relative_leaf_area.

    Image dimensions: 50 x 100 -> Total pixels = 5,000.
    Mask: Rectangle of 25 x 50 = 1,250 foreground pixels.
    Expected relative_leaf_area = 1,250 / 5,000 = 0.250000000.
    """
    h, w = 50, 100
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[10:35, 20:70] = True  # 25 rows x 50 cols = 1250 pixels

    vec = extract_spinach_features(img, mask)
    idx = vec.feature_names.index("relative_leaf_area")
    assert pytest.approx(vec.values[idx], abs=1e-6) == 0.25


def test_hand_computed_bounding_aspect_ratio_and_solidity() -> None:
    """Hand-computed test for bounding_aspect_ratio and solidity.

    Rectangle mask of width 60, height 30:
    - Bounding Box: w = 60, h = 30 -> Aspect Ratio = 60 / 30 = 2.000.
    - Solidity: For a solid convex rectangle, contour area == convex hull area -> Solidity = 1.000.
    """
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 10:70] = True  # 30 rows (height), 60 cols (width)

    ar, sol = compute_shape_descriptors(mask)
    # Accounting for discrete pixel coordinates (60-1)/(30-1) = 59/29 ~ 2.03 or bounding box w/h
    # cv2.boundingRect covers the full enclosing pixel span: [10..69] -> w=60, [20..49] -> h=30
    assert pytest.approx(ar, abs=1e-2) == 2.00
    assert pytest.approx(sol, abs=1e-2) == 1.00


def test_hand_computed_non_convex_solidity() -> None:
    """Hand-computed test for non-convex shape solidity.

    An L-shaped polygon covering 3 out of 4 quadrants of a 40x40 square:
    Quadrants: Top-Left (20x20), Bottom-Left (20x20), Bottom-Right (20x20).
    Missing Top-Right quadrant.
    Contour Area = 3 * (20 * 20) = 1200.
    Convex Hull connects (20, 40) and (40, 20) with a straight line, adding
    a triangle of area (1/2) * 20 * 20 = 200 over the missing quadrant.
    Total Hull Area = 1200 + 200 = 1400.
    Theoretical Solidity = 1200 / 1400 = 6/7 ~ 0.857143.
    """
    mask = np.zeros((80, 80), dtype=bool)
    # Fill 3 quadrants of 40x40 block from (20, 20) to (60, 60):
    mask[20:40, 20:40] = True  # Top-Left
    mask[40:60, 20:40] = True  # Bottom-Left
    mask[40:60, 40:60] = True  # Bottom-Right
    # Top-Right [20:40, 40:60] is left False (background)

    _, sol = compute_shape_descriptors(mask)
    theoretical_solidity = 6.0 / 7.0  # 0.857143
    assert pytest.approx(sol, rel=0.02) == theoretical_solidity


def test_droop_angle_decision_status() -> None:
    """Verify droop_angle was removed per ADR 0007 and is absent from canonical features."""
    assert "droop_angle" not in TOMATO_FEATURE_NAMES
    assert "droop_angle" not in LETTUCE_FEATURE_NAMES
    assert "droop_angle" not in SPINACH_FEATURE_NAMES
    assert (
        DEFAULT_TOMATO_CONFIG.to_dict()["droop_angle_decision"]
        == "REMOVED per ADR 0007"
    )


# ==============================================================================
# Criterion 2: Channel-Order Bug Caught by Test
# ==============================================================================


def test_channel_order_bug_caught() -> None:
    """Verify that treating BGR as RGB severely corrupts color indices (ExG and ExR).

    Consider an agronomic leaf sample with a necrotic reddish/brown spot:
    True RGB values: R=200, G=50, B=30.
    In RGB:
      Sum = 200 + 50 + 30 = 280.
      r = 200 / 280 = 0.714286
      g = 50 / 280 = 0.178571
      b = 30 / 280 = 0.107143
      ExG_RGB = 2*(0.178571) - 0.714286 - 0.107143 = -0.464286 (strongly negative, indicating non-green)
      ExR_RGB = 1.4*(0.714286) - 0.178571 = 1.000000 - 0.178571 = +0.821429 (strongly positive, indicating red)

    If the image is erroneously treated as BGR (where Blue is loaded as Red, and Red as Blue):
      Inverted "RGB" values: R=30, G=50, B=200.
      r_corrupted = 30 / 280 = 0.107143
      g_corrupted = 50 / 280 = 0.178571
      b_corrupted = 200 / 280 = 0.714286
      ExR_corrupted = 1.4*(0.107143) - 0.178571 = 0.150000 - 0.178571 = -0.028571!

    The corruption alters ExR from +0.821 (severe necrotic symptom) to -0.029 (near zero).
    This test verifies that:
    1. The BGR inversion produces an enormous divergence (> 0.80 difference in ExR).
    2. The standard load_image utility converts BGR to RGB so downstream extractors receive valid RGB.
    """
    true_rgb_pixel = np.array([[200, 50, 30]], dtype=np.uint8)
    corrupted_bgr_pixel = np.array([[30, 50, 200]], dtype=np.uint8)

    exr_correct = compute_excess_red(true_rgb_pixel)[0]
    exr_corrupted = compute_excess_red(corrupted_bgr_pixel)[0]

    # Hand-computed difference
    assert pytest.approx(exr_correct, abs=1e-3) == 0.821
    assert pytest.approx(exr_corrupted, abs=1e-3) == -0.029

    # The error is caught: discrepancy exceeds 0.80!
    discrepancy = abs(exr_correct - exr_corrupted)
    assert discrepancy > 0.80, (
        f"Expected channel inversion discrepancy > 0.80, got {discrepancy}"
    )

    # Verify load_image prevents this bug by explicitly decoding into RGB
    # Encode test image to JPEG bytes (OpenCV imencode uses BGR)
    img_bgr = np.full((10, 10, 3), [30, 50, 200], dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img_bgr)
    loaded_rgb = load_image(buf.tobytes())

    # loaded_rgb should have R ~ 200 and B ~ 30 (swapped back to RGB)
    mean_loaded_r = np.mean(loaded_rgb[:, :, 0])
    mean_loaded_b = np.mean(loaded_rgb[:, :, 2])
    assert mean_loaded_r > 180, f"Expected R channel > 180, got {mean_loaded_r}"
    assert mean_loaded_b < 50, f"Expected B channel < 50, got {mean_loaded_b}"


# ==============================================================================
# Criterion 3: Degenerate Inputs Raise Typed Errors
# ==============================================================================


def test_empty_mask_raises_typed_error() -> None:
    """Empty mask must raise EmptyMaskError (never silent zeros)."""
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    empty_mask = np.zeros((40, 40), dtype=bool)

    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_lettuce_features(img, empty_mask)

    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_spinach_features(img, empty_mask)

    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_tomato_features(img, empty_mask)


def test_dimension_mismatch_raises_typed_error() -> None:
    """Dimension mismatch must raise DimensionMismatchError."""
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    mismatched_mask = np.zeros((60, 50), dtype=bool)

    with pytest.raises(DimensionMismatchError):
        extract_lettuce_features(img, mismatched_mask)

    with pytest.raises(DimensionMismatchError):
        extract_spinach_features(img, mismatched_mask)

    with pytest.raises(DimensionMismatchError):
        extract_tomato_features(img, mismatched_mask)


def test_zero_perimeter_raises_typed_error() -> None:
    """Degenerate contour with zero perimeter must raise ZeroPerimeterError."""
    single_pixel_mask = np.zeros((40, 40), dtype=bool)
    single_pixel_mask[20, 20] = True

    with pytest.raises(ZeroPerimeterError, match="zero or non-positive"):
        compute_contour_compactness(single_pixel_mask)

    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    with pytest.raises(ZeroPerimeterError):
        extract_lettuce_features(img, single_pixel_mask)


def test_zero_hull_area_raises_degenerate_contour_error() -> None:
    """Degenerate contour with zero hull area must raise DegenerateContourError."""
    single_pixel_mask = np.zeros((30, 30), dtype=bool)
    single_pixel_mask[15, 15] = True

    with pytest.raises(
        DegenerateContourError, match="convex hull area is zero or non-positive"
    ):
        compute_shape_descriptors(single_pixel_mask)

    img = np.full((30, 30, 3), 120, dtype=np.uint8)
    with pytest.raises(DegenerateContourError):
        extract_tomato_features(img, single_pixel_mask)


def test_non_finite_feature_raises_typed_error() -> None:
    """NaN or Inf feature values must raise NonFiniteFeatureError (never imputed with 0)."""
    # 1. Direct assert_finite_features mapping validation
    features_with_nan = {"exg_mean": 0.5, "exr_mean": float("nan")}
    with pytest.raises(NonFiniteFeatureError, match="Non-finite feature value"):
        assert_finite_features(features_with_nan)

    features_with_inf = {"glcm_contrast": float("inf"), "solidity": 0.9}
    with pytest.raises(NonFiniteFeatureError, match="Non-finite feature value"):
        assert_finite_features(features_with_inf)

    # 2. Sequence validation
    with pytest.raises(NonFiniteFeatureError):
        assert_finite_features([0.1, float("-inf"), 0.3], ["f1", "f2", "f3"])
