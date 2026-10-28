"""Unit tests for spinach leaf image feature extraction (T5-005)."""

from __future__ import annotations

import json
import numpy as np
import pytest

from ml.features.common import (
    DimensionMismatchError,
    EmptyMaskError,
    FeatureVector,
    NonFiniteFeatureError,
)
from ml.features.spinach import (
    DEFAULT_SPINACH_CONFIG,
    SPINACH_FEATURE_NAMES,
    SpinachFeatureConfig,
    compute_excess_green,
    compute_excess_red,
    compute_hsv_ratio,
    extract_spinach_features,
)
from scripts.verify_spinach_features import run_verification


def test_spinach_feature_names_order():
    """Verify acceptance criterion 1: Order matches Overview §16.2."""
    expected = (
        "exg_mean",
        "exr_mean",
        "yellow_pixel_ratio",
        "brown_pixel_ratio",
        "relative_leaf_area",
        "glcm_contrast",
    )
    assert SPINACH_FEATURE_NAMES == expected

    # Test that extracted vector matches order
    h, w = 40, 40
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [40, 180, 60]  # Healthy green
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 10:30] = True

    vec = extract_spinach_features(img, mask)
    assert isinstance(vec, FeatureVector)
    assert tuple(vec.feature_names) == expected
    assert len(vec.values) == 6


def test_spinach_relative_leaf_area_denominator_metadata():
    """Verify acceptance criterion 2: Denominator is documented and stored in metadata."""
    h, w = 100, 50
    total_pixels = h * w
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 10:25] = True  # 20 * 15 = 300 foreground pixels
    fg_pixels = 20 * 15

    vec = extract_spinach_features(img, mask)
    config = vec.extraction_config

    assert "relative_leaf_area_denominator" in config
    assert "total_image_pixels" in config["relative_leaf_area_denominator"]
    assert config["total_image_pixels"] == total_pixels
    assert config["foreground_pixels"] == fg_pixels

    idx = vec.feature_names.index("relative_leaf_area")
    expected_ratio = fg_pixels / float(total_pixels)
    assert pytest.approx(vec.values[idx], abs=1e-7) == expected_ratio


def test_spinach_empty_mask_raises():
    """Verify acceptance criterion 3: Empty mask raises EmptyMaskError."""
    h, w = 32, 32
    img = np.full((h, w, 3), 128, dtype=np.uint8)

    # Boolean empty mask
    mask_bool = np.zeros((h, w), dtype=bool)
    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_spinach_features(img, mask_bool)

    # Integer empty mask
    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_spinach_features(img, mask_uint8)


def test_spinach_mismatched_dimensions_raises():
    """Verify spatial dimension mismatch raises DimensionMismatchError."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((60, 50), dtype=bool)
    with pytest.raises(DimensionMismatchError):
        extract_spinach_features(img, mask)

    # Invalid image shape
    with pytest.raises(DimensionMismatchError):
        extract_spinach_features(np.zeros((50, 50), dtype=np.uint8), np.zeros((50, 50), dtype=bool))


def test_spinach_exg_and_exr_indices():
    """Verify Excess Green and Excess Red calculation logic."""
    # Pure Green pixel: [0, 255, 0] -> r=0, g=1, b=0
    # ExG = 2g - r - b = 2(1) - 0 - 0 = 2.0
    # ExR = 1.4r - g = 1.4(0) - 1 = -1.0
    green_pixels = np.array([[0, 255, 0]], dtype=np.uint8)
    exg_green = compute_excess_green(green_pixels)
    exr_green = compute_excess_red(green_pixels)
    assert pytest.approx(exg_green[0], abs=1e-5) == 2.0
    assert pytest.approx(exr_green[0], abs=1e-5) == -1.0

    # Pure Red pixel: [255, 0, 0] -> r=1, g=0, b=0
    # ExG = 2g - r - b = 2(0) - 1 - 0 = -1.0
    # ExR = 1.4r - g = 1.4(1) - 0 = 1.4
    red_pixels = np.array([[255, 0, 0]], dtype=np.uint8)
    exg_red = compute_excess_green(red_pixels)
    exr_red = compute_excess_red(red_pixels)
    assert pytest.approx(exg_red[0], abs=1e-5) == -1.0
    assert pytest.approx(exr_red[0], abs=1e-5) == 1.4


def test_spinach_hsv_yellow_and_brown_ratios():
    """Verify chlorosis (yellow) and necrosis (brown) ratio detection."""
    cfg = DEFAULT_SPINACH_CONFIG
    # Pure yellow in RGB is [255, 255, 0] (Hue ~30 in OpenCV 0..180)
    yellow_pixels = np.array([[255, 255, 0]], dtype=np.uint8)
    ratio_yellow = compute_hsv_ratio(
        yellow_pixels,
        hue_range=(cfg.yellow_hue_min, cfg.yellow_hue_max),
        sat_range=(cfg.yellow_sat_min, cfg.yellow_sat_max),
        val_range=(cfg.yellow_val_min, cfg.yellow_val_max),
    )
    assert ratio_yellow == 1.0

    # Brown RGB example: [139, 69, 19] (Hue ~12, Sat ~220, Val ~139)
    # Using brown configuration
    brown_pixels = np.array([[120, 60, 25]], dtype=np.uint8)
    ratio_brown = compute_hsv_ratio(
        brown_pixels,
        hue_range=(cfg.brown_hue_min, cfg.brown_hue_max),
        sat_range=(cfg.brown_sat_min, cfg.brown_sat_max),
        val_range=(cfg.brown_val_min, cfg.brown_val_max),
    )
    assert ratio_brown == 1.0


def test_spinach_feature_vector_serialization():
    """Verify serialization to dict, json, and numpy."""
    h, w = 30, 30
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[5:25, 5:25] = [35, 175, 45]
    mask = np.zeros((h, w), dtype=bool)
    mask[5:25, 5:25] = True

    vec = extract_spinach_features(img, mask)

    # 1. to_dict()
    d = vec.to_dict()
    assert len(d) == 6
    assert list(d.keys()) == list(SPINACH_FEATURE_NAMES)
    for k, v in d.items():
        assert isinstance(v, float)
        assert np.isfinite(v)

    # 2. to_numpy()
    arr = vec.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6,)
    assert arr.dtype == np.float64

    # 3. to_json()
    j_str = vec.to_json()
    parsed = json.loads(j_str)
    assert parsed["schema_version"] == DEFAULT_SPINACH_CONFIG.schema_version
    assert parsed["crop"] == "spinach"
    assert parsed["feature_names"] == list(SPINACH_FEATURE_NAMES)
    assert len(parsed["values"]) == 6


def test_spinach_verification_script():
    """Verify that verify_spinach_features.py runs and passes completely."""
    assert run_verification() is True
