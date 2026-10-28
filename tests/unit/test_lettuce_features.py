"""Unit tests for lettuce leaf image feature extraction (T5-004)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.common import (
    DimensionMismatchError,
    EmptyMaskError,
    FeatureVector,
    ZeroPerimeterError,
)
from ml.features.lettuce import (
    DEFAULT_LETTUCE_CONFIG,
    LETTUCE_FEATURE_NAMES,
    LettuceFeatureConfig,
    compute_contour_compactness,
    extract_lettuce_features,
)
from scripts.verify_lettuce_features import run_verification


def test_lettuce_feature_names_order() -> None:
    """Verify acceptance criterion 1: Order matches Overview §16.1."""
    expected = (
        "exg_mean",
        "exr_mean",
        "yellow_pixel_ratio",
        "brown_pixel_ratio",
        "mean_hue",
        "mean_saturation",
        "contour_compactness",
    )
    assert LETTUCE_FEATURE_NAMES == expected

    # Test that extracted vector matches order
    h, w = 40, 40
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [40, 180, 60]  # Healthy green
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 10:30] = True

    vec = extract_lettuce_features(img, mask)
    assert isinstance(vec, FeatureVector)
    assert tuple(vec.feature_names) == expected
    assert len(vec.values) == 7


def test_lettuce_zero_perimeter_raises() -> None:
    """Verify acceptance criterion 2: Zero perimeter produces ZeroPerimeterError."""
    # 1. Single isolated pixel mask
    mask_single = np.zeros((30, 30), dtype=bool)
    mask_single[15, 15] = True

    with pytest.raises(ZeroPerimeterError, match="zero or non-positive"):
        compute_contour_compactness(mask_single)

    img = np.full((30, 30, 3), 120, dtype=np.uint8)
    with pytest.raises(ZeroPerimeterError):
        extract_lettuce_features(img, mask_single)


def test_lettuce_empty_mask_raises() -> None:
    """Verify empty mask raises EmptyMaskError."""
    h, w = 32, 32
    img = np.full((h, w, 3), 128, dtype=np.uint8)

    mask_bool = np.zeros((h, w), dtype=bool)
    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_lettuce_features(img, mask_bool)

    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_lettuce_features(img, mask_uint8)


def test_lettuce_hsv_thresholds_configurable() -> None:
    """Verify acceptance criterion 3: HSV thresholds stored in config and not hardcoded."""
    h, w = 50, 50
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [220, 220, 20]  # Chlorotic yellow
    mask = np.ones((h, w), dtype=bool)

    # 1. Default config detects yellow pixels
    vec_def = extract_lettuce_features(img, mask, config=DEFAULT_LETTUCE_CONFIG)
    assert vec_def.to_dict()["yellow_pixel_ratio"] > 0.5
    assert "yellow_hue_min" in vec_def.extraction_config
    assert "brown_hue_min" in vec_def.extraction_config
    assert vec_def.extraction_config["crop"] == "lettuce"

    # 2. Custom config shifts thresholds
    custom_cfg = LettuceFeatureConfig(yellow_hue_min=120.0, yellow_hue_max=130.0)
    vec_custom = extract_lettuce_features(img, mask, config=custom_cfg)
    assert vec_custom.to_dict()["yellow_pixel_ratio"] == 0.0


def test_lettuce_mismatched_dimensions_raises() -> None:
    """Verify spatial dimension mismatch raises DimensionMismatchError."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((60, 50), dtype=bool)
    with pytest.raises(DimensionMismatchError):
        extract_lettuce_features(img, mask)


def test_lettuce_automatic_mask_generation() -> None:
    """Verify automatic Otsu mask generation when mask is omitted for leaf crop."""
    h, w = 60, 60
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright leaf square on dark background
    img[15:45, 15:45] = [40, 200, 50]

    vec = extract_lettuce_features(img, mask=None)
    assert isinstance(vec, FeatureVector)
    assert len(vec.values) == 7
    assert vec.to_dict()["exg_mean"] > 0.0


def test_lettuce_run_verification_script() -> None:
    """Verify verification script executes and passes completely."""
    assert run_verification() is True
