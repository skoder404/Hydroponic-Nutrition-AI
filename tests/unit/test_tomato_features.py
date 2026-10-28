"""Unit tests for tomato leaf image feature extraction (T5-007)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.common import (
    DegenerateContourError,
    DimensionMismatchError,
    EmptyMaskError,
    FeatureVector,
)
from ml.features.tomato import (
    DEFAULT_TOMATO_CONFIG,
    TOMATO_FEATURE_NAMES,
    compute_shape_descriptors,
    extract_tomato_features,
)
from scripts.verify_tomato_features import run_verification


def test_tomato_feature_names_order() -> None:
    """Verify acceptance criterion 1: Feature list equals ADR 0007 decision."""
    expected = (
        "glcm_contrast",
        "glcm_entropy",
        "bounding_aspect_ratio",
        "solidity",
    )
    assert TOMATO_FEATURE_NAMES == expected

    h, w = 50, 50
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [40, 180, 60]  # Healthy green
    mask = np.zeros((h, w), dtype=bool)
    mask[15:35, 15:35] = True

    vec = extract_tomato_features(img, mask)
    assert isinstance(vec, FeatureVector)
    assert tuple(vec.feature_names) == expected
    assert len(vec.values) == 4


def test_tomato_metadata_definitions() -> None:
    """Verify acceptance criterion 2: Exact definitions recorded in metadata."""
    cfg = DEFAULT_TOMATO_CONFIG
    config_dict = cfg.to_dict()

    assert "definitions" in config_dict
    defs = config_dict["definitions"]
    assert "bounding_aspect_ratio" in defs
    assert "solidity" in defs
    assert "droop_angle_decision" in config_dict
    assert "REMOVED" in config_dict["droop_angle_decision"]


def test_tomato_degenerate_contour_raises() -> None:
    """Verify acceptance criterion 3: Degenerate contour (zero hull area) raises an error."""
    # 1. Single isolated pixel
    mask_single = np.zeros((30, 30), dtype=bool)
    mask_single[15, 15] = True

    with pytest.raises(
        DegenerateContourError, match="convex hull area is zero or non-positive"
    ):
        compute_shape_descriptors(mask_single)

    img = np.full((30, 30, 3), 120, dtype=np.uint8)
    with pytest.raises(DegenerateContourError):
        extract_tomato_features(img, mask_single)


def test_tomato_empty_mask_raises() -> None:
    """Verify empty mask raises EmptyMaskError."""
    h, w = 32, 32
    img = np.full((h, w, 3), 128, dtype=np.uint8)

    mask_bool = np.zeros((h, w), dtype=bool)
    with pytest.raises(EmptyMaskError, match="zero foreground pixels"):
        extract_tomato_features(img, mask_bool)


def test_tomato_mismatched_dimensions_raises() -> None:
    """Verify spatial dimension mismatch raises DimensionMismatchError."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((60, 50), dtype=bool)
    with pytest.raises(DimensionMismatchError):
        extract_tomato_features(img, mask)


def test_tomato_automatic_mask_generation() -> None:
    """Verify automatic Otsu mask generation when mask is None."""
    h, w = 60, 60
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[15:45, 15:45] = [40, 200, 50]

    vec = extract_tomato_features(img, mask=None)
    assert isinstance(vec, FeatureVector)
    assert len(vec.values) == 4
    assert vec.to_dict()["solidity"] > 0.8


def test_tomato_run_verification_script() -> None:
    """Verify verification script executes and passes completely."""
    assert run_verification() is True
