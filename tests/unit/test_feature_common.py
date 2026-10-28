"""Unit tests for shared image feature extraction utilities (T5-002)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.features.common import (
    DEFAULT_FEATURE_SCHEMA_VERSION,
    DimensionMismatchError,
    EmptyMaskError,
    FeatureExtractionError,
    FeatureVector,
    ImageLoadError,
    NonFiniteFeatureError,
    assert_finite_features,
    load_image,
    select_masked_pixels,
)


# ==============================================================================
# Image Loader Tests (Acceptance Criterion 1)
# ==============================================================================


def test_load_image_documents_channel_order_in_docstring():
    """Acceptance Criterion 1: Loader documents channel order in its docstring."""
    import inspect

    doc = inspect.getdoc(load_image)
    assert doc is not None
    assert "Channel Order" in doc
    assert "RGB" in doc
    assert "BGR" in doc


def test_load_image_converts_bgr_to_rgb(tmp_path: Path):
    """Verify that an image on disk or in bytes is loaded strictly in RGB order."""
    # Create pure Red pixel in BGR [0, 0, 255]
    bgr_img = np.zeros((20, 20, 3), dtype=np.uint8)
    bgr_img[:, :, 2] = 255  # Red channel in BGR

    file_path = tmp_path / "test_red.png"
    cv2.imwrite(str(file_path), bgr_img)

    # Load via path
    loaded = load_image(file_path)
    assert loaded.shape == (20, 20, 3)
    assert loaded.dtype == np.uint8
    # In RGB, Red is channel 0: [255, 0, 0]
    assert loaded[0, 0, 0] == 255
    assert loaded[0, 0, 1] == 0
    assert loaded[0, 0, 2] == 0


def test_load_image_from_bytes():
    """Verify image loading directly from raw bytes."""
    # Pure Green in BGR is [0, 255, 0]
    bgr_img = np.zeros((15, 15, 3), dtype=np.uint8)
    bgr_img[:, :, 1] = 200

    _, buf = cv2.imencode(".jpg", bgr_img)
    raw_bytes = buf.tobytes()

    loaded = load_image(raw_bytes)
    assert loaded.shape == (15, 15, 3)
    # Green is channel 1 in both BGR and RGB
    assert loaded[0, 0, 1] >= 195  # JPEG compression tolerance


def test_load_image_from_numpy_array():
    """Verify loading from existing 2D, 3D, and 4D numpy arrays."""
    # 2D Grayscale -> 3-channel RGB
    gray = np.full((10, 10), 128, dtype=np.uint8)
    rgb = load_image(gray)
    assert rgb.shape == (10, 10, 3)
    assert np.all(rgb == 128)

    # 4D RGBA -> 3-channel RGB
    rgba = np.full((10, 10, 4), 50, dtype=np.uint8)
    rgb4 = load_image(rgba)
    assert rgb4.shape == (10, 10, 3)
    assert np.all(rgb4 == 50)


def test_load_image_resize():
    """Verify target_size parameter resizes correctly."""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = load_image(arr, target_size=(50, 30))
    # (width, height) = (50, 30) -> shape (30, 50, 3)
    assert resized.shape == (30, 50, 3)


def test_load_image_error_cases(tmp_path: Path):
    """Verify typed ImageLoadError on non-existent files or corrupt buffers."""
    with pytest.raises(ImageLoadError, match="does not exist"):
        load_image(tmp_path / "non_existent.jpg")

    with pytest.raises(ImageLoadError, match="empty"):
        load_image(b"")

    with pytest.raises(ImageLoadError, match="Failed to decode"):
        load_image(b"not_an_image_binary_data")

    with pytest.raises(ImageLoadError, match="empty"):
        load_image(np.array([]))

    with pytest.raises(ImageLoadError, match="Unsupported array dimension"):
        load_image(np.zeros((2, 2, 2, 2, 2)))


# ==============================================================================
# Mask-Aware Selection Tests
# ==============================================================================


def test_select_masked_pixels_without_mask():
    """Verify behavior when no mask is provided."""
    img = np.ones((10, 10, 3), dtype=np.uint8) * 42
    flat = select_masked_pixels(img, mask=None, return_flat=True)
    assert flat.shape == (100, 3)
    assert np.all(flat == 42)

    spatial = select_masked_pixels(img, mask=None, return_flat=False)
    assert spatial.shape == (10, 10, 3)
    assert np.all(spatial == 42)


def test_select_masked_pixels_with_valid_mask():
    """Verify selecting foreground pixels using a binary mask."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:5, 2:5] = [10, 20, 30]

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255  # 9 pixels

    # Flat extraction
    pixels = select_masked_pixels(img, mask=mask, return_flat=True)
    assert pixels.shape == (9, 3)
    assert np.all(pixels == [10, 20, 30])

    # Spatial extraction (background zeroed)
    spatial = select_masked_pixels(img, mask=mask, return_flat=False)
    assert spatial.shape == (10, 10, 3)
    assert np.all(spatial[2:5, 2:5] == [10, 20, 30])
    assert np.all(spatial[0:2, 0:2] == 0)


def test_select_masked_pixels_empty_mask_raises():
    """Verify that an all-zero mask raises EmptyMaskError."""
    img = np.ones((10, 10, 3), dtype=np.uint8)
    empty_mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(EmptyMaskError, match="0 foreground pixels"):
        select_masked_pixels(img, mask=empty_mask)


def test_select_masked_pixels_dimension_mismatch_raises():
    """Verify that a shape mismatch raises DimensionMismatchError."""
    img = np.ones((20, 20, 3), dtype=np.uint8)
    bad_mask = np.ones((10, 10), dtype=np.uint8)

    with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
        select_masked_pixels(img, mask=bad_mask)


# ==============================================================================
# NaN / Inf Validation Tests (Acceptance Criterion 2)
# ==============================================================================


def test_assert_finite_features_passes_on_valid_inputs():
    """Valid finite numbers pass without exception."""
    assert_finite_features({"exg": 1.2, "exr": -0.5, "hue": 45.0})
    assert_finite_features([1.0, 2.0, 3.0], ["f1", "f2", "f3"])


def test_assert_finite_features_raises_non_finite_error():
    """Acceptance Criterion 2: NaN/Inf raises a typed error (never zeroed)."""
    # Test NaN in mapping
    with pytest.raises(NonFiniteFeatureError) as exc_info:
        assert_finite_features({"exg": 1.0, "bad_feat": float("nan")})
    assert exc_info.value.feature_name == "bad_feat"
    assert "never be replaced by 0" in str(exc_info.value)

    # Test +Inf in sequence
    with pytest.raises(NonFiniteFeatureError) as exc_info:
        assert_finite_features([1.0, float("inf")], ["a", "inf_feat"])
    assert exc_info.value.feature_name == "inf_feat"

    # Test -Inf
    with pytest.raises(NonFiniteFeatureError) as exc_info:
        assert_finite_features({"neg_inf": float("-inf")})
    assert exc_info.value.feature_name == "neg_inf"


# ==============================================================================
# FeatureVector Container Tests (Acceptance Criterion 3)
# ==============================================================================


def test_feature_vector_attaches_schema_version():
    """Acceptance Criterion 3: Schema version is attached to every vector."""
    vec = FeatureVector(
        values=[1.5, 2.5],
        feature_names=["f1", "f2"],
        schema_version="1.0.0",
        crop="lettuce",
        extraction_config={"hsv_min": [10, 20, 30]},
    )
    assert vec.schema_version == "1.0.0"
    assert vec.crop == "lettuce"
    assert vec.values == [1.5, 2.5]
    assert vec.feature_names == ["f1", "f2"]


def test_feature_vector_serialization_and_deserialization():
    """Verify dictionary, numpy, and JSON roundtrip serialization."""
    vec = FeatureVector(
        values=[0.123, 4.56],
        feature_names=["exg_mean", "glcm_contrast"],
        schema_version="1.0.0",
        crop="spinach",
        metadata={"source_image": "spinach_01.jpg"},
    )

    # to_dict
    d = vec.to_dict()
    assert d == {"exg_mean": 0.123, "glcm_contrast": 4.56}

    # to_numpy
    np_arr = vec.to_numpy()
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.dtype == np.float64
    assert np.allclose(np_arr, [0.123, 4.56])

    # to_json and from_json
    json_str = vec.to_json()
    assert '"schema_version": "1.0.0"' in json_str

    reconstructed = FeatureVector.from_json(json_str)
    assert reconstructed.schema_version == "1.0.0"
    assert reconstructed.crop == "spinach"
    assert reconstructed.values == [0.123, 4.56]
    assert reconstructed.feature_names == ["exg_mean", "glcm_contrast"]
    assert reconstructed.metadata == {"source_image": "spinach_01.jpg"}


def test_feature_vector_from_dict():
    """Verify construction via from_dict factory method."""
    vec = FeatureVector.from_dict(
        {"exg": 1.0, "exr": 2.0},
        schema_version="2.0.0",
        crop="tomato",
    )
    assert vec.schema_version == "2.0.0"
    assert vec.feature_names == ["exg", "exr"]
    assert vec.values == [1.0, 2.0]


def test_feature_vector_validations():
    """Verify constructor error handling for mismatched lengths or invalid versions."""
    # Length mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        FeatureVector(values=[1.0, 2.0], feature_names=["f1"], schema_version="1.0.0")

    # Empty schema_version
    with pytest.raises(ValueError, match="non-empty schema_version"):
        FeatureVector(values=[1.0], feature_names=["f1"], schema_version="")

    # Non-finite value in constructor
    with pytest.raises(NonFiniteFeatureError):
        FeatureVector(values=[float("nan")], feature_names=["f1"], schema_version="1.0.0")
