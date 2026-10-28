"""Unit tests for GLCM texture feature extraction (T5-003)."""

from __future__ import annotations

import inspect
import numpy as np
import pytest

import ml.features.glcm as glcm_module
from ml.features.common import (
    DimensionMismatchError,
    EmptyMaskError,
    NonFiniteFeatureError,
)
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
from scripts.verify_glcm import run_verification


def test_glcm_docstring_documents_flags_and_log_base():
    """Verify acceptance criterion 1: Flags and log base are documented in the module docstring."""
    doc = glcm_module.__doc__
    assert doc is not None, "ml.features.glcm must have a module docstring"
    assert "symmetric" in doc, "Docstring must document symmetric flag"
    assert "normed" in doc, "Docstring must document normed flag"
    assert "base 2" in doc or "log_base" in doc, "Docstring must document base-2 logarithm"

    # Also verify default constants match documented rules
    assert DEFAULT_LEVELS == 32
    assert DEFAULT_DISTANCES == (1,)
    assert len(DEFAULT_ANGLES) == 4
    assert DEFAULT_SYMMETRIC is True
    assert DEFAULT_NORMED is True
    assert DEFAULT_LOG_BASE == 2.0


def test_glcm_constant_image_zero_contrast_and_zero_entropy():
    """Verify acceptance criterion 2: Constant-colour image does not crash or yield NaN."""
    # Test multiple types of constant images
    constant_cases = [
        np.zeros((30, 30), dtype=np.uint8),
        np.full((20, 20), 127, dtype=np.uint8),
        np.full((25, 25), 255, dtype=np.uint8),
        np.ones((15, 15), dtype=np.float32),
        np.full((30, 30, 3), 200, dtype=np.uint8),  # RGB constant
    ]

    for img in constant_cases:
        features = extract_glcm_features(img)
        assert "glcm_contrast" in features
        assert "glcm_entropy" in features

        contrast = features["glcm_contrast"]
        entropy = features["glcm_entropy"]

        assert np.isfinite(contrast), f"Contrast was non-finite on constant image: {contrast}"
        assert np.isfinite(entropy), f"Entropy was non-finite on constant image: {entropy}"
        assert abs(contrast) < 1e-6, f"Expected 0.0 contrast, got {contrast}"
        assert abs(entropy) < 1e-6, f"Expected 0.0 entropy, got {entropy}"


def test_glcm_hand_computed_4x4_analytical_match():
    """Verify acceptance criterion 3: Output matches a hand-computed 4x4 example."""
    hand_4x4 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ], dtype=np.uint8)

    # 1. Direct GLCM computation on integer grid
    glcm = compute_glcm(hand_4x4, levels=32, distances=(1,), angles=DEFAULT_ANGLES)
    c_direct = compute_glcm_contrast(glcm)
    h_direct = compute_glcm_entropy(glcm, log_base=2.0)

    # Analytical derivation:
    # Contrast: 0 deg = 1/3, 45 deg = 19/9, 90 deg = 4/3, 135 deg = 11/9. Mean = 5/4 = 1.25.
    # Entropy: 0 deg = 4/3 + log2(3) = 2.918296, 45 deg = 3.725481. Mean = 3.321888.
    assert pytest.approx(c_direct, abs=1e-5) == 1.250000
    assert pytest.approx(h_direct, abs=1e-5) == 3.321888

    # 2. End-to-end extraction through extract_glcm_features
    # Map values to uint8 bins (bin 0: 4, bin 1: 12, bin 2: 20, bin 3: 28)
    img_mapped = (hand_4x4 * 8 + 4).astype(np.uint8)
    features = extract_glcm_features(img_mapped, levels=32)
    assert pytest.approx(features["glcm_contrast"], abs=1e-5) == 1.250000
    assert pytest.approx(features["glcm_entropy"], abs=1e-5) == 3.321888


def test_quantize_gray():
    """Test quantization into discrete levels."""
    # Float [0.0, 1.0]
    float_img = np.array([[0.0, 0.5, 0.999]], dtype=np.float64)
    q_float = quantize_gray(float_img, levels=32)
    assert q_float.shape == (1, 3)
    assert q_float.dtype == np.uint8
    assert q_float[0, 0] == 0
    assert q_float[0, 1] == 16
    assert q_float[0, 2] == 31

    # Uint8 [0, 255]
    uint8_img = np.array([[0, 7, 8, 255]], dtype=np.uint8)
    q_uint8 = quantize_gray(uint8_img, levels=32)
    assert q_uint8[0, 0] == 0
    assert q_uint8[0, 1] == 0
    assert q_uint8[0, 2] == 1
    assert q_uint8[0, 3] == 31

    # Invalid levels
    with pytest.raises(ValueError, match="levels must be in range"):
        quantize_gray(uint8_img, levels=1)
    with pytest.raises(ValueError, match="levels must be in range"):
        quantize_gray(uint8_img, levels=300)


def test_to_grayscale():
    """Test color-to-grayscale conversion."""
    # 2D already grayscale
    gray_2d = np.full((10, 10), 128, dtype=np.uint8)
    conv_2d = to_grayscale(gray_2d)
    assert conv_2d.shape == (10, 10)
    assert conv_2d.dtype == np.float64
    assert pytest.approx(conv_2d[0, 0], abs=1e-3) == 128 / 255.0

    # 3D RGB array
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # Red channel
    conv_rgb = to_grayscale(rgb)
    assert conv_rgb.shape == (10, 10)
    # ITU-R BT.601 or skimage rgb2gray red weight is ~0.2125
    assert conv_rgb[0, 0] > 0.15

    # Invalid dimensions
    with pytest.raises(DimensionMismatchError):
        to_grayscale(np.zeros((10, 10, 4), dtype=np.uint8))
    with pytest.raises(TypeError):
        to_grayscale("not an array")


def test_extract_glcm_features_with_mask():
    """Test GLCM extraction with mask bounding box cropping."""
    img = np.zeros((20, 20), dtype=np.uint8)
    # Put high texture in center 4x4
    img[8:12, 8:12] = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]) * 8 + 4

    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True

    features = extract_glcm_features(img, mask=mask)
    assert pytest.approx(features["glcm_contrast"], abs=1e-5) == 1.250000
    assert pytest.approx(features["glcm_entropy"], abs=1e-5) == 3.321888


def test_extract_glcm_features_empty_mask_raises():
    """Verify empty mask raises EmptyMaskError."""
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(EmptyMaskError):
        extract_glcm_features(img, mask=mask)


def test_extract_glcm_features_mismatched_mask_raises():
    """Verify mismatched mask shape raises DimensionMismatchError."""
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((15, 15), dtype=bool)
    with pytest.raises(DimensionMismatchError):
        extract_glcm_features(img, mask=mask)


def test_glcm_invalid_log_base():
    """Verify invalid log_base raises ValueError."""
    glcm = np.ones((4, 4, 1, 1), dtype=np.float64) / 16.0
    with pytest.raises(ValueError, match="log_base must be positive and != 1.0"):
        compute_glcm_entropy(glcm, log_base=1.0)
    with pytest.raises(ValueError, match="log_base must be positive and != 1.0"):
        compute_glcm_entropy(glcm, log_base=-2.0)


def test_glcm_verification_script():
    """Verify that verify_glcm.py runs and passes completely."""
    assert run_verification() is True
