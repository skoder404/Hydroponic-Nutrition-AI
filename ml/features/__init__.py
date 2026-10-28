"""Image feature extraction module for hydroponic crop anomaly detection."""

from __future__ import annotations

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
from ml.features.glcm import (
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    DEFAULT_LEVELS,
    DEFAULT_LOG_BASE,
    compute_glcm,
    compute_glcm_contrast,
    compute_glcm_entropy,
    extract_glcm_features,
    quantize_gray,
    to_grayscale,
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

__all__ = [
    "DEFAULT_ANGLES",
    "DEFAULT_DISTANCES",
    "DEFAULT_FEATURE_SCHEMA_VERSION",
    "DEFAULT_LEVELS",
    "DEFAULT_LOG_BASE",
    "DEFAULT_SPINACH_CONFIG",
    "DimensionMismatchError",
    "EmptyMaskError",
    "FeatureExtractionError",
    "FeatureVector",
    "ImageLoadError",
    "NonFiniteFeatureError",
    "SPINACH_FEATURE_NAMES",
    "SpinachFeatureConfig",
    "assert_finite_features",
    "compute_excess_green",
    "compute_excess_red",
    "compute_glcm",
    "compute_glcm_contrast",
    "compute_glcm_entropy",
    "compute_hsv_ratio",
    "extract_glcm_features",
    "extract_spinach_features",
    "load_image",
    "quantize_gray",
    "select_masked_pixels",
    "to_grayscale",
]
