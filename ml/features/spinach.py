"""Spinach leaf image feature extraction (6 features per Overview §16.2).

This module extracts the standardized 6-feature vector from segmented spinach leaf images:
1. exg_mean: Mean Excess Green index of foreground leaf pixels
2. exr_mean: Mean Excess Red index of foreground leaf pixels
3. yellow_pixel_ratio: Proportion of foreground pixels matching chlorotic yellow HSV bands
4. brown_pixel_ratio: Proportion of foreground pixels matching necrotic brown HSV bands
5. relative_leaf_area: Segmented leaf foreground area divided by total image pixels (H * W)
6. glcm_contrast: Second-order Gray-Level Co-occurrence Matrix contrast (quantized 32, distance 1, 4 angles)

================================================================================
Design Decisions & Contracts (T5-005)
================================================================================

1. Feature Ordering:
   Strictly follows Overview §16.2:
   ['exg_mean', 'exr_mean', 'yellow_pixel_ratio', 'brown_pixel_ratio', 'relative_leaf_area', 'glcm_contrast']

2. Relative Leaf Area Denominator:
   The denominator is explicitly defined as `total_image_pixels = H * W` (the total pixel area of the input frame):
   relative_leaf_area = count(foreground_pixels) / (image_height * image_width)
   This convention is stored in the extraction configuration metadata and guarantees
   scale-invariant canopy fraction measurements across cameras of known resolution.

3. Empty Mask Handling:
   If the segmentation mask contains zero foreground pixels (or is empty / False everywhere),
   feature extraction immediately raises a typed `EmptyMaskError`. No silent zeroing or
   dummy fallbacks are allowed.

4. Numerical Stability:
   Every feature value is strictly validated using `assert_finite_features` to guarantee
   no NaN or Inf values enter downstream pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ml.features.common import (
    DEFAULT_FEATURE_SCHEMA_VERSION,
    DimensionMismatchError,
    EmptyMaskError,
    FeatureVector,
    assert_finite_features,
    select_masked_pixels,
)
from ml.features.glcm import (
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    DEFAULT_LEVELS,
    DEFAULT_LOG_BASE,
    compute_glcm,
    compute_glcm_contrast,
)

# Canonical 6-feature ordering defined in Overview §16.2
SPINACH_FEATURE_NAMES: tuple[str, ...] = (
    "exg_mean",
    "exr_mean",
    "yellow_pixel_ratio",
    "brown_pixel_ratio",
    "relative_leaf_area",
    "glcm_contrast",
)


@dataclass(frozen=True)
class SpinachFeatureConfig:
    """Configuration for spinach feature extraction."""

    # Schema version
    schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION

    # Relative leaf area denominator convention
    relative_leaf_area_denominator: str = "total_image_pixels (H * W)"

    # HSV thresholds for yellow (chlorosis) detection:
    # In OpenCV HSV: H in [0, 180], S in [0, 255], V in [0, 255]
    # Yellow hue is ~60 deg, which corresponds to H ~ 20..35 in OpenCV scale
    yellow_hue_min: float = 18.0
    yellow_hue_max: float = 38.0
    yellow_sat_min: float = 40.0
    yellow_sat_max: float = 255.0
    yellow_val_min: float = 50.0
    yellow_val_max: float = 255.0

    # HSV thresholds for brown (necrosis) detection:
    # Low-to-moderate orange/brown hue, moderate saturation, low-to-medium value
    brown_hue_min: float = 5.0
    brown_hue_max: float = 22.0
    brown_sat_min: float = 30.0
    brown_sat_max: float = 255.0
    brown_val_min: float = 20.0
    brown_val_max: float = 160.0

    # GLCM settings
    glcm_levels: int = DEFAULT_LEVELS
    glcm_distances: tuple[int, ...] = DEFAULT_DISTANCES
    glcm_angles: tuple[float, ...] = DEFAULT_ANGLES
    glcm_log_base: float = DEFAULT_LOG_BASE

    def to_dict(self) -> dict[str, str | float | int | list]:
        """Serialize configuration to a JSON-compatible dictionary."""
        d = asdict(self)
        d["glcm_angles"] = list(self.glcm_angles)
        d["glcm_distances"] = list(self.glcm_distances)
        d["crop"] = "spinach"
        d["feature_names"] = list(SPINACH_FEATURE_NAMES)
        return d


DEFAULT_SPINACH_CONFIG = SpinachFeatureConfig()


def compute_excess_green(rgb_pixels: np.ndarray) -> np.ndarray:
    """Compute Excess Green index (ExG = 2g - r - b) for RGB pixel arrays.

    Parameters
    ----------
    rgb_pixels : np.ndarray
        Array of shape (N, 3) with RGB values in [0..255] (uint8) or [0..1] (float).

    Returns
    -------
    np.ndarray
        1D float64 array of ExG values for each pixel.
    """
    if rgb_pixels.size == 0:
        return np.empty(0, dtype=np.float64)

    # Convert to float64 normalized chromaticity coordinates
    pixels = rgb_pixels.astype(np.float64)
    if np.nanmax(pixels) > 1.0:
        # Scale to [0, 1]
        pixels /= 255.0

    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    sum_rgb = r + g + b
    # Avoid division by zero on pitch-black pixels
    eps = 1e-6
    valid = sum_rgb > eps

    r_norm = np.zeros_like(r)
    g_norm = np.zeros_like(g)
    b_norm = np.zeros_like(b)

    r_norm[valid] = r[valid] / sum_rgb[valid]
    g_norm[valid] = g[valid] / sum_rgb[valid]
    b_norm[valid] = b[valid] / sum_rgb[valid]

    # ExG = 2 * g - r - b
    return 2.0 * g_norm - r_norm - b_norm


def compute_excess_red(rgb_pixels: np.ndarray) -> np.ndarray:
    """Compute Excess Red index (ExR = 1.4r - g) for RGB pixel arrays.

    Parameters
    ----------
    rgb_pixels : np.ndarray
        Array of shape (N, 3) with RGB values in [0..255] (uint8) or [0..1] (float).

    Returns
    -------
    np.ndarray
        1D float64 array of ExR values for each pixel.
    """
    if rgb_pixels.size == 0:
        return np.empty(0, dtype=np.float64)

    pixels = rgb_pixels.astype(np.float64)
    if np.nanmax(pixels) > 1.0:
        pixels /= 255.0

    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    sum_rgb = r + g + b
    eps = 1e-6
    valid = sum_rgb > eps

    r_norm = np.zeros_like(r)
    g_norm = np.zeros_like(g)

    r_norm[valid] = r[valid] / sum_rgb[valid]
    g_norm[valid] = g[valid] / sum_rgb[valid]

    # ExR = 1.4 * r - g (Meyer & Neto standard agronomic definition)
    return 1.4 * r_norm - g_norm


def compute_hsv_ratio(
    rgb_pixels: np.ndarray,
    hue_range: tuple[float, float],
    sat_range: tuple[float, float],
    val_range: tuple[float, float],
) -> float:
    """Compute the fraction of foreground pixels falling within specified HSV bounds.

    Parameters
    ----------
    rgb_pixels : np.ndarray
        Foreground pixels of shape (N, 3) with RGB values in [0..255] (uint8).
    hue_range : Tuple[float, float]
        (min_hue, max_hue) in OpenCV [0, 180] scale.
    sat_range : Tuple[float, float]
        (min_sat, max_sat) in OpenCV [0, 255] scale.
    val_range : Tuple[float, float]
        (min_val, max_val) in OpenCV [0, 255] scale.

    Returns
    -------
    float
        Ratio in [0.0, 1.0].
    """
    n_pixels = len(rgb_pixels)
    if n_pixels == 0:
        return 0.0

    # Ensure uint8 for cv2.cvtColor
    if np.issubdtype(rgb_pixels.dtype, np.floating):
        if np.nanmax(rgb_pixels) <= 1.0 + 1e-6:
            pixels_uint8 = np.clip(rgb_pixels * 255.0, 0, 255).astype(np.uint8)
        else:
            pixels_uint8 = np.clip(rgb_pixels, 0, 255).astype(np.uint8)
    else:
        pixels_uint8 = rgb_pixels.astype(np.uint8)

    # Reshape (N, 1, 3) to convert from RGB to HSV using OpenCV
    pixels_rgb = pixels_uint8.reshape(-1, 1, 3)
    pixels_hsv = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h = pixels_hsv[:, 0].astype(np.float64)
    s = pixels_hsv[:, 1].astype(np.float64)
    v = pixels_hsv[:, 2].astype(np.float64)

    h_min, h_max = hue_range
    s_min, s_max = sat_range
    v_min, v_max = val_range

    mask_h = (h >= h_min) & (h <= h_max)
    mask_s = (s >= s_min) & (s <= s_max)
    mask_v = (v >= v_min) & (v <= v_max)

    matching_count = int(np.sum(mask_h & mask_s & mask_v))
    return float(matching_count / n_pixels)


def extract_spinach_features(
    image: np.ndarray,
    mask: np.ndarray,
    config: SpinachFeatureConfig | None = None,
) -> FeatureVector:
    """Extract standardized 6-feature vector from a spinach leaf image and mask.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    mask : np.ndarray
        Binary segmentation mask of shape (H, W).
        Non-zero/True entries indicate leaf foreground pixels.
    config : Optional[SpinachFeatureConfig]
        Feature extraction parameters. If None, `DEFAULT_SPINACH_CONFIG` is used.

    Returns
    -------
    FeatureVector
        Container with:
        - schema_version: config.schema_version
        - feature_names: SPINACH_FEATURE_NAMES
        - values: list of 6 float values:
            1. exg_mean: Excess Green index mean
            2. exr_mean: Excess Red index mean
            3. yellow_pixel_ratio: Ratio of chlorotic yellow foreground pixels
            4. brown_pixel_ratio: Ratio of necrotic brown foreground pixels
            5. relative_leaf_area: Fractional leaf canopy coverage. The denominator
               is explicitly documented as total_image_pixels = H * W (total image pixels).
            6. glcm_contrast: Second-order GLCM contrast texture feature
        - extraction_config: metadata dictionary documenting all parameters and denominator.

    Raises
    ------
    EmptyMaskError
        If mask contains no foreground pixels.
    DimensionMismatchError
        If mask shape does not match image (H, W).
    NonFiniteFeatureError
        If any extracted feature is NaN or Inf.
    """
    cfg = config or DEFAULT_SPINACH_CONFIG

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be np.ndarray, got {type(image).__name__}")
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected mask to be np.ndarray, got {type(mask).__name__}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise DimensionMismatchError(
            f"Expected RGB image of shape (H, W, 3), got {image.shape}"
        )

    h, w = image.shape[:2]
    mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
    if mask_2d.shape != (h, w):
        raise DimensionMismatchError(
            f"Mask dimensions {mask_2d.shape} do not match image dimensions {(h, w)}"
        )

    # 1. Validate mask & select foreground pixels
    binary_mask = mask_2d.astype(bool)
    foreground_count = int(np.sum(binary_mask))
    if foreground_count == 0:
        raise EmptyMaskError(
            "Cannot extract spinach features: mask is empty (zero foreground pixels)."
        )

    # Extract foreground pixels (K, 3)
    fg_pixels = select_masked_pixels(image, binary_mask, return_flat=True)

    # 2. Compute ExG mean and ExR mean
    exg_vals = compute_excess_green(fg_pixels)
    exr_vals = compute_excess_red(fg_pixels)
    exg_mean = float(np.mean(exg_vals))
    exr_mean = float(np.mean(exr_vals))

    # 3. Compute Yellow Pixel Ratio (Chlorosis)
    yellow_ratio = compute_hsv_ratio(
        fg_pixels,
        hue_range=(cfg.yellow_hue_min, cfg.yellow_hue_max),
        sat_range=(cfg.yellow_sat_min, cfg.yellow_sat_max),
        val_range=(cfg.yellow_val_min, cfg.yellow_val_max),
    )

    # 4. Compute Brown Pixel Ratio (Necrosis)
    brown_ratio = compute_hsv_ratio(
        fg_pixels,
        hue_range=(cfg.brown_hue_min, cfg.brown_hue_max),
        sat_range=(cfg.brown_sat_min, cfg.brown_sat_max),
        val_range=(cfg.brown_val_min, cfg.brown_val_max),
    )

    # 5. Compute Relative Leaf Area (Denominator: total image pixels H * W)
    total_image_pixels = float(h * w)
    relative_leaf_area = float(foreground_count / total_image_pixels)

    # 6. Compute GLCM Contrast (Overview §16.4 / T5-003)
    # Crop to foreground bounding box for texture analysis
    rows, cols = np.where(binary_mask)
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    cropped_image = image[min_r : max_r + 1, min_c : max_c + 1]

    # Convert cropped region to grayscale and quantize to 32 levels
    from ml.features.glcm import quantize_gray, to_grayscale

    cropped_gray = to_grayscale(cropped_image)
    cropped_quantized = quantize_gray(cropped_gray, levels=cfg.glcm_levels)

    glcm = compute_glcm(
        cropped_quantized,
        levels=cfg.glcm_levels,
        distances=cfg.glcm_distances,
        angles=cfg.glcm_angles,
        symmetric=True,
        normed=True,
    )
    glcm_contrast_val = compute_glcm_contrast(glcm)

    # Assemble feature vector in the strict canonical order
    feature_dict = {
        "exg_mean": exg_mean,
        "exr_mean": exr_mean,
        "yellow_pixel_ratio": yellow_ratio,
        "brown_pixel_ratio": brown_ratio,
        "relative_leaf_area": relative_leaf_area,
        "glcm_contrast": glcm_contrast_val,
    }

    # Strict NaN/Inf check (raises NonFiniteFeatureError if any value is non-finite)
    assert_finite_features(feature_dict, list(SPINACH_FEATURE_NAMES))

    ordered_values = [feature_dict[name] for name in SPINACH_FEATURE_NAMES]

    extraction_config = cfg.to_dict()
    extraction_config["image_shape"] = [h, w, 3]
    extraction_config["foreground_pixels"] = foreground_count
    extraction_config["total_image_pixels"] = int(total_image_pixels)

    return FeatureVector(
        schema_version=cfg.schema_version,
        values=ordered_values,
        feature_names=list(SPINACH_FEATURE_NAMES),
        crop="spinach",
        extraction_config=extraction_config,
    )
