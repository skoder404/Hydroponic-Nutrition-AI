"""Lettuce leaf image feature extraction (7 features per Overview §16.1).

This module extracts the standardized 7-feature vector from individual lettuce leaf crop images:
1. exg_mean: Mean Excess Green index of foreground leaf pixels (2g - r - b)
2. exr_mean: Mean Excess Red index of foreground leaf pixels (1.4r - g)
3. yellow_pixel_ratio: Proportion of foreground pixels matching chlorotic yellow HSV bounds
4. brown_pixel_ratio: Proportion of foreground pixels matching necrotic brown HSV bounds
5. mean_hue: Mean hue channel intensity of foreground leaf pixels in OpenCV scale [0, 180]
6. mean_saturation: Mean saturation channel intensity of foreground leaf pixels in OpenCV scale [0, 255]
7. contour_compactness: Isoperimetric quotient = 4 * pi * Area / (Perimeter^2)

================================================================================
Design Decisions & Contracts (T5-004)
================================================================================

1. Feature Ordering & Naming:
   Strictly follows Overview §16.1:
   ['exg_mean', 'exr_mean', 'yellow_pixel_ratio', 'brown_pixel_ratio',
    'mean_hue', 'mean_saturation', 'contour_compactness']

2. Configurable HSV Thresholds:
   HSV thresholds for chlorotic yellow and necrotic brown are NOT hard-coded.
   They are encapsulated in `LettuceFeatureConfig` and persisted with each
   extracted `FeatureVector` in `extraction_config` for complete provenance.

3. Zero Perimeter Handling:
   Per Overview §16.1 and TECH_RULES, if the leaf contour perimeter is zero or
   non-positive (e.g. single isolated pixel or degenerate geometry), an explicit
   `ZeroPerimeterError` is raised. Features are never silently replaced by 0.

4. Numerical Stability & Validation:
   Every feature value is validated using `assert_finite_features` to guarantee
   no NaN or +/-Inf enters downstream models.
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
    ZeroPerimeterError,
    assert_finite_features,
    select_masked_pixels,
)
from ml.features.spinach import (
    compute_excess_green,
    compute_excess_red,
    compute_hsv_ratio,
)

# Canonical 7-feature ordering defined in Overview §16.1
LETTUCE_FEATURE_NAMES: tuple[str, ...] = (
    "exg_mean",
    "exr_mean",
    "yellow_pixel_ratio",
    "brown_pixel_ratio",
    "mean_hue",
    "mean_saturation",
    "contour_compactness",
)


@dataclass(frozen=True)
class LettuceFeatureConfig:
    """Configuration for lettuce feature extraction.

    All color thresholds are configurable and serialized into extraction metadata.
    """

    schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION

    # HSV thresholds for chlorotic yellow detection:
    # In OpenCV HSV: H in [0, 180], S in [0, 255], V in [0, 255]
    yellow_hue_min: float = 18.0
    yellow_hue_max: float = 38.0
    yellow_sat_min: float = 40.0
    yellow_sat_max: float = 255.0
    yellow_val_min: float = 50.0
    yellow_val_max: float = 255.0

    # HSV thresholds for necrotic brown detection:
    brown_hue_min: float = 5.0
    brown_hue_max: float = 22.0
    brown_sat_min: float = 30.0
    brown_sat_max: float = 255.0
    brown_val_min: float = 20.0
    brown_val_max: float = 160.0

    def to_dict(self) -> dict[str, str | float | int | list]:
        """Serialize configuration to a JSON-compatible dictionary."""
        d = asdict(self)
        d["crop"] = "lettuce"
        d["feature_names"] = list(LETTUCE_FEATURE_NAMES)
        return d


DEFAULT_LETTUCE_CONFIG = LettuceFeatureConfig()


def compute_mean_hue_and_saturation(rgb_pixels: np.ndarray) -> tuple[float, float]:
    """Compute mean hue and saturation for foreground RGB pixels in OpenCV HSV scale.

    Parameters
    ----------
    rgb_pixels : np.ndarray
        Foreground pixels of shape (N, 3) in RGB order.

    Returns
    -------
    Tuple[float, float]
        (mean_hue, mean_saturation) where hue is in [0, 180] and saturation in [0, 255].
    """
    n_pixels = len(rgb_pixels)
    if n_pixels == 0:
        return 0.0, 0.0

    if np.issubdtype(rgb_pixels.dtype, np.floating):
        if np.nanmax(rgb_pixels) <= 1.0 + 1e-6:
            pixels_uint8 = np.clip(rgb_pixels * 255.0, 0, 255).astype(np.uint8)
        else:
            pixels_uint8 = np.clip(rgb_pixels, 0, 255).astype(np.uint8)
    else:
        pixels_uint8 = rgb_pixels.astype(np.uint8)

    pixels_rgb = pixels_uint8.reshape(-1, 1, 3)
    pixels_hsv = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    mean_hue = float(np.mean(pixels_hsv[:, 0]))
    mean_sat = float(np.mean(pixels_hsv[:, 1]))
    return mean_hue, mean_sat


def compute_contour_compactness(mask: np.ndarray) -> float:
    """Compute contour compactness = 4 * pi * Area / (Perimeter^2).

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask of shape (H, W) where foreground pixels are non-zero / True.

    Returns
    -------
    float
        Isoperimetric compactness measure. For a perfect circle, value = 1.0.

    Raises
    ------
    EmptyMaskError
        If mask contains no foreground pixels.
    ZeroPerimeterError
        If the extracted leaf contour has zero or non-positive perimeter.
    """
    if mask.ndim != 2:
        raise DimensionMismatchError(
            f"Expected 2D mask (H, W), got {mask.ndim}D shape {mask.shape}"
        )

    mask_uint8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(mask_uint8) == 0:
        raise EmptyMaskError(
            "Cannot compute contour compactness: mask contains zero foreground pixels."
        )

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ZeroPerimeterError(
            "Cannot compute contour compactness: no contours detected in mask."
        )

    # Select the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, closed=True))

    if perimeter <= 0.0:
        raise ZeroPerimeterError(
            f"Cannot compute contour compactness: leaf contour perimeter is zero or non-positive ({perimeter})."
        )

    compactness = (4.0 * np.pi * area) / (perimeter**2)
    return float(compactness)


def extract_lettuce_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    config: LettuceFeatureConfig | None = None,
) -> FeatureVector:
    """Extract standardized 7-feature vector from a lettuce leaf crop image.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    mask : Optional[np.ndarray]
        Optional binary segmentation mask of shape (H, W).
        If None, an Otsu-based leaf foreground mask is automatically derived.
    config : Optional[LettuceFeatureConfig]
        Feature extraction parameters and HSV thresholds. If None,
        `DEFAULT_LETTUCE_CONFIG` is used.

    Returns
    -------
    FeatureVector
        Container with:
        - schema_version: config.schema_version
        - feature_names: LETTUCE_FEATURE_NAMES
        - values: list of 7 float values:
            1. exg_mean: Excess Green index mean (2g - r - b)
            2. exr_mean: Excess Red index mean (1.4r - g)
            3. yellow_pixel_ratio: Ratio of chlorotic yellow foreground pixels
            4. brown_pixel_ratio: Ratio of necrotic brown foreground pixels
            5. mean_hue: Mean hue of foreground pixels in OpenCV scale [0, 180]
            6. mean_saturation: Mean saturation of foreground pixels [0, 255]
            7. contour_compactness: 4 * pi * Area / (Perimeter^2)
        - extraction_config: metadata dictionary documenting all parameters and thresholds.

    Raises
    ------
    EmptyMaskError
        If mask contains no foreground pixels.
    DimensionMismatchError
        If mask shape does not match image dimensions (H, W).
    ZeroPerimeterError
        If the leaf contour has zero perimeter.
    NonFiniteFeatureError
        If any extracted feature is NaN or Inf.
    """
    cfg = config or DEFAULT_LETTUCE_CONFIG

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be np.ndarray, got {type(image).__name__}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise DimensionMismatchError(
            f"Expected RGB image of shape (H, W, 3), got {image.shape}"
        )

    h, w = image.shape[:2]

    # Resolve or derive segmentation mask
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"Expected mask to be np.ndarray, got {type(mask).__name__}"
            )
        mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
        if mask_2d.shape != (h, w):
            raise DimensionMismatchError(
                f"Mask dimensions {mask_2d.shape} do not match image dimensions {(h, w)}"
            )
        binary_mask = mask_2d.astype(bool)
    else:
        # Automatic leaf foreground segmentation for individual leaf crop
        # Convert to grayscale and apply Otsu thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = thresh > 0

    foreground_count = int(np.sum(binary_mask))
    if foreground_count == 0:
        raise EmptyMaskError(
            "Cannot extract lettuce features: mask is empty (zero foreground pixels)."
        )

    # 1. Select foreground pixels
    fg_pixels = select_masked_pixels(image, binary_mask, return_flat=True)

    # 2. Compute ExG mean and ExR mean
    exg_vals = compute_excess_green(fg_pixels)
    exr_vals = compute_excess_red(fg_pixels)
    exg_mean = float(np.mean(exg_vals))
    exr_mean = float(np.mean(exr_vals))

    # 3. Compute Yellow Pixel Ratio using configured HSV thresholds
    yellow_ratio = compute_hsv_ratio(
        fg_pixels,
        hue_range=(cfg.yellow_hue_min, cfg.yellow_hue_max),
        sat_range=(cfg.yellow_sat_min, cfg.yellow_sat_max),
        val_range=(cfg.yellow_val_min, cfg.yellow_val_max),
    )

    # 4. Compute Brown Pixel Ratio using configured HSV thresholds
    brown_ratio = compute_hsv_ratio(
        fg_pixels,
        hue_range=(cfg.brown_hue_min, cfg.brown_hue_max),
        sat_range=(cfg.brown_sat_min, cfg.brown_sat_max),
        val_range=(cfg.brown_val_min, cfg.brown_val_max),
    )

    # 5. Compute Mean Hue and Mean Saturation
    mean_hue, mean_saturation = compute_mean_hue_and_saturation(fg_pixels)

    # 6. Compute Contour Compactness (raises ZeroPerimeterError if perimeter <= 0)
    contour_compactness = compute_contour_compactness(binary_mask)

    # Assemble feature dictionary
    feature_dict: dict[str, float] = {
        "exg_mean": exg_mean,
        "exr_mean": exr_mean,
        "yellow_pixel_ratio": yellow_ratio,
        "brown_pixel_ratio": brown_ratio,
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "contour_compactness": contour_compactness,
    }

    # Strict NaN/Inf check (raises NonFiniteFeatureError if any value is non-finite)
    assert_finite_features(feature_dict, list(LETTUCE_FEATURE_NAMES))

    ordered_values = [feature_dict[name] for name in LETTUCE_FEATURE_NAMES]

    extraction_config = cfg.to_dict()
    extraction_config["image_shape"] = [h, w, 3]
    extraction_config["foreground_pixels"] = foreground_count
    extraction_config["total_image_pixels"] = int(h * w)

    return FeatureVector(
        schema_version=cfg.schema_version,
        values=ordered_values,
        feature_names=list(LETTUCE_FEATURE_NAMES),
        crop="lettuce",
        extraction_config=extraction_config,
    )
