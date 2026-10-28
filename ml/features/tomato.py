"""Tomato leaf image feature extraction (4 defensible features per ADR 0007).

This module extracts the standardized 4-feature vector from tomato leaf crop images:
1. glcm_contrast: Mean second-order GLCM contrast texture feature across 4 angles (0, 45, 90, 135 deg).
2. glcm_entropy: Mean information-theoretic GLCM entropy in bits (base 2) across 4 angles.
3. bounding_aspect_ratio: Bounding box width divided by height (w / h) via cv2.boundingRect.
4. solidity: Ratio of contour area to convex hull area (contour_area / hull_area).

================================================================================
Design Decisions & Contracts (T5-006, T5-007)
================================================================================

1. droop_angle Removal:
   Per ADR 0007, candidate feature `droop_angle` is formally REMOVED due to lack of
   an objective gravity or plant stem reference in ex-situ single-leaf imagery.
   The canonical feature list consists strictly of the 4 defensible texture and shape features:
   ['glcm_contrast', 'glcm_entropy', 'bounding_aspect_ratio', 'solidity']

2. Aspect Ratio Convention:
   Defined explicitly as the axis-aligned bounding box ratio:
   bounding_aspect_ratio = bbox_width / bbox_height
   measured using `cv2.boundingRect(largest_contour)`. This convention is scale-invariant
   and persisted in metadata.

3. Solidity Convention:
   Defined explicitly as:
   solidity = cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour))
   For any convex shape, solidity = 1.0; lobed or diseased/necrotic eroded leaves exhibit
   significantly lower solidity.

4. Degenerate Contour Handling:
   If the convex hull area is zero or non-positive (e.g. single isolated pixel, colinear
   points, or empty geometry), extraction raises an explicit `DegenerateContourError`.
   Features are never silently replaced by 0 or dummy fallbacks.

5. Numerical Stability & Validation:
   Every feature value is validated using `assert_finite_features` to guarantee
   no NaN or +/-Inf values enter downstream models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from ml.features.common import (
    DEFAULT_FEATURE_SCHEMA_VERSION,
    DegenerateContourError,
    DimensionMismatchError,
    EmptyMaskError,
    FeatureVector,
    assert_finite_features,
)
from ml.features.glcm import (
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    DEFAULT_LEVELS,
    DEFAULT_LOG_BASE,
    extract_glcm_features,
)

# Canonical 4-feature ordering defined in ADR 0007 & T5-007
TOMATO_FEATURE_NAMES: tuple[str, ...] = (
    "glcm_contrast",
    "glcm_entropy",
    "bounding_aspect_ratio",
    "solidity",
)


@dataclass(frozen=True)
class TomatoFeatureConfig:
    """Configuration for tomato feature extraction and metadata provenance."""

    schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION

    # GLCM settings
    glcm_levels: int = DEFAULT_LEVELS
    glcm_distances: tuple[int, ...] = DEFAULT_DISTANCES
    glcm_angles: tuple[float, ...] = DEFAULT_ANGLES
    glcm_log_base: float = DEFAULT_LOG_BASE

    # Shape descriptor definitions
    aspect_ratio_definition: str = (
        "Bounding box width divided by height (w / h) via cv2.boundingRect"
    )
    solidity_definition: str = (
        "Contour area divided by convex hull area (contour_area / hull_area)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary with definitions."""
        d = asdict(self)
        d["glcm_angles"] = list(self.glcm_angles)
        d["glcm_distances"] = list(self.glcm_distances)
        d["crop"] = "tomato"
        d["feature_names"] = list(TOMATO_FEATURE_NAMES)
        d["droop_angle_decision"] = "REMOVED per ADR 0007"
        d["definitions"] = {
            "bounding_aspect_ratio": self.aspect_ratio_definition,
            "solidity": self.solidity_definition,
            "glcm_contrast": "Mean GLCM contrast across 4 angles (32 levels, d=1)",
            "glcm_entropy": "Mean base-2 GLCM entropy across 4 angles (32 levels, d=1)",
        }
        return d


DEFAULT_TOMATO_CONFIG = TomatoFeatureConfig()


def compute_shape_descriptors(mask: np.ndarray) -> tuple[float, float]:
    """Compute bounding aspect ratio and solidity from a binary leaf mask.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask of shape (H, W) where foreground pixels are non-zero / True.

    Returns
    -------
    tuple[float, float]
        (bounding_aspect_ratio, solidity)

    Raises
    ------
    EmptyMaskError
        If mask contains zero foreground pixels.
    DegenerateContourError
        If no contour is found or convex hull area is zero / non-positive.
    """
    if mask.ndim != 2:
        raise DimensionMismatchError(
            f"Expected 2D mask (H, W), got {mask.ndim}D shape {mask.shape}"
        )

    mask_uint8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(mask_uint8) == 0:
        raise EmptyMaskError(
            "Cannot compute shape descriptors: mask contains zero foreground pixels."
        )

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise DegenerateContourError(
            "Cannot compute shape descriptors: no contours found in mask."
        )

    largest_contour = max(contours, key=cv2.contourArea)

    # 1. Bounding aspect ratio
    _, _, w, h = cv2.boundingRect(largest_contour)
    if h <= 0 or w <= 0:
        raise DegenerateContourError(
            f"Degenerate bounding box dimensions: width={w}, height={h}."
        )
    aspect_ratio = float(w / h)

    # 2. Solidity = contour_area / hull_area
    contour_area = float(cv2.contourArea(largest_contour))
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))

    if hull_area <= 0.0:
        raise DegenerateContourError(
            f"Cannot compute solidity: convex hull area is zero or non-positive ({hull_area}). "
            "Contour points are degenerate or colinear."
        )

    solidity = float(contour_area / hull_area)
    return aspect_ratio, solidity


def extract_tomato_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    config: TomatoFeatureConfig | None = None,
) -> FeatureVector:
    """Extract standardized 4-feature vector from a tomato leaf image.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    mask : Optional[np.ndarray]
        Optional binary segmentation mask of shape (H, W).
        If None, an Otsu-based leaf foreground mask is automatically derived.
    config : Optional[TomatoFeatureConfig]
        Extraction parameters. If None, `DEFAULT_TOMATO_CONFIG` is used.

    Returns
    -------
    FeatureVector
        Container with:
        - schema_version: config.schema_version
        - feature_names: TOMATO_FEATURE_NAMES
        - values: list of 4 float values:
            1. glcm_contrast: Mean GLCM contrast texture feature
            2. glcm_entropy: Mean GLCM entropy texture feature (in bits)
            3. bounding_aspect_ratio: Bounding box width / height
            4. solidity: Contour area / convex hull area
        - extraction_config: metadata dictionary documenting exact definitions and ADR 0007 decision.

    Raises
    ------
    EmptyMaskError
        If mask contains no foreground pixels.
    DimensionMismatchError
        If mask shape does not match image dimensions (H, W).
    DegenerateContourError
        If convex hull area is zero or non-positive.
    NonFiniteFeatureError
        If any extracted feature is NaN or Inf.
    """
    cfg = config or DEFAULT_TOMATO_CONFIG

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
        # Automatic leaf foreground segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = thresh > 0

    foreground_count = int(np.sum(binary_mask))
    if foreground_count == 0:
        raise EmptyMaskError(
            "Cannot extract tomato features: mask is empty (zero foreground pixels)."
        )

    # 1. Compute GLCM texture features (contrast and entropy)
    glcm_feats = extract_glcm_features(
        image,
        binary_mask,
        levels=cfg.glcm_levels,
        distances=cfg.glcm_distances,
        angles=cfg.glcm_angles,
        log_base=cfg.glcm_log_base,
    )
    glcm_contrast = float(glcm_feats["glcm_contrast"])
    glcm_entropy = float(glcm_feats["glcm_entropy"])

    # 2. Compute shape descriptors (bounding_aspect_ratio and solidity)
    aspect_ratio, solidity = compute_shape_descriptors(binary_mask)

    feature_dict: dict[str, float] = {
        "glcm_contrast": glcm_contrast,
        "glcm_entropy": glcm_entropy,
        "bounding_aspect_ratio": aspect_ratio,
        "solidity": solidity,
    }

    # Strict NaN/Inf check
    assert_finite_features(feature_dict, list(TOMATO_FEATURE_NAMES))

    ordered_values = [feature_dict[name] for name in TOMATO_FEATURE_NAMES]

    extraction_config = cfg.to_dict()
    extraction_config["image_shape"] = [h, w, 3]
    extraction_config["foreground_pixels"] = foreground_count
    extraction_config["total_image_pixels"] = int(h * w)

    return FeatureVector(
        schema_version=cfg.schema_version,
        values=ordered_values,
        feature_names=list(TOMATO_FEATURE_NAMES),
        crop="tomato",
        extraction_config=extraction_config,
    )
