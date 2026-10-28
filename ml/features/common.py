"""Shared image feature extraction utilities (T5-002).

Adheres to TECH_RULES and Canonical Overview §16:
1. Single Image Loader: Loads images strictly into RGB channel order, documenting
   channel conventions explicitly to prevent the OpenCV BGR-vs-RGB channel-inversion bug.
2. Mask-Aware Pixel Selection: Selects foreground pixels from binary masks (e.g. SAM 2.1),
   with strict validation against empty masks or shape mismatches.
3. Strict NaN/Inf Policy: Non-finite features (NaN or Inf) raise a typed NonFiniteFeatureError;
   a failed feature is NEVER silently replaced by 0.
4. FeatureVector Container: Standardized feature representation attaching mandatory
   `schema_version`, ordered feature names, and extraction configuration to every vector.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Current semantic version for extracted image features schema
DEFAULT_FEATURE_SCHEMA_VERSION = "1.0.0"


# ==============================================================================
# Typed Exceptions
# ==============================================================================


class FeatureExtractionError(ValueError):
    """Base exception for all feature extraction and image preparation failures."""


class NonFiniteFeatureError(FeatureExtractionError):
    """Raised when a computed feature is NaN or +/-Inf.

    Per Overview §16 and TECH_RULES, failed features must raise an explicit
    error and must NEVER be silently imputed with 0.
    """

    def __init__(
        self, feature_name: str, value: float, message: str | None = None
    ) -> None:
        self.feature_name = feature_name
        self.value = value
        msg = (
            message
            or f"Non-finite feature value encountered for '{feature_name}': {value}. "
            "A failed feature raises an explicit error and must never be replaced by 0."
        )
        super().__init__(msg)


class ImageLoadError(FeatureExtractionError):
    """Raised when an image cannot be read, decoded, or has invalid dimensions."""


class EmptyMaskError(FeatureExtractionError):
    """Raised when a provided segmentation mask contains zero foreground pixels."""


class DimensionMismatchError(FeatureExtractionError):
    """Raised when image and segmentation mask spatial dimensions do not match."""


class ZeroPerimeterError(FeatureExtractionError):
    """Raised when leaf contour perimeter is zero, non-positive, or cannot be computed."""


class DegenerateContourError(FeatureExtractionError):
    """Raised when leaf contour is degenerate (e.g. zero convex hull area, colinear points)."""


# ==============================================================================
# Single Image Loader
# ==============================================================================


def load_image(
    source: str | Path | bytes | np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load an image strictly into RGB channel order.

    Channel Convention (Documented per T5-002 Acceptance Criterion 1):
    ------------------------------------------------------------------
    - Channel Order: RGB (Red, Green, Blue).
    - Dimension Layout: (Height, Width, Channels) with Channels == 3.
    - Data Type: np.uint8 with pixel intensities in [0, 255].
    - Rationale: Standard OpenCV `cv2.imread` decodes files in BGR order. Treating
      BGR as RGB corrupts vegetation indices (ExG = 2G - R - B, ExR = 1.4R - G)
      and HSV thresholds. This loader explicitly converts BGR to RGB immediately
      upon reading.

    Args:
        source: File path (str or Path), raw bytes, or an existing numpy array.
        target_size: Optional (width, height) tuple to resize the loaded RGB image.

    Returns:
        np.ndarray: uint8 image array with shape (H, W, 3) in RGB channel order.

    Raises:
        ImageLoadError: If the source does not exist, cannot be decoded, or has
            an unsupported shape.
    """
    img_rgb: np.ndarray

    if isinstance(source, np.ndarray):
        arr = source
        if arr.size == 0:
            raise ImageLoadError("Input numpy array is empty.")

        if arr.ndim == 2:
            # Grayscale -> 3-channel RGB
            img_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 3:
                # Assume caller provided RGB if already array, ensure uint8
                img_rgb = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr.copy()
            elif channels == 4:
                # RGBA -> RGB
                img_rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            else:
                raise ImageLoadError(
                    f"Unsupported channel count: {channels}. Expected 1, 3, or 4 channels."
                )
        else:
            raise ImageLoadError(
                f"Unsupported array dimension: {arr.ndim}. Expected 2D or 3D array."
            )

    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ImageLoadError("Input byte buffer is empty.")
        np_buf = np.frombuffer(source, dtype=np.uint8)
        bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageLoadError("Failed to decode image from byte buffer.")
        # OpenCV decodes as BGR -> convert explicitly to RGB
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(source, (str, Path)):
        path = Path(source).resolve()
        if not path.is_file():
            raise ImageLoadError(f"Image file does not exist at: '{path}'")

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageLoadError(
                f"Failed to read image file from: '{path}' (unsupported or corrupt format)."
            )
        # OpenCV reads as BGR -> convert explicitly to RGB
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    else:
        raise ImageLoadError(f"Unsupported source type: {type(source).__name__}")

    # Optional resize
    if target_size is not None:
        w, h = target_size
        if w <= 0 or h <= 0:
            raise ImageLoadError(
                f"Invalid target_size: {target_size}. Dimensions must be positive."
            )
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_rgb


# ==============================================================================
# Mask-Aware Pixel Selection
# ==============================================================================


def select_masked_pixels(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    return_flat: bool = True,
) -> np.ndarray:
    """Select foreground leaf pixels using an optional binary segmentation mask.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Optional binary mask array of shape (H, W). Non-zero or True entries
            indicate valid foreground leaf pixels.
        return_flat: If True, returns a 2D array of shape (K, 3) containing only the
            foreground pixels. If False, returns a 3D array of shape (H, W, 3) where
            background pixels outside the mask are zeroed.

    Returns:
        np.ndarray: Masked pixels (either flat (K, 3) or spatial (H, W, 3)).

    Raises:
        DimensionMismatchError: If mask spatial dimensions do not match image.
        EmptyMaskError: If the mask contains 0 positive foreground pixels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageLoadError(
            f"Expected 3-channel RGB image (H, W, 3), got shape {image.shape}"
        )

    if mask is None:
        if return_flat:
            return image.reshape(-1, 3)
        return image.copy()

    # Validate mask dimensions
    if mask.ndim != 2:
        raise DimensionMismatchError(
            f"Expected 2D mask array (H, W), got {mask.ndim}D shape {mask.shape}"
        )

    if mask.shape[:2] != image.shape[:2]:
        raise DimensionMismatchError(
            f"Spatial dimension mismatch between image {image.shape[:2]} and mask {mask.shape[:2]}"
        )

    # Boolean foreground indicator
    binary_mask = mask.astype(bool) if mask.dtype != bool else mask
    num_foreground = int(np.count_nonzero(binary_mask))

    if num_foreground == 0:
        raise EmptyMaskError(
            "Segmentation mask is empty (0 foreground pixels). Cannot extract leaf features."
        )

    if return_flat:
        return image[binary_mask]  # Shape: (K, 3)

    # Spatial return: zero out background
    masked_spatial = np.zeros_like(image)
    masked_spatial[binary_mask] = image[binary_mask]
    return masked_spatial


# ==============================================================================
# NaN / Inf Validation
# ==============================================================================


def assert_finite_features(
    features: Mapping[str, float] | Sequence[float],
    feature_names: Sequence[str] | None = None,
) -> None:
    """Verify that all feature values are strictly finite numbers.

    Per Acceptance Criterion 2 and Overview §16:
    A failed feature raises an explicit NonFiniteFeatureError; it is NEVER
    silently replaced by 0.

    Args:
        features: Mapping of name->value or Sequence of numeric values.
        feature_names: Optional ordered names if features is passed as a sequence.

    Raises:
        NonFiniteFeatureError: If any feature value is NaN, +Inf, or -Inf.
    """
    if isinstance(features, Mapping):
        for name, val in features.items():
            if val is None or np.isnan(val) or np.isinf(val):
                raise NonFiniteFeatureError(
                    feature_name=name,
                    value=float("nan") if val is None or np.isnan(val) else float(val),
                )
    else:
        values = list(features)
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(len(values))]
        )
        if len(names) != len(values):
            raise ValueError(
                f"Length mismatch: {len(names)} names vs {len(values)} values."
            )

        for name, val in zip(names, values):
            if val is None or np.isnan(val) or np.isinf(val):
                raise NonFiniteFeatureError(
                    feature_name=name,
                    value=float("nan") if val is None or np.isnan(val) else float(val),
                )


# ==============================================================================
# Standardized FeatureVector Container
# ==============================================================================


@dataclass
class FeatureVector:
    """Standardized image feature extraction container (Acceptance Criterion 3).

    Attributes:
        values: 1D sequence or array of numerical feature values.
        feature_names: Exact ordered names corresponding to values.
        schema_version: Semantic version of the feature schema (mandatory).
        crop: Optional crop family identifier ('lettuce', 'spinach', 'tomato').
        extraction_config: Extraction hyperparameters (HSV bounds, GLCM levels, etc.).
        metadata: Arbitrary provenance and audit metadata.
    """

    values: list[float]
    feature_names: list[str]
    schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION
    crop: str | None = None
    extraction_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 1. Enforce non-empty schema version
        if not self.schema_version or not str(self.schema_version).strip():
            raise ValueError(
                "FeatureVector requires a non-empty schema_version string."
            )

        # 2. Check length consistency
        if len(self.values) != len(self.feature_names):
            raise ValueError(
                f"Dimension mismatch in FeatureVector: {len(self.values)} values "
                f"but {len(self.feature_names)} feature_names."
            )

        # 3. Cast values to Python float
        self.values = [float(v) for v in self.values]
        self.feature_names = [str(n) for n in self.feature_names]

        # 4. Strict NaN/Inf check (Criterion 2)
        assert_finite_features(self.values, self.feature_names)

    def to_dict(self) -> dict[str, float]:
        """Return an ordered dictionary mapping feature names to their values."""
        return dict(zip(self.feature_names, self.values))

    def to_numpy(self) -> np.ndarray:
        """Return feature values as a 1D float64 numpy array."""
        return np.array(self.values, dtype=np.float64)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize FeatureVector into a JSON string."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "crop": self.crop,
                "feature_names": self.feature_names,
                "values": self.values,
                "features": self.to_dict(),
                "extraction_config": self.extraction_config,
                "metadata": self.metadata,
            },
            indent=indent,
        )

    @classmethod
    def from_dict(
        cls,
        feature_dict: Mapping[str, float],
        schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION,
        crop: str | None = None,
        extraction_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureVector:
        """Construct a FeatureVector directly from a name->value mapping."""
        names = list(feature_dict.keys())
        vals = [feature_dict[k] for k in names]
        return cls(
            values=vals,
            feature_names=names,
            schema_version=schema_version,
            crop=crop,
            extraction_config=extraction_config or {},
            metadata=metadata or {},
        )

    @classmethod
    def from_json(cls, json_str: str) -> FeatureVector:
        """Reconstruct a FeatureVector from a JSON string."""
        data = json.loads(json_str)
        return cls(
            values=data["values"],
            feature_names=data["feature_names"],
            schema_version=data.get("schema_version", DEFAULT_FEATURE_SCHEMA_VERSION),
            crop=data.get("crop"),
            extraction_config=data.get("extraction_config", {}),
            metadata=data.get("metadata", {}),
        )
