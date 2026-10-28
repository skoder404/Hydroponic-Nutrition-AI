"""Gray-Level Co-occurrence Matrix (GLCM) Texture Feature Extraction.

This module provides utilities to extract texture features (contrast and entropy)
from RGB or grayscale plant leaf images using Gray-Level Co-occurrence Matrices.

================================================================================
Mathematical Formulation and Default Conventions (Overview §16.4)
================================================================================

1. Color Conversion & Quantization:
   - RGB images are converted to single-channel luminance (grayscale) using standard
     ITU-R BT.601 weights: Y = 0.2989 * R + 0.5870 * G + 0.1140 * B.
   - Grayscale intensities [0, 255] are uniformly quantized into 32 discrete bins
     (levels = 32), where bin width is 256 / 32 = 8:
     level(Y) = min(floor(Y / 8), 31) in range [0, 31].

2. Offsets, Distances, and Angles:
   - Distance: d = 1 pixel neighbor.
   - Four directional angles: [0, 45, 90, 135] degrees, corresponding to radians
     [0, pi/4, pi/2, 3*pi/4].
   - Features (contrast and entropy) are computed for each direction and averaged
     across all four angles to ensure rotational invariance.

3. Matrix Properties & Flags:
   - symmetric = True:
     For every pixel pair (i, j) separated by offset (dr, dc), both transitions
     (i -> j) and (j -> i) are accumulated into the co-occurrence matrix:
     P_sym(i, j) = P(i, j) + P(j, i).
     This guarantees P_sym(i, j) = P_sym(j, i) and makes the texture analysis
     invariant to scanning direction.
   - normed = True:
     Each directional matrix is normalized by the total number of pair transitions
     so that the sum of all elements equals 1.0:
     sum_{i=0}^{N-1} sum_{j=0}^{N-1} P(i, j) = 1.0.
     The normalized matrix represents an empirical joint probability mass function
     of pixel intensity pairs.

4. Feature Definitions:
   - GLCM Contrast:
     Measures local intensity variation and edge contrast in the image:
     Contrast_theta = sum_{i=0}^{N-1} sum_{j=0}^{N-1} (i - j)^2 * P(i, j; theta)
     Reported feature 'glcm_contrast' is the arithmetic mean across the 4 angles.

   - GLCM Entropy:
     Measures textural randomness, complexity, and disorder of gray-level distribution.
     Logarithm base: base 2 (log_base = 2.0).
     Entropy is therefore expressed in bits (shannons):
     Entropy_theta = - sum_{i=0}^{N-1} sum_{j=0}^{N-1} P(i, j; theta) * log2(P(i, j; theta))
     evaluated strictly for non-zero probabilities (P(i, j) > 0), because
     lim_{p -> 0+} p * log2(p) = 0.
     Reported feature 'glcm_entropy' is the arithmetic mean across the 4 angles.

5. Edge Case Behavior:
   - Uniform / constant-color images:
     All pixel pairs have identical intensity (i = j), yielding Contrast = 0.0
     and a single probability P(v, v) = 1.0. Because 1.0 * log2(1.0) = 0.0,
     Entropy = 0.0. The implementation never evaluates log(0) and will never
     crash or yield NaN.
   - Strict finite checking:
     All returned features pass through `assert_finite_features` ensuring no NaN/Inf.
"""

from collections.abc import Sequence

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

from ml.features.common import (
    DimensionMismatchError,
    EmptyMaskError,
    assert_finite_features,
)

# Canonical configuration constants
DEFAULT_LEVELS: int = 32
DEFAULT_DISTANCES: tuple[int, ...] = (1,)
DEFAULT_ANGLES: tuple[float, ...] = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
DEFAULT_LOG_BASE: float = 2.0
DEFAULT_SYMMETRIC: bool = True
DEFAULT_NORMED: bool = True


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or 2D image array to floating-point grayscale in [0.0, 1.0].

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3) or 2D grayscale image of shape (H, W).
        May be integer [0..255] or float [0..1].

    Returns
    -------
    np.ndarray
        2D float64 grayscale array with values in [0.0, 1.0].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected image to be a numpy ndarray, got {type(image).__name__}"
        )

    if image.ndim == 2:
        if np.issubdtype(image.dtype, np.floating):
            # If already floating point, check if bounded in [0, 1] or [0, 255]
            if np.nanmax(image) > 1.0:
                return np.clip(image / 255.0, 0.0, 1.0).astype(np.float64)
            return np.clip(image, 0.0, 1.0).astype(np.float64)
        return (image.astype(np.float64) / 255.0).clip(0.0, 1.0)

    if image.ndim == 3 and image.shape[2] == 3:
        # skimage rgb2gray handles uint8 [0..255] and float [0..1] correctly
        # using Y = 0.2125 R + 0.7154 G + 0.0721 B or standard luminance
        return rgb2gray(image).astype(np.float64)

    raise DimensionMismatchError(
        f"Expected image array of shape (H, W) or (H, W, 3), got shape {image.shape}"
    )


def quantize_gray(gray: np.ndarray, levels: int = DEFAULT_LEVELS) -> np.ndarray:
    """Uniformly quantize a 2D grayscale array into discrete levels in [0, levels - 1].

    Parameters
    ----------
    gray : np.ndarray
        2D grayscale array with values in [0.0, 1.0] (float) or [0, 255] (int).
    levels : int
        Number of discrete quantization bins (default: 32).

    Returns
    -------
    np.ndarray
        2D uint8 array with integer values in [0, levels - 1].
    """
    if levels < 2 or levels > 256:
        raise ValueError(f"levels must be in range [2, 256], got {levels}")

    if np.issubdtype(gray.dtype, np.floating):
        max_val = np.nanmax(gray) if gray.size > 0 else 0.0
        scale = levels if max_val <= 1.0 + 1e-6 else (levels / 256.0)
        quantized = np.floor(gray * scale).astype(np.int64)
    else:
        # Integer values [0, 255]
        bin_width = 256.0 / levels
        quantized = np.floor(gray.astype(np.float64) / bin_width).astype(np.int64)

    return np.clip(quantized, 0, levels - 1).astype(np.uint8)


def compute_glcm(
    image: np.ndarray,
    levels: int = DEFAULT_LEVELS,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    angles: Sequence[float] = DEFAULT_ANGLES,
    symmetric: bool = DEFAULT_SYMMETRIC,
    normed: bool = DEFAULT_NORMED,
) -> np.ndarray:
    """Compute the Gray-Level Co-occurrence Matrix (GLCM).

    Parameters
    ----------
    image : np.ndarray
        Grayscale quantized 2D uint8 image of shape (H, W) with values in [0, levels - 1].
    levels : int
        Number of gray levels (default: 32).
    distances : Sequence[int]
        Pixel pair displacement distances (default: (1,)).
    angles : Sequence[float]
        Pixel pair angles in radians (default: [0, pi/4, pi/2, 3*pi/4]).
    symmetric : bool
        If True, matrix is symmetric P(i, j) = P(j, i) (default: True).
    normed : bool
        If True, each 2D matrix sums to 1.0 (default: True).

    Returns
    -------
    np.ndarray
        4D GLCM array of shape (levels, levels, len(distances), len(angles)).
    """
    if image.ndim != 2:
        raise DimensionMismatchError(
            f"GLCM computation requires 2D array, got shape {image.shape}"
        )

    # Ensure integer array within [0, levels - 1]
    img_uint8 = np.clip(image, 0, levels - 1).astype(np.uint8)

    return graycomatrix(
        img_uint8,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )


def compute_glcm_contrast(glcm: np.ndarray) -> float:
    """Calculate mean GLCM contrast averaged across all distances and angles.

    Contrast = sum_{i, j} |i - j|^2 * P(i, j).

    Parameters
    ----------
    glcm : np.ndarray
        4D GLCM array of shape (levels, levels, distances, angles).

    Returns
    -------
    float
        Averaged contrast value.
    """
    contrast_matrix = graycoprops(glcm, "contrast")
    return float(np.mean(contrast_matrix))


def compute_glcm_entropy(glcm: np.ndarray, log_base: float = DEFAULT_LOG_BASE) -> float:
    """Calculate mean GLCM Shannon entropy averaged across all distances and angles.

    Entropy = - sum_{i, j, P>0} P(i, j) * log_{log_base}(P(i, j)).
    For log_base = 2.0, entropy is measured in bits (shannons).

    Parameters
    ----------
    glcm : np.ndarray
        4D normalized GLCM array of shape (levels, levels, distances, angles).
    log_base : float
        Logarithm base for entropy computation (default: 2.0).

    Returns
    -------
    float
        Averaged entropy value in bits (non-negative).
    """
    if log_base <= 0.0 or log_base == 1.0:
        raise ValueError(f"log_base must be positive and != 1.0, got {log_base}")

    n_dist = glcm.shape[2]
    n_ang = glcm.shape[3]
    entropies = []

    log_factor = np.log(log_base)

    for d in range(n_dist):
        for a in range(n_ang):
            p = glcm[:, :, d, a]
            # Select non-zero entries to avoid log(0)
            nz = p[p > 0.0]
            if len(nz) == 0:
                entropies.append(0.0)
            else:
                # - sum(p * log(p) / log(base))
                ent = -np.sum(nz * (np.log(nz) / log_factor))
                # Protect against negative zero
                entropies.append(max(0.0, float(ent)))

    return float(np.mean(entropies))


def extract_glcm_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    levels: int = DEFAULT_LEVELS,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    angles: Sequence[float] = DEFAULT_ANGLES,
    log_base: float = DEFAULT_LOG_BASE,
) -> dict[str, float]:
    """Extract GLCM contrast and entropy texture features.

    Parameters
    ----------
    image : np.ndarray
        Input image, either RGB (H, W, 3) or grayscale (H, W).
    mask : Optional[np.ndarray]
        Optional binary mask (H, W). If provided, crops to the mask bounding box.
        If mask is empty (all False/0), raises `EmptyMaskError`.
    levels : int
        Number of gray quantization levels (default: 32).
    distances : Sequence[int]
        Pixel pair distances (default: (1,)).
    angles : Sequence[float]
        Pixel pair angles in radians (default: [0, pi/4, pi/2, 3*pi/4]).
    log_base : float
        Base of logarithm for entropy calculation (default: 2.0).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - "glcm_contrast": mean contrast across 4 angles
        - "glcm_entropy": mean entropy across 4 angles (in bits)
    """
    # Validate mask if provided
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"Expected mask to be a numpy ndarray, got {type(mask).__name__}"
            )
        if mask.shape[:2] != image.shape[:2]:
            raise DimensionMismatchError(
                f"Mask shape {mask.shape[:2]} does not match image shape {image.shape[:2]}"
            )

        binary_mask = (
            mask.astype(bool) if mask.ndim == 2 else mask[:, :, 0].astype(bool)
        )
        if not np.any(binary_mask):
            raise EmptyMaskError(
                "Provided mask is completely empty (no foreground pixels)."
            )

        # Crop image to foreground bounding box
        rows, cols = np.where(binary_mask)
        min_r, max_r = int(rows.min()), int(rows.max())
        min_c, max_c = int(cols.min()), int(cols.max())
        image = image[min_r : max_r + 1, min_c : max_c + 1]

    # Convert to grayscale in [0.0, 1.0]
    gray = to_grayscale(image)

    # Quantize to specified levels [0, levels - 1]
    quantized = quantize_gray(gray, levels=levels)

    # Compute GLCM
    glcm = compute_glcm(
        quantized,
        levels=levels,
        distances=distances,
        angles=angles,
        symmetric=DEFAULT_SYMMETRIC,
        normed=DEFAULT_NORMED,
    )

    # Compute contrast and entropy
    contrast = compute_glcm_contrast(glcm)
    entropy = compute_glcm_entropy(glcm, log_base=log_base)

    features = {
        "glcm_contrast": contrast,
        "glcm_entropy": entropy,
    }

    # Strict NaN/Inf check (raises NonFiniteFeatureError on NaN/Inf)
    assert_finite_features(features, ["glcm_contrast", "glcm_entropy"])

    return features
