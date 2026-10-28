#!/usr/bin/env python3
"""Verification script for T5-004: Implement lettuce feature extractor (7 features).

Acceptance Criteria:
1. Order and names match Overview §16.1:
   ("exg_mean", "exr_mean", "yellow_pixel_ratio", "brown_pixel_ratio", "mean_hue", "mean_saturation", "contour_compactness")
2. Zero perimeter produces an explicit error:
   Zero / non-positive perimeter raises ZeroPerimeterError (subclass of FeatureExtractionError).
3. HSV thresholds are stored in config, not hard-coded:
   LettuceFeatureConfig stores yellow and brown HSV thresholds, can be customized, and is persisted in FeatureVector.extraction_config.

Usage:
    python scripts/verify_lettuce_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.common import EmptyMaskError, FeatureVector, ZeroPerimeterError  # noqa: E402
from ml.features.lettuce import (  # noqa: E402
    DEFAULT_LETTUCE_CONFIG,
    LETTUCE_FEATURE_NAMES,
    LettuceFeatureConfig,
    compute_contour_compactness,
    extract_lettuce_features,
)


def verify_order_and_names() -> tuple[bool, str]:
    """Criterion 1: Order and names match Overview §16.1."""
    expected_order = (
        "exg_mean",
        "exr_mean",
        "yellow_pixel_ratio",
        "brown_pixel_ratio",
        "mean_hue",
        "mean_saturation",
        "contour_compactness",
    )

    if LETTUCE_FEATURE_NAMES != expected_order:
        return (
            False,
            f"LETTUCE_FEATURE_NAMES mismatch: expected {expected_order}, got {LETTUCE_FEATURE_NAMES}",
        )

    # Create synthetic leaf image and circular mask
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [35, 180, 55]  # Healthy green

    # Circular leaf mask
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
    mask = dist_from_center <= 30

    vec = extract_lettuce_features(img, mask)

    if not isinstance(vec, FeatureVector):
        return False, f"Expected FeatureVector instance, got {type(vec).__name__}"

    if tuple(vec.feature_names) != expected_order:
        return (
            False,
            f"Feature names order mismatch: expected {expected_order}, got {tuple(vec.feature_names)}",
        )

    if len(vec.values) != 7:
        return False, f"Expected 7 feature values, got {len(vec.values)}"

    as_dict = vec.to_dict()
    if list(as_dict.keys()) != list(expected_order):
        return False, f"FeatureVector.to_dict() keys mismatch: {list(as_dict.keys())}"

    # For a circular mask of radius 30:
    # Area ~ pi * 30^2 ~ 2827, Perimeter ~ 2 * pi * 30 ~ 188.5 -> compactness ~ 1.0
    compactness = as_dict["contour_compactness"]
    if not (0.85 <= compactness <= 1.05):
        return (
            False,
            f"Expected circular leaf compactness close to 1.0, got {compactness:.4f}",
        )

    return True, (
        f"Verified 7-feature ordering and names match Overview §16.1: {', '.join(expected_order)}"
    )


def verify_zero_perimeter_handling() -> tuple[bool, str]:
    """Criterion 2: Zero perimeter produces an explicit error."""
    # 1. Single isolated pixel mask has perimeter 0.0 in cv2.arcLength
    mask_single_pixel = np.zeros((50, 50), dtype=bool)
    mask_single_pixel[25, 25] = True

    try:
        compute_contour_compactness(mask_single_pixel)
        return (
            False,
            "Expected ZeroPerimeterError on single-pixel mask with zero perimeter, but none was raised.",
        )
    except ZeroPerimeterError as e:
        zero_err_msg = str(e)
    except Exception as e:  # noqa: BLE001
        return (
            False,
            f"Expected ZeroPerimeterError, but got unexpected {type(e).__name__}: {e}",
        )

    # Also test through extract_lettuce_features
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    try:
        extract_lettuce_features(img, mask_single_pixel)
        return (
            False,
            "Expected ZeroPerimeterError from extract_lettuce_features on zero-perimeter mask.",
        )
    except ZeroPerimeterError:
        pass
    except Exception as e:  # noqa: BLE001
        return (
            False,
            f"Expected ZeroPerimeterError from extract_lettuce_features, got {type(e).__name__}: {e}",
        )

    # 2. Empty mask raises EmptyMaskError
    empty_mask = np.zeros((50, 50), dtype=bool)
    try:
        extract_lettuce_features(img, empty_mask)
        return False, "Expected EmptyMaskError on empty mask, but none was raised."
    except EmptyMaskError:
        pass
    except Exception as e:  # noqa: BLE001
        return False, f"Expected EmptyMaskError, got {type(e).__name__}: {e}"

    return True, (
        f"Verified zero perimeter produces explicit ZeroPerimeterError: '{zero_err_msg}'"
    )


def verify_hsv_thresholds_in_config() -> tuple[bool, str]:
    """Criterion 3: HSV thresholds are stored in config, not hard-coded."""
    # 1. Check default config properties
    cfg = DEFAULT_LETTUCE_CONFIG
    for attr in [
        "yellow_hue_min",
        "yellow_hue_max",
        "yellow_sat_min",
        "yellow_sat_max",
        "yellow_val_min",
        "yellow_val_max",
        "brown_hue_min",
        "brown_hue_max",
        "brown_sat_min",
        "brown_sat_max",
        "brown_val_min",
        "brown_val_max",
    ]:
        if not hasattr(cfg, attr):
            return False, f"LettuceFeatureConfig missing attribute '{attr}'."

    # 2. Check metadata persistence in extraction_config
    h, w = 60, 60
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Fill with chlorotic yellow: OpenCV HSV ~ (30, 200, 200) -> RGB
    img[:, :] = [215, 215, 30]
    mask = np.ones((h, w), dtype=bool)

    vec_default = extract_lettuce_features(img, mask, config=cfg)
    extraction_config = vec_default.extraction_config

    assert "yellow_hue_min" in extraction_config, (
        "yellow_hue_min not found in extraction_config"
    )
    assert "brown_hue_min" in extraction_config, (
        "brown_hue_min not found in extraction_config"
    )
    assert extraction_config["crop"] == "lettuce"

    default_yellow_ratio = vec_default.to_dict()["yellow_pixel_ratio"]
    if default_yellow_ratio <= 0.5:
        return (
            False,
            f"Expected yellow pixels to be detected with default config, got ratio {default_yellow_ratio}",
        )

    # 3. Verify custom config alters extraction behavior (not hardcoded)
    custom_cfg = LettuceFeatureConfig(
        yellow_hue_min=100.0,  # Narrow/shift away from yellow
        yellow_hue_max=110.0,
    )
    vec_custom = extract_lettuce_features(img, mask, config=custom_cfg)
    custom_yellow_ratio = vec_custom.to_dict()["yellow_pixel_ratio"]

    if custom_yellow_ratio != 0.0:
        return (
            False,
            f"Expected custom HSV config to alter detection to 0.0, got {custom_yellow_ratio}",
        )

    return True, (
        f"Verified HSV thresholds are configurable and persisted in metadata: "
        f"default={default_yellow_ratio:.2f}, custom={custom_yellow_ratio:.2f}"
    )


def run_verification() -> bool:
    checks = [
        ("Criterion 1: Order and names match Overview §16.1", verify_order_and_names),
        (
            "Criterion 2: Zero perimeter produces explicit error",
            verify_zero_perimeter_handling,
        ),
        (
            "Criterion 3: HSV thresholds are stored in config, not hard-coded",
            verify_hsv_thresholds_in_config,
        ),
    ]

    all_passed = True
    print("=" * 70)
    print("Verification for T5-004: Lettuce feature extractor (7 features)")
    print("=" * 70)

    for name, check_fn in checks:
        passed, msg = check_fn()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
