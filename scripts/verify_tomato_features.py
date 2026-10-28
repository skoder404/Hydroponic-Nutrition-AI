#!/usr/bin/env python3
"""Verification script for T5-007: Implement tomato feature extractor (4 features).

Acceptance Criteria:
1. Feature list equals the ADR decision:
   ("glcm_contrast", "glcm_entropy", "bounding_aspect_ratio", "solidity") per ADR 0007.
2. Definitions are written in metadata:
   Exact definitions of bounding_aspect_ratio and solidity recorded in extraction_config.
3. Degenerate contour (zero hull area) raises an error:
   DegenerateContourError is raised when convex hull area is zero or non-positive.

Usage:
    python scripts/verify_tomato_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.common import (  # noqa: E402
    DegenerateContourError,
    EmptyMaskError,
    FeatureVector,
)
from ml.features.tomato import (  # noqa: E402
    DEFAULT_TOMATO_CONFIG,
    TOMATO_FEATURE_NAMES,
    compute_shape_descriptors,
    extract_tomato_features,
)


def verify_feature_list_equals_adr_decision() -> tuple[bool, str]:
    """Criterion 1: Feature list equals ADR 0007 decision."""
    expected_features = (
        "glcm_contrast",
        "glcm_entropy",
        "bounding_aspect_ratio",
        "solidity",
    )

    if TOMATO_FEATURE_NAMES != expected_features:
        return (
            False,
            f"Expected {expected_features}, got {TOMATO_FEATURE_NAMES}",
        )

    # Synthetic leaf image and elliptical mask
    h, w = 80, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [45, 175, 65]  # Tomato green

    y, x = np.ogrid[:h, :w]
    # Elliptical leaf: major radius 25, minor radius 15
    mask = (((x - 40) / 25) ** 2 + ((y - 40) / 15) ** 2) <= 1.0

    vec = extract_tomato_features(img, mask)

    if not isinstance(vec, FeatureVector):
        return False, f"Expected FeatureVector instance, got {type(vec).__name__}"

    if tuple(vec.feature_names) != expected_features:
        return (
            False,
            f"Feature names order mismatch: expected {expected_features}, got {tuple(vec.feature_names)}",
        )

    if len(vec.values) != 4:
        return False, f"Expected 4 feature values, got {len(vec.values)}"

    as_dict = vec.to_dict()
    if list(as_dict.keys()) != list(expected_features):
        return False, f"FeatureVector.to_dict() keys mismatch: {list(as_dict.keys())}"

    # For an ellipse, solidity (area / hull_area) is approximately 1.0 (convex shape)
    solidity = as_dict["solidity"]
    if not (0.95 <= solidity <= 1.05):
        return False, f"Expected convex ellipse solidity ~ 1.0, got {solidity:.4f}"

    return True, (
        f"Verified 4-feature list matches ADR 0007: {', '.join(expected_features)}"
    )


def verify_definitions_written_in_metadata() -> tuple[bool, str]:
    """Criterion 2: Definitions are written in metadata."""
    cfg = DEFAULT_TOMATO_CONFIG
    config_dict = cfg.to_dict()

    if "definitions" not in config_dict:
        return False, "Key 'definitions' missing from extraction_config."

    defs = config_dict["definitions"]
    if "bounding_aspect_ratio" not in defs:
        return False, "bounding_aspect_ratio definition missing from metadata."

    if "solidity" not in defs:
        return False, "solidity definition missing from metadata."

    # Verify extraction vector carries these definitions
    h, w = 60, 60
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 20:50] = True  # 20x30 rectangle -> aspect ratio = 30/20 = 1.5

    vec = extract_tomato_features(img, mask, config=cfg)
    ext_cfg = vec.extraction_config

    assert "definitions" in ext_cfg
    assert "droop_angle_decision" in ext_cfg
    assert "REMOVED" in ext_cfg["droop_angle_decision"]

    # Verify measured aspect ratio
    measured_ar = vec.to_dict()["bounding_aspect_ratio"]
    if not (1.4 <= measured_ar <= 1.6):
        return (
            False,
            f"Expected bounding_aspect_ratio ~ 1.5 for 30x20 rect, got {measured_ar:.3f}",
        )

    return True, (
        f"Verified exact definitions and ADR 0007 removal rationale recorded in metadata: "
        f"AR={defs['bounding_aspect_ratio'][:40]}..., Solidity={defs['solidity'][:40]}..."
    )


def verify_degenerate_contour_raises_error() -> tuple[bool, str]:
    """Criterion 3: Degenerate contour (zero hull area) raises an error."""
    # 1. Single isolated pixel mask -> convex hull area = 0.0
    single_pixel_mask = np.zeros((40, 40), dtype=bool)
    single_pixel_mask[20, 20] = True

    try:
        compute_shape_descriptors(single_pixel_mask)
        return (
            False,
            "Expected DegenerateContourError on single-pixel mask, but none was raised.",
        )
    except DegenerateContourError as e:
        degen_msg = str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"Expected DegenerateContourError, got {type(e).__name__}: {e}"

    # Also test via extract_tomato_features
    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    try:
        extract_tomato_features(img, single_pixel_mask)
        return (
            False,
            "Expected DegenerateContourError from extract_tomato_features on single pixel.",
        )
    except DegenerateContourError:
        pass
    except Exception as e:  # noqa: BLE001
        return (
            False,
            f"Expected DegenerateContourError from extract_tomato_features, got {type(e).__name__}: {e}",
        )

    # 2. Empty mask raises EmptyMaskError
    empty_mask = np.zeros((40, 40), dtype=bool)
    try:
        extract_tomato_features(img, empty_mask)
        return False, "Expected EmptyMaskError on empty mask, but none was raised."
    except EmptyMaskError:
        pass
    except Exception as e:  # noqa: BLE001
        return False, f"Expected EmptyMaskError, got {type(e).__name__}: {e}"

    return True, (
        f"Verified degenerate contour with zero hull area raises DegenerateContourError: '{degen_msg}'"
    )


def run_verification() -> bool:
    checks = [
        (
            "Criterion 1: Feature list equals ADR 0007 decision",
            verify_feature_list_equals_adr_decision,
        ),
        (
            "Criterion 2: Definitions are written in metadata",
            verify_definitions_written_in_metadata,
        ),
        (
            "Criterion 3: Degenerate contour raises error",
            verify_degenerate_contour_raises_error,
        ),
    ]

    all_passed = True
    print("=" * 70)
    print("Verification for T5-007: Tomato feature extractor (4 features)")
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
