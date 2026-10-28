#!/usr/bin/env python3
"""Verification script for T5-002: Shared feature-extraction utilities.

Acceptance Criteria:
1. Loader documents channel order (docstring explicitly specifies RGB and conversion from BGR).
2. NaN/Inf raises a typed error (NonFiniteFeatureError is raised, never replaced by 0).
3. Schema version is attached to every vector.

Usage:
    python scripts/verify_feature_common.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.common import (
    DEFAULT_FEATURE_SCHEMA_VERSION,
    DimensionMismatchError,
    EmptyMaskError,
    FeatureVector,
    ImageLoadError,
    NonFiniteFeatureError,
    assert_finite_features,
    load_image,
    select_masked_pixels,
)


def verify_loader_documents_channel_order() -> tuple[bool, str]:
    """Criterion 1: Loader documents channel order and loads RGB."""
    # 1. Check docstring
    doc = inspect.getdoc(load_image)
    if not doc:
        return False, "load_image is missing a docstring."

    if "RGB" not in doc or "Channel Order" not in doc:
        return False, f"load_image docstring does not explicitly document channel order: {doc[:200]}"

    # 2. Check channel order behavior with a known BGR image
    # Synthetic BGR: Pure Blue is [255, 0, 0] in BGR; in RGB it should be [0, 0, 255]
    bgr_synthetic = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_synthetic[:, :, 0] = 255  # Blue channel in BGR

    import cv2
    _, encoded = cv2.imencode(".png", bgr_synthetic)
    loaded_rgb = load_image(encoded.tobytes())

    if loaded_rgb.shape != (10, 10, 3):
        return False, f"Expected shape (10, 10, 3), got {loaded_rgb.shape}"

    # In RGB, channel 0 is Red (should be 0) and channel 2 is Blue (should be 255)
    if loaded_rgb[0, 0, 0] != 0 or loaded_rgb[0, 0, 2] != 255:
        return False, (
            f"Channel order inversion failed! Expected RGB [0, 0, 255], got {loaded_rgb[0, 0].tolist()}"
        )

    return True, "load_image explicitly documents RGB channel order and correctly converts BGR input to RGB."


def verify_nan_inf_raises_typed_error() -> tuple[bool, str]:
    """Criterion 2: NaN/Inf raises a typed error (NonFiniteFeatureError)."""
    # Case A: NaN in dict
    try:
        assert_finite_features({"exg_mean": float("nan"), "exr_mean": 0.5})
        return False, "Failed to raise NonFiniteFeatureError for NaN feature."
    except NonFiniteFeatureError as exc:
        if exc.feature_name != "exg_mean":
            return False, f"Wrong feature_name in error: {exc.feature_name}"
    except Exception as exc:
        return False, f"Raised wrong exception type: {type(exc).__name__}"

    # Case B: +Inf in list
    try:
        assert_finite_features([1.2, float("inf"), 3.4], ["f1", "f2", "f3"])
        return False, "Failed to raise NonFiniteFeatureError for +Inf feature."
    except NonFiniteFeatureError as exc:
        if exc.feature_name != "f2":
            return False, f"Wrong feature_name in error: {exc.feature_name}"

    # Case C: -Inf in FeatureVector constructor
    try:
        FeatureVector(
            values=[0.1, float("-inf")],
            feature_names=["f1", "f2"],
            schema_version="1.0.0",
        )
        return False, "FeatureVector constructor failed to raise NonFiniteFeatureError for -Inf."
    except NonFiniteFeatureError as exc:
        if exc.feature_name != "f2":
            return False, f"Wrong feature_name in error: {exc.feature_name}"

    return True, "NonFiniteFeatureError correctly raised for NaN, +Inf, and -Inf; failed features are never zeroed."


def verify_schema_version_attached_to_vector() -> tuple[bool, str]:
    """Criterion 3: Schema version is attached to every vector."""
    vec = FeatureVector(
        values=[1.25, 0.88, 0.42],
        feature_names=["exg_mean", "exr_mean", "glcm_contrast"],
        schema_version="1.0.0",
        crop="lettuce",
        extraction_config={"hsv_hue_range": [20, 35]},
    )

    if not hasattr(vec, "schema_version") or vec.schema_version != "1.0.0":
        return False, f"Missing or incorrect schema_version attribute: {getattr(vec, 'schema_version', None)}"

    # Check serialization
    json_repr = vec.to_json()
    if '"schema_version": "1.0.0"' not in json_repr:
        return False, f"schema_version missing from JSON representation: {json_repr}"

    # Check empty schema_version rejection
    try:
        FeatureVector(values=[1.0], feature_names=["f1"], schema_version="")
        return False, "FeatureVector failed to reject empty schema_version string."
    except ValueError:
        pass

    return True, f"schema_version '{vec.schema_version}' is attached to every FeatureVector instance and preserved in JSON."


def main() -> int:
    checks = [
        ("Loader documents channel order (RGB)", verify_loader_documents_channel_order),
        ("NaN/Inf raises a typed error (NonFiniteFeatureError)", verify_nan_inf_raises_typed_error),
        ("Schema version is attached to every vector", verify_schema_version_attached_to_vector),
    ]

    print("================================================================================")
    print("VERIFY SHARED FEATURE-EXTRACTION UTILITIES (T5-002)")
    print("================================================================================\n")

    all_passed = True
    for title, check_fn in checks:
        ok, msg = check_fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {title}\n       {msg}")
        if not ok:
            all_passed = False

    print("\n--------------------------------------------------------------------------------")
    if all_passed:
        print("All T5-002 acceptance criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
