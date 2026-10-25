#!/usr/bin/env python3
"""Verification script for SAM 2.1 Segmenter Wrapper (T5-008).

Acceptance Criteria:
1. Segmenter returns mask plus config/version string.
2. Simulated OOM triggers the documented fallback or a typed error.
3. Model is not loaded until first use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.segmentation.sam2_segmenter import (
    CudaOOMError,
    SAM2Segmenter,
    SegmentationResult,
)


def get_test_image_path() -> Path:
    """Return a path to a representative sample image."""
    img_path = PROJECT_ROOT / "image_data" / "Lettuce_Images" / "Healthier" / "Healthier_0001.jpg"
    if not img_path.exists():
        # Fallback to any image file
        for p in (PROJECT_ROOT / "image_data").rglob("*.jpg"):
            return p
    return img_path


def verify_lazy_loading() -> tuple[bool, str]:
    """Criterion 3: Model is not loaded until first use."""
    segmenter = SAM2Segmenter()

    if segmenter.is_loaded:
        return False, "Model is loaded prematurely upon instantiation."
    if segmenter.loaded_device is not None:
        return False, f"Loaded device should be None prior to first use, got '{segmenter.loaded_device}'."

    # Perform segmentation to trigger first use
    test_img = get_test_image_path()
    res = segmenter.segment(test_img)

    if not segmenter.is_loaded:
        return False, "Model failed to load on first use."
    if segmenter.loaded_device is None:
        return False, "Loaded device remains None after first use."

    # Unload and verify memory cleanup
    segmenter.unload()
    if segmenter.is_loaded:
        return False, "Model remains loaded after explicit unload()."

    return True, f"Verified model is not loaded until first use (lazy-load) and unloads cleanly on {res.device_used}."


def verify_mask_and_config_version() -> tuple[bool, str]:
    """Criterion 1: Segmenter returns mask plus config/version string."""
    segmenter = SAM2Segmenter()
    test_img = get_test_image_path()

    result: SegmentationResult = segmenter.segment(test_img)

    # 1. Mask verification
    if not isinstance(result.mask, np.ndarray):
        return False, f"Mask is not a numpy array: {type(result.mask)}"
    if result.mask.dtype != bool:
        return False, f"Mask dtype should be bool, got {result.mask.dtype}"
    if result.mask.ndim != 2:
        return False, f"Mask should be 2D array (H, W), got {result.mask.ndim}D"
    if not np.any(result.mask):
        return False, "Mask is completely empty (all False)."

    # 2. Config/version string verification
    if not result.config_version or not isinstance(result.config_version, str):
        return False, f"Missing or invalid config_version: {result.config_version}"
    if "sam2.1" not in result.config_version:
        return False, f"config_version string '{result.config_version}' does not identify SAM 2.1"

    segmenter.unload()
    return True, f"Returned 2D boolean mask ({result.mask.shape}) with score={result.score} and config_version='{result.config_version}'."


def verify_oom_handling_and_fallback() -> tuple[bool, str]:
    """Criterion 2: Simulated OOM triggers documented fallback or a typed error."""
    test_img = get_test_image_path()

    # Part A: With fallback_on_oom=True -> Should trigger CPU fallback and return mask
    segmenter_fallback = SAM2Segmenter(fallback_on_oom=True)
    res_fallback = segmenter_fallback.segment(test_img, simulate_oom=True)

    if not res_fallback.fallback_triggered:
        return False, "fallback_triggered flag was not set to True during simulated OOM."
    if res_fallback.device_used != "cpu":
        return False, f"Device used during fallback should be 'cpu', got '{res_fallback.device_used}'."
    if not isinstance(res_fallback.mask, np.ndarray) or not np.any(res_fallback.mask):
        return False, "Fallback CPU inference failed to produce valid mask."
    segmenter_fallback.unload()

    # Part B: With fallback_on_oom=False -> Should raise typed CudaOOMError
    segmenter_strict = SAM2Segmenter(fallback_on_oom=False)
    raised_typed_error = False
    try:
        segmenter_strict.segment(test_img, simulate_oom=True)
    except CudaOOMError as exc:
        raised_typed_error = True
        if "out-of-memory" not in str(exc).lower() and "oom" not in str(exc).lower():
            return False, f"CudaOOMError missing descriptive message: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Expected typed CudaOOMError, but received: {type(exc).__name__}: {exc}"
    finally:
        segmenter_strict.unload()

    if not raised_typed_error:
        return False, "Failed to raise typed CudaOOMError when fallback_on_oom=False."

    return True, "Simulated OOM successfully triggers documented CPU fallback (when enabled) and raises typed CudaOOMError (when disabled)."


def main() -> int:
    checks = [
        ("Model is not loaded until first use", verify_lazy_loading),
        ("Segmenter returns mask plus config/version string", verify_mask_and_config_version),
        ("Simulated OOM triggers documented fallback or typed error", verify_oom_handling_and_fallback),
    ]

    print("================================================================================")
    print("VERIFY SAM 2.1 SEGMENTER WRAPPER (T5-008)")
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
        print("All criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
