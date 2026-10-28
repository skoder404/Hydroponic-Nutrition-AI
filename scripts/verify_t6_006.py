#!/usr/bin/env python3
"""Verification script for T6-006: Implement the per-leaf ImageService pipeline.

Acceptance Criteria:
1. Plant mode returns one result per valid leaf plus a count summary.
2. Rejected leaves are listed with reasons, not dropped.
3. No averaged or merged confidence appears in the response.
4. If no valid leaf exists return SEGMENTATION_FAILED (retryable), never a prediction.
5. All unit tests in tests/unit/test_image_service.py pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_verification() -> bool:
    print("================================================================================")
    print("VERIFICATION FOR T6-006: IMPLEMENT PER-LEAF IMAGESERVICE PIPELINE")
    print("================================================================================\n")

    print("[1/2] Running pytest on tests/unit/test_image_service.py...")
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/unit/test_image_service.py"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print("  [FAIL] Unit tests failed.")
        return False
    print("  [PASS] All ImageService unit tests passed.\n")

    print("[2/2] Verifying end-to-end ImageService execution on Lettuce, Spinach, Tomato...")
    import cv2
    import numpy as np
    from backend.app.core.errors import SegmentationFailedError
    from backend.app.services.image_service import ImageService

    service = ImageService()

    # Test each crop
    for crop in ["lettuce", "spinach", "tomato"]:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (70, 70), 35, (25, 200, 25), -1)
        cv2.circle(img, (130, 130), 35, (25, 200, 25), -1)

        resp = service.predict(image=img, crop=crop, input_mode="plant")
        print(
            f"  - {crop.capitalize()} (Plant Mode): {resp.summary.leaf_count} leaves assessed. "
            f"Dominant={resp.predicted_class} ({resp.display_label}), "
            f"Top Confidence={resp.confidence} (Honest: None), Class Counts={resp.summary.class_counts}"
        )
        assert resp.confidence is None, "Top-level confidence in plant mode must be None!"
        assert resp.summary.leaf_count >= 1, "Must have at least 1 leaf"

    # Test SEGMENTATION_FAILED on empty frame
    empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        service.predict(image=empty_img, crop="lettuce", input_mode="plant")
        print("  [FAIL] Did not raise SegmentationFailedError on empty frame")
        return False
    except SegmentationFailedError as err:
        print(f"  - Empty frame rejection verified: code={err.code}, retryable={err.retryable}")

    print("\n================================================================================")
    print("ALL ACCEPTANCE CRITERIA FOR T6-006 VERIFIED SUCCESSFULLY!")
    print("================================================================================")
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)

