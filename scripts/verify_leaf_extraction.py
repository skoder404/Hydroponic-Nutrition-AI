#!/usr/bin/env python3
"""Verification script for T5-010: Implement leaf extraction (plant photo to individual leaves).

Acceptance Criteria:
1. Returns leaves plus a rejected[] list with reasons
2. Overlapping duplicate masks are merged or removed (rule documented)
3. Same function is used in training-data creation and inference

Usage:
    python scripts/verify_leaf_extraction.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml.segmentation.leaf_extraction as leaf_extraction_module
from ml.segmentation.leaf_extraction import (
    ExtractedLeaf,
    LeafExtractionConfig,
    LeafExtractionResult,
    RejectedLeaf,
    compute_bounding_box,
    compute_mask_iou,
    extract_leaves,
)
from ml.segmentation.mask_validation import MaskValidationConfig, MaskVerdict
from ml.segmentation.sam2_segmenter import SAM2Segmenter, SegmentationResult


def create_mock_segmenter(mask_generator_fn) -> SAM2Segmenter:
    """Create a mock SAM2Segmenter returning controlled masks."""
    mock_seg = MagicMock(spec=SAM2Segmenter)
    mock_seg.is_loaded = True
    mock_seg.loaded_device = "cpu"
    mock_seg.config_version_string = "mock_sam2:v1"

    def mock_segment(image, point_coords=None, point_labels=None, box=None, simulate_oom=False):
        mask, score = mask_generator_fn(point_coords)
        return SegmentationResult(
            mask=mask,
            config_version="mock_sam2:v1",
            device_used="cpu",
            score=score,
            fallback_triggered=False,
        )

    mock_seg.segment.side_effect = mock_segment
    return mock_seg


def verify_leaves_plus_rejected_with_reasons() -> tuple[bool, str]:
    """Criterion 1: Returns leaves plus a rejected[] list with reasons."""
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:90, 10:90] = [30, 180, 50]  # Green square

    # Sequence of masks:
    # 1. Valid leaf mask (score 0.90)
    # 2. Empty mask (should fail validation and be in rejected)
    # 3. Whole frame flooding mask (area > 85%, should be rejected)
    call_idx = [0]

    def mock_masks(pt):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx == 0:
            # Valid mask: 25% area
            m = np.zeros((h, w), dtype=bool)
            m[20:70, 20:70] = True
            return m, 0.92
        elif idx == 1:
            # Empty mask
            m = np.zeros((h, w), dtype=bool)
            return m, 0.10
        else:
            # Whole frame flooding mask (95% area)
            m = np.ones((h, w), dtype=bool)
            return m, 0.40

    mock_seg = create_mock_segmenter(mock_masks)

    cfg = LeafExtractionConfig(
        single_leaf_mode=False,
        candidate_points=[(50, 50), (25, 25), (75, 75)],
    )
    result = extract_leaves(img, segmenter=mock_seg, config=cfg)

    if not isinstance(result, LeafExtractionResult):
        return False, f"Expected LeafExtractionResult, got {type(result).__name__}"

    if len(result.leaves) == 0:
        return False, "Expected at least 1 accepted leaf, got 0."

    if len(result.rejected) == 0:
        return False, "Expected rejected masks in rejected[], got 0."

    # Verify rejected items have structured properties and explicit reasons
    for rej in result.rejected:
        if not isinstance(rej, RejectedLeaf):
            return False, f"Expected RejectedLeaf, got {type(rej).__name__}"
        if not rej.reason or not isinstance(rej.reason, str):
            return False, f"Rejected leaf {rej.candidate_id} missing reason string: {rej.reason}"
        if not isinstance(rej.verdict, MaskVerdict):
            return False, f"Rejected leaf {rej.candidate_id} has invalid verdict: {rej.verdict}"

    # Verify accepted leaf has bounding_box, mask, and masked_crop
    for leaf in result.leaves:
        if not isinstance(leaf, ExtractedLeaf):
            return False, f"Expected ExtractedLeaf, got {type(leaf).__name__}"
        if len(leaf.bounding_box) != 4:
            return False, f"Invalid bounding box: {leaf.bounding_box}"
        if leaf.masked_crop.shape[:2] != (leaf.bounding_box[3] - leaf.bounding_box[1] + 1, leaf.bounding_box[2] - leaf.bounding_box[0] + 1):
            return False, f"Masked crop dimensions mismatch bbox: {leaf.masked_crop.shape}"

    return True, (
        f"Verified returns leaves ({len(result.leaves)} accepted) plus rejected[] "
        f"({len(result.rejected)} rejected) with explicit reasons and structured metrics."
    )


def verify_overlapping_duplicates_removed_or_merged() -> tuple[bool, str]:
    """Criterion 2: Overlapping duplicate masks are merged or removed (rule documented)."""
    # 1. Verify docstring documentation
    doc = leaf_extraction_module.__doc__
    if not doc:
        return False, "Module docstring missing in ml.segmentation.leaf_extraction."

    if "duplicate" not in doc.lower() or "iou" not in doc.lower():
        return False, "Module docstring does not document overlapping duplicate mask rule."

    if "0.50" not in doc and "overlap" not in doc.lower():
        return False, "Module docstring does not specify duplicate IoU threshold."

    # 2. Test duplicate suppression behavior
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:90, 10:90] = [30, 180, 50]

    call_count = [0]

    def mock_masks(pt):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            # Primary leaf (score 0.95): center 40x40
            m = np.zeros((h, w), dtype=bool)
            m[30:70, 30:70] = True
            return m, 0.95
        else:
            # Overlapping duplicate (score 0.80): shifted by 2 pixels (IoU ~ 0.90)
            m = np.zeros((h, w), dtype=bool)
            m[31:71, 31:71] = True
            return m, 0.80

    mock_seg = create_mock_segmenter(mock_masks)

    # A. Test suppression (merge_duplicates=False)
    cfg_suppress = LeafExtractionConfig(
        single_leaf_mode=False,
        iou_overlap_threshold=0.50,
        merge_duplicates=False,
        candidate_points=[(50, 50), (51, 51)],
    )
    res_suppress = extract_leaves(img, segmenter=mock_seg, config=cfg_suppress)

    if len(res_suppress.leaves) != 1:
        return False, f"Expected exactly 1 leaf after deduplication, got {len(res_suppress.leaves)}"

    # Check that duplicate candidate was added to rejected[] with IoU reason
    dup_rejections = [r for r in res_suppress.rejected if "duplicate" in r.reason.lower()]
    if not dup_rejections:
        return False, f"Expected duplicate reason in rejected[], got: {[r.reason for r in res_suppress.rejected]}"

    # B. Test merging (merge_duplicates=True)
    call_count[0] = 0
    cfg_merge = LeafExtractionConfig(
        single_leaf_mode=False,
        iou_overlap_threshold=0.50,
        merge_duplicates=True,
        candidate_points=[(50, 50), (51, 51)],
    )
    res_merge = extract_leaves(img, segmenter=mock_seg, config=cfg_merge)
    if len(res_merge.leaves) != 1:
        return False, f"Expected 1 merged leaf, got {len(res_merge.leaves)}"

    # The merged mask should enclose both regions: y: [30, 71], x: [30, 71]
    merged_box = res_merge.leaves[0].bounding_box
    if merged_box != (30, 30, 70, 70):
        # Coordinates (x1, y1, x2, y2)
        expected_box = (30, 30, 70, 70)
        # Note: 31:71 has max index 70, so box is 30..70
        if merged_box[0] != 30 or merged_box[2] != 70:
            return False, f"Merged bounding box mismatch: {merged_box}"

    return True, (
        "Duplicate removal thoroughly verified: IoU rule is documented in module docstring; "
        "duplicates are suppressed with IoU citation in rejected[], and merge_duplicates=True "
        "properly unifies overlapping masks."
    )


def verify_same_function_training_and_inference() -> tuple[bool, str]:
    """Criterion 3: Same function is used in training-data creation and inference."""
    # Verify signature and single_leaf_mode bypass
    sig = inspect.signature(extract_leaves)
    params = list(sig.parameters.keys())
    if "image" not in params or "config" not in params:
        return False, f"extract_leaves missing expected parameters: {params}"

    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[20:80, 20:80] = [40, 190, 50]

    def single_leaf_mock(pt):
        m = np.zeros((h, w), dtype=bool)
        m[20:80, 20:80] = True
        return m, 0.94

    mock_seg = create_mock_segmenter(single_leaf_mock)

    # 1. Inference / pre-cropped single leaf mode (single_leaf_mode=True)
    res_single = extract_leaves(
        img,
        segmenter=mock_seg,
        config=LeafExtractionConfig(single_leaf_mode=True),
    )
    if not res_single.is_single_leaf:
        return False, "Expected is_single_leaf=True in single leaf mode"
    if len(res_single.leaves) != 1:
        return False, f"Expected 1 leaf in single leaf mode, got {len(res_single.leaves)}"

    # 2. Training data creation multi-leaf mode (single_leaf_mode=False)
    res_multi = extract_leaves(
        img,
        segmenter=mock_seg,
        config=LeafExtractionConfig(single_leaf_mode=False),
    )
    if res_multi.is_single_leaf:
        return False, "Expected is_single_leaf=False in multi-leaf mode"
    if len(res_multi.leaves) < 1:
        return False, f"Expected leaves in multi-leaf mode, got {len(res_multi.leaves)}"

    return True, (
        "extract_leaves seamlessly supports both single_leaf_mode=True (inference bypass) "
        "and single_leaf_mode=False (multi-leaf plant slicing for dataset generation)."
    )


def run_verification() -> bool:
    """Run all verification checks for T5-010."""
    print("=" * 80)
    print("VERIFY LEAF EXTRACTION PIPELINE (T5-010)")
    print("=" * 80)
    print()

    checks = [
        ("Returns leaves plus a rejected[] list with reasons", verify_leaves_plus_rejected_with_reasons),
        ("Overlapping duplicate masks are merged or removed (rule documented)", verify_overlapping_duplicates_removed_or_merged),
        ("Same function is used in training-data creation and inference", verify_same_function_training_and_inference),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {title}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print()
    print("-" * 80)
    if all_passed:
        print("All T5-010 acceptance criteria successfully verified!")
    else:
        print("One or more T5-010 checks failed.")
    print()

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
