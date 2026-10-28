"""Unit tests for leaf instance extraction pipeline (T5-010)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.common import DimensionMismatchError
from ml.segmentation.leaf_extraction import (
    ExtractedLeaf,
    LeafExtractionConfig,
    LeafExtractionResult,
    RejectedLeaf,
    compute_bounding_box,
    compute_mask_iou,
    create_masked_crop,
    extract_leaves,
    generate_candidate_proposals,
)
from ml.segmentation.mask_validation import MaskVerdict
from ml.segmentation.sam2_segmenter import SAM2Segmenter, SegmentationResult
from scripts.verify_leaf_extraction import create_mock_segmenter, run_verification


def test_leaf_extraction_verification_script():
    """Verify all T5-010 criteria pass via the verification script."""
    assert run_verification() is True


def test_compute_mask_iou():
    """Test IoU calculation between masks."""
    m1 = np.zeros((20, 20), dtype=bool)
    m1[5:15, 5:15] = True  # 10x10 = 100 px

    # Identical
    assert pytest.approx(compute_mask_iou(m1, m1), abs=1e-5) == 1.0

    # Non-overlapping
    m2 = np.zeros((20, 20), dtype=bool)
    m2[0:4, 0:4] = True
    assert compute_mask_iou(m1, m2) == 0.0

    # 50% overlap (overlap = 50, union = 150 -> IoU = 1/3)
    m3 = np.zeros((20, 20), dtype=bool)
    m3[5:15, 10:20] = True
    # intersection: [5:15, 10:15] = 10 * 5 = 50
    # union: 100 + 100 - 50 = 150
    assert pytest.approx(compute_mask_iou(m1, m3), abs=1e-5) == 50.0 / 150.0


def test_compute_bounding_box():
    """Test bounding box calculation."""
    empty_mask = np.zeros((30, 30), dtype=bool)
    assert compute_bounding_box(empty_mask) == (0, 0, 0, 0)

    rect_mask = np.zeros((50, 50), dtype=bool)
    rect_mask[10:25, 15:35] = True  # y in [10, 24], x in [15, 34]
    bbox = compute_bounding_box(rect_mask)
    assert bbox == (15, 10, 34, 24)  # (x1, y1, x2, y2)


def test_create_masked_crop():
    """Test masked crop extraction and background zeroing."""
    img = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True
    bbox = (10, 10, 19, 19)

    crop_mask, rgb_crop, rgba_crop = create_masked_crop(img, mask, bbox, background_fill=(0, 0, 0))

    assert crop_mask.shape == (10, 10)
    assert rgb_crop.shape == (10, 10, 3)
    assert rgba_crop.shape == (10, 10, 4)

    # All pixels inside the square are True
    assert np.all(crop_mask)
    assert np.all(rgb_crop == 200)
    assert np.all(rgba_crop[:, :, 3] == 255)

    # Test partial mask inside crop
    mask[15:20, 15:20] = False  # Make a corner background
    crop_mask2, rgb_crop2, rgba_crop2 = create_masked_crop(img, mask, bbox, background_fill=(0, 0, 0))
    # In the corner [5:, 5:], pixels should be black (0, 0, 0) and alpha 0
    assert np.all(rgb_crop2[5:, 5:] == 0)
    assert np.all(rgba_crop2[5:, 5:, 3] == 0)


def test_single_leaf_bypass_accepted():
    """Test single leaf mode with a valid leaf mask."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[15:45, 15:45] = [40, 180, 50]

    def mock_fn(pt):
        m = np.zeros((60, 60), dtype=bool)
        m[15:45, 15:45] = True
        return m, 0.95

    mock_seg = create_mock_segmenter(mock_fn)
    cfg = LeafExtractionConfig(single_leaf_mode=True)
    res = extract_leaves(img, segmenter=mock_seg, config=cfg)

    assert res.is_single_leaf is True
    assert len(res.leaves) == 1
    assert len(res.rejected) == 0
    assert res.leaves[0].leaf_id == "leaf_0"
    assert res.leaves[0].confidence == 0.95


def test_single_leaf_bypass_rejected():
    """Test single leaf mode with an invalid mask (empty or out of bounds)."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)

    def mock_fn(pt):
        # Empty mask
        return np.zeros((40, 40), dtype=bool), 0.10

    mock_seg = create_mock_segmenter(mock_fn)
    cfg = LeafExtractionConfig(single_leaf_mode=True)
    res = extract_leaves(img, segmenter=mock_seg, config=cfg)

    assert res.is_single_leaf is True
    assert len(res.leaves) == 0
    assert len(res.rejected) == 1
    assert res.rejected[0].verdict == MaskVerdict.REJECT
    assert "empty" in res.rejected[0].reason.lower()


def test_multi_leaf_max_leaves_cap():
    """Test that candidate leaves beyond max_leaves are placed in rejected[]."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 4 distinct non-overlapping leaves
    def mock_fn(pt):
        x, y = pt[0]
        m = np.zeros((100, 100), dtype=bool)
        # 10x10 square around pt
        m[max(0, y - 5) : min(100, y + 5), max(0, x - 5) : min(100, x + 5)] = True
        return m, 0.90

    mock_seg = create_mock_segmenter(mock_fn)
    cfg = LeafExtractionConfig(
        single_leaf_mode=False,
        max_leaves=2,  # Cap at 2 leaves
        candidate_points=[(20, 20), (20, 80), (80, 20), (80, 80)],
    )

    res = extract_leaves(img, segmenter=mock_seg, config=cfg)
    assert len(res.leaves) == 2
    assert len(res.rejected) == 2
    assert "max_leaves" in res.rejected[0].reason


def test_invalid_image_inputs():
    """Test error handling for invalid input shapes and types."""
    with pytest.raises(DimensionMismatchError):
        extract_leaves(np.zeros((30, 30), dtype=np.uint8))  # 2D instead of 3D
    with pytest.raises(DimensionMismatchError):
        extract_leaves(np.zeros((30, 30, 4), dtype=np.uint8))  # 4-channel
    with pytest.raises(TypeError):
        extract_leaves(12345)  # Invalid type
