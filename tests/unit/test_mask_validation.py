"""Unit tests for per-leaf mask validation quality gate (T5-009)."""

import cv2
import numpy as np
import pytest

from ml.segmentation.mask_validation import (
    MaskValidationConfig,
    MaskVerdict,
    validate_mask,
    validate_masks,
)


@pytest.fixture
def base_frame():
    """Returns a standard 200x200 zero-filled frame."""
    return np.zeros((200, 200), dtype=np.uint8)


def test_empty_mask_rejected(base_frame):
    mask = base_frame.copy()
    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert not res.is_valid
    assert "empty" in res.reason.lower()
    assert res.metrics.mask_pixels == 0
    assert res.metrics.area_ratio == 0.0


def test_tiny_mask_rejected(base_frame):
    # Frame is 200x200 = 40,000 pixels. 1% is 400 pixels.
    # Create a 5x5 speck = 25 pixels (0.06% area)
    mask = base_frame.copy()
    mask[50:55, 50:55] = 1

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert not res.is_valid
    assert "too small" in res.reason.lower()
    assert res.metrics.area_ratio < 0.01


def test_whole_image_mask_rejected(base_frame):
    # Fill 95% of frame (38,000 pixels)
    mask = np.ones((200, 200), dtype=np.uint8)
    # Clear only small 10x10 corner
    mask[0:10, 0:10] = 0

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert not res.is_valid
    assert "too large" in res.reason.lower()
    assert res.metrics.area_ratio > 0.85


def test_severe_border_contact_rejected(base_frame):
    # A band along the top and left border touching the frame edge
    mask = base_frame.copy()
    mask[0:10, :] = 1  # 2000 pixels, of which 200 are top row border
    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert not res.is_valid
    assert "border" in res.reason.lower()
    assert res.metrics.border_contact_ratio > 0.08


def test_severely_fragmented_mask_rejected(base_frame):
    # 5 disjoint fragments of equal size (each 20% of mask)
    # Largest component is only 20% (< 60% retry threshold)
    mask = base_frame.copy()
    mask[20:30, 20:30] = 1
    mask[60:70, 60:70] = 1
    mask[100:110, 100:110] = 1
    mask[140:150, 140:150] = 1
    mask[170:180, 20:30] = 1

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert not res.is_valid
    assert "fragmented" in res.reason.lower()
    assert res.metrics.largest_component_ratio < 0.60


def test_retry_on_minor_fragmentation(base_frame):
    # Dominant component is 75% of mask, secondary is 25%
    # Between 60% and 85% -> RETRY
    mask = base_frame.copy()
    mask[40:100, 40:100] = 1  # 3600 pixels
    mask[140:175, 140:175] = 1  # 1225 pixels (total 4825, dominant ratio ~74.6%)

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.RETRY
    assert not res.is_valid
    assert "retry" in res.reason.lower()
    assert 0.60 <= res.metrics.largest_component_ratio < 0.85


def test_retry_on_moderate_border_touch(base_frame):
    # Clean leaf in center, but a narrow tip touches the border slightly (between 3% and 8%)
    mask = base_frame.copy()
    # 50x50 block in center = 2500 pixels
    mask[50:100, 50:100] = 1
    # Add a thin line from center (row 50) up to row 0 (column 75)
    # 50 pixels line, touching 1 pixel on top border (border_contact = 1 / 2550 = 0.04% - too small for retry).
    # Let's create a block touching border:
    # 1000 pixels total, 40 pixels on border -> 4%
    mask = base_frame.copy()
    mask[0:25, 50:90] = (
        1  # 25 * 40 = 1000 pixels. Row 0 has 40 pixels on border -> 40 / 1000 = 4%
    )
    config = MaskValidationConfig(
        min_area_ratio=0.01,
        retry_border_contact_ratio=0.03,
        max_border_contact_ratio=0.08,
    )
    res = validate_mask(mask, config=config)
    assert res.verdict == MaskVerdict.RETRY
    assert not res.is_valid
    assert "border contact" in res.reason.lower()


def test_valid_leaf_mask_accepted(base_frame):
    # Clean ellipse in the center of the frame (~15% of frame)
    mask = base_frame.copy()
    cv2.ellipse(mask, (100, 100), (50, 30), 45, 0, 360, 1, -1)

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.OK
    assert res.is_valid
    assert "valid" in res.reason.lower()
    assert 0.05 < res.metrics.area_ratio < 0.50
    assert res.metrics.border_contact_ratio == 0.0
    assert res.metrics.largest_component_ratio == 1.0


def test_custom_config_origins():
    # Verify documented origins in docstrings
    cfg = MaskValidationConfig()
    assert cfg.min_area_ratio == 0.01
    assert cfg.max_area_ratio == 0.85
    assert cfg.max_border_contact_ratio == 0.08
    assert cfg.min_component_ratio == 0.85

    doc = MaskValidationConfig.__doc__
    assert doc is not None
    assert "min_area_ratio" in doc
    assert "max_area_ratio" in doc
    assert "max_border_contact_ratio" in doc
    assert "min_component_ratio" in doc


def test_validate_masks_batch(base_frame):
    mask_empty = base_frame.copy()
    mask_valid = base_frame.copy()
    cv2.circle(mask_valid, (100, 100), 30, 1, -1)

    results = validate_masks([mask_empty, mask_valid])
    assert len(results) == 2
    assert results[0].verdict == MaskVerdict.REJECT
    assert results[1].verdict == MaskVerdict.OK
    assert results[1].is_valid
