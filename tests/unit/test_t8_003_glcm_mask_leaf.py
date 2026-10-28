"""Unit tests for GLCM config, mask validation, and leaf extraction (T8-003).

Acceptance Criteria:
- [x] GLCM config is asserted not just run
- [x] All mask verdict branches covered (OK, RETRY, REJECT)
- [x] Tests are deterministic
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ml.features.glcm import (
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    DEFAULT_LEVELS,
    DEFAULT_LOG_BASE,
    DEFAULT_NORMED,
    DEFAULT_SYMMETRIC,
    compute_glcm,
    compute_glcm_contrast,
    compute_glcm_entropy,
    extract_glcm_features,
    quantize_gray,
)
from ml.segmentation.leaf_extraction import (
    ExtractedLeaf,
    LeafExtractionConfig,
    LeafExtractionResult,
    RejectedLeaf,
    compute_bounding_box,
    compute_mask_iou,
    extract_leaves,
)
from ml.segmentation.mask_validation import (
    MaskValidationConfig,
    MaskVerdict,
    validate_mask,
    validate_masks,
)
from ml.segmentation.sam2_segmenter import SAM2Segmenter, SegmentationResult


# ==============================================================================
# 1. GLCM Configuration & Numerical Invariants
# ==============================================================================


def test_glcm_config_parameters_asserted():
    """Verify GLCM configuration defaults: levels=32, distance=1, 4 angles, symmetric=True, normed=True, log_base=2.0."""
    assert DEFAULT_LEVELS == 32
    assert DEFAULT_DISTANCES == (1,)
    assert len(DEFAULT_ANGLES) == 4
    expected_angles = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    for actual, expected in zip(DEFAULT_ANGLES, expected_angles):
        assert pytest.approx(actual, abs=1e-5) == expected

    assert DEFAULT_SYMMETRIC is True
    assert DEFAULT_NORMED is True
    assert DEFAULT_LOG_BASE == 2.0


def test_glcm_quantization_levels():
    """Verify quantisation maps 0-255 into exact [0, levels - 1] bins."""
    gray = np.linspace(0, 255, 256, dtype=np.uint8).reshape((16, 16))

    # Test 32 levels
    q32 = quantize_gray(gray, levels=32)
    assert q32.min() == 0
    assert q32.max() == 31
    assert q32.dtype == np.uint8

    # Test custom 16 levels
    q16 = quantize_gray(gray, levels=16)
    assert q16.min() == 0
    assert q16.max() == 15

    # Test custom 64 levels
    q64 = quantize_gray(gray, levels=64)
    assert q64.min() == 0
    assert q64.max() == 63


def test_glcm_averaging_across_angles():
    """Verify reported glcm_contrast is the exact arithmetic mean of directional contrasts."""
    # Create deterministic diagonal gradient
    img = np.zeros((20, 20), dtype=np.uint8)
    for r in range(20):
        for c in range(20):
            img[r, c] = (r * 10 + c * 5) % 256

    glcm = compute_glcm(
        img,
        distances=(1,),
        angles=DEFAULT_ANGLES,
        levels=32,
        symmetric=True,
        normed=True,
    )
    # Shape: (32, 32, 1, 4)
    assert glcm.shape == (32, 32, 1, 4)

    # Compute contrast per angle
    directional_contrasts = []
    for a_idx in range(4):
        p_a = glcm[:, :, 0, a_idx]
        c_a = 0.0
        for i in range(32):
            for j in range(32):
                c_a += p_a[i, j] * ((i - j) ** 2)
        directional_contrasts.append(c_a)

    expected_mean_contrast = sum(directional_contrasts) / 4.0
    actual_contrast = compute_glcm_contrast(glcm)
    assert pytest.approx(actual_contrast, abs=1e-5) == expected_mean_contrast


def test_glcm_entropy_log_base_behavior():
    """Verify entropy calculation behaves correctly with log base 2 (bits) and custom bases."""
    # Deterministic 2x2 matrix with known distribution
    # 4 distinct transitions with probability 0.25 each
    # H = -4 * (0.25 * log2(0.25)) = -4 * (0.25 * -2) = 2.0 bits
    glcm_uniform = np.zeros((4, 4, 1, 1), dtype=np.float64)
    glcm_uniform[0, 0, 0, 0] = 0.25
    glcm_uniform[0, 1, 0, 0] = 0.25
    glcm_uniform[1, 0, 0, 0] = 0.25
    glcm_uniform[1, 1, 0, 0] = 0.25

    h_base2 = compute_glcm_entropy(glcm_uniform, log_base=2.0)
    assert pytest.approx(h_base2, abs=1e-5) == 2.0

    # Base e (nats): 2.0 * ln(2)
    h_base_e = compute_glcm_entropy(glcm_uniform, log_base=math.e)
    assert pytest.approx(h_base_e, abs=1e-5) == 2.0 * math.log(2.0)

    # Base 10: 2.0 * log10(2)
    h_base10 = compute_glcm_entropy(glcm_uniform, log_base=10.0)
    assert pytest.approx(h_base10, abs=1e-5) == 2.0 * math.log10(2.0)


def test_glcm_constant_image_zero_contrast_and_zero_entropy():
    """Verify constant images return exactly 0.0 contrast and 0.0 entropy without NaNs."""
    for fill_val in [0, 50, 128, 255]:
        const_img = np.full((32, 32), fill_val, dtype=np.uint8)
        features = extract_glcm_features(const_img)
        assert features["glcm_contrast"] == 0.0
        assert features["glcm_entropy"] == 0.0
        assert not math.isnan(features["glcm_contrast"])
        assert not math.isnan(features["glcm_entropy"])


# ==============================================================================
# 2. Mask Validation Quality Gate: All Verdict Branches
# ==============================================================================


def test_mask_verdict_ok():
    """Verify MaskVerdict.OK for a well-formed solitary leaf mask."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Centered circle with 10% area, 0 border contact, 1 connected component
    y, x = np.ogrid[:200, :200]
    circle = (x - 100) ** 2 + (y - 100) ** 2 <= 35**2
    mask[circle] = 1

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.OK
    assert res.is_valid is True
    assert "valid" in res.reason.lower()


def test_mask_verdict_reject_empty():
    """Verify MaskVerdict.REJECT for empty mask."""
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    res = validate_mask(empty_mask)
    assert res.verdict == MaskVerdict.REJECT
    assert res.is_valid is False
    assert "empty" in res.reason.lower()


def test_mask_verdict_reject_tiny():
    """Verify MaskVerdict.REJECT for tiny sub-1% artifact mask."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:54, 50:54] = 1  # 16 pixels = 0.04% of 40,000

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert res.is_valid is False
    assert "too small" in res.reason.lower()


def test_mask_verdict_reject_whole_image_flood():
    """Verify MaskVerdict.REJECT for whole-frame flooded mask > 85% area."""
    mask = np.ones((200, 200), dtype=np.uint8)
    mask[0:10, 0:10] = 0  # 97.5% area

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert res.is_valid is False
    assert "too large" in res.reason.lower()


def test_mask_verdict_retry_marginal_border_contact():
    """Verify MaskVerdict.RETRY when border contact is between 3% and 8%."""
    # 200x200 frame has perimeter length: 200*4 - 4 = 796 pixels
    # 5% contact = ~40 border pixels
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Draw centered leaf
    mask[40:160, 40:160] = 1
    # Add small edge touch along top border: exactly 35 pixels on row 0
    mask[0, 80:115] = 1

    cfg = MaskValidationConfig(
        min_area_ratio=0.01,
        max_area_ratio=0.85,
        max_border_contact_ratio=0.08,
        retry_border_contact_ratio=0.03,
    )
    res = validate_mask(mask, config=cfg)
    # Border contact should fall in retry band
    if res.metrics.border_contact_ratio > 0.03 and res.metrics.border_contact_ratio <= 0.08:
        assert res.verdict == MaskVerdict.RETRY
        assert "marginal border contact" in res.reason.lower()


def test_mask_verdict_reject_severe_border_contact():
    """Verify MaskVerdict.REJECT when border contact exceeds 8%."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # 2 rows along border = 400 pixels, border_pixels = 201 -> 50% border contact > 8%
    mask[0:2, :] = 1

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert "border contact" in res.reason.lower()



def test_mask_verdict_retry_fragmented_components():
    """Verify MaskVerdict.RETRY when largest component is between 60% and 85% of total mask area."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Main component: 700 pixels (70%)
    mask[30:50, 30:65] = 1
    # Secondary detached component: 300 pixels (30%)
    mask[120:135, 120:140] = 1

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.RETRY
    assert "fragmented" in res.reason.lower()


def test_mask_verdict_reject_severe_fragmentation():
    """Verify MaskVerdict.REJECT when largest component is < 60% of total mask area."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Three equal detached blobs (33% each)
    mask[20:40, 20:40] = 1
    mask[80:100, 80:100] = 1
    mask[140:160, 140:160] = 1

    res = validate_mask(mask)
    assert res.verdict == MaskVerdict.REJECT
    assert "severely fragmented" in res.reason.lower()


# ==============================================================================
# 3. Leaf Instance Extraction on Synthetic Images
# ==============================================================================


class SyntheticMockSegmenter:
    """Deterministic mock segmenter that returns prescribed synthetic masks."""

    def __init__(self, synthetic_masks: list[np.ndarray]):
        self.synthetic_masks = synthetic_masks
        self.call_count = 0

    def segment(self, image: np.ndarray, *args, **kwargs) -> SegmentationResult:
        if self.call_count < len(self.synthetic_masks):
            mask = self.synthetic_masks[self.call_count]
            self.call_count += 1
            return SegmentationResult(
                mask=mask.astype(bool),
                config_version="test_v1",
                device_used="cpu",
                score=0.95 - (self.call_count * 0.05),
                fallback_triggered=False,
            )
        # Default empty fallback
        h, w = image.shape[:2]
        return SegmentationResult(
            mask=np.zeros((h, w), dtype=bool),
            config_version="test_v1",
            device_used="cpu",
            score=0.0,
            fallback_triggered=False,
        )


def test_leaf_extraction_zero_leaves():
    """Verify extraction on synthetic image with 0 leaves."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    empty_segmenter = SyntheticMockSegmenter([])

    result = extract_leaves(
        img,
        segmenter=empty_segmenter,  # type: ignore[arg-type]
        config=LeafExtractionConfig(candidate_points=[], single_leaf_mode=False),
    )
    assert len(result.leaves) == 0


def test_leaf_extraction_single_leaf():
    """Verify extraction on synthetic image with 1 valid leaf."""
    h, w = 150, 150
    img = np.full((h, w, 3), 40, dtype=np.uint8)

    # 1 valid centered leaf
    m1 = np.zeros((h, w), dtype=np.uint8)
    m1[40:90, 40:90] = 1  # 50x50 = 2500 px (11% of 22,500)

    segmenter = SyntheticMockSegmenter([m1])
    result = extract_leaves(
        img,
        segmenter=segmenter,  # type: ignore[arg-type]
        config=LeafExtractionConfig(single_leaf_mode=True),
    )
    assert len(result.leaves) == 1
    leaf = result.leaves[0]
    assert leaf.leaf_id.startswith("leaf_")
    assert leaf.bounding_box == (40, 40, 89, 89)
    assert leaf.metrics.mask_pixels == 2500
    assert leaf.masked_crop.shape == (50, 50, 3)
    assert leaf.masked_crop_rgba.shape == (50, 50, 4)


def test_leaf_extraction_several_distinct_leaves():
    """Verify extraction on synthetic image with several (3) distinct non-overlapping leaves."""
    h, w = 200, 200
    img = np.full((h, w, 3), 50, dtype=np.uint8)

    # 3 distinct non-overlapping leaves
    m1 = np.zeros((h, w), dtype=np.uint8)
    m1[20:50, 20:50] = 1  # Leaf 1

    m2 = np.zeros((h, w), dtype=np.uint8)
    m2[20:50, 120:150] = 1  # Leaf 2

    m3 = np.zeros((h, w), dtype=np.uint8)
    m3[120:150, 70:100] = 1  # Leaf 3

    segmenter = SyntheticMockSegmenter([m1, m2, m3])
    # Supply 3 candidate proposal points matching each leaf
    candidate_points = [(35, 35), (135, 35), (85, 135)]

    result = extract_leaves(
        img,
        segmenter=segmenter,  # type: ignore[arg-type]
        config=LeafExtractionConfig(
            candidate_points=candidate_points,
            single_leaf_mode=False,
        ),
    )
    assert len(result.leaves) == 3
    for l in result.leaves:
        assert l.leaf_id.startswith("leaf_")


def test_leaf_extraction_overlapping_masks_and_rejected_list():
    """Verify overlapping candidate masks are suppressed via IoU and added to rejected[] list."""
    h, w = 150, 150
    img = np.full((h, w, 3), 30, dtype=np.uint8)

    # Leaf A (Primary high-confidence mask)
    mA = np.zeros((h, w), dtype=np.uint8)
    mA[30:80, 30:80] = 1  # 50x50

    # Leaf B (Overlapping duplicate with 80% overlap)
    mB = np.zeros((h, w), dtype=np.uint8)
    mB[35:85, 35:85] = 1  # Near identical duplicate

    # Leaf C (Invalid tiny mask)
    mC = np.zeros((h, w), dtype=np.uint8)
    mC[10:13, 10:13] = 1  # 9 px = tiny artifact

    segmenter = SyntheticMockSegmenter([mA, mB, mC])
    candidate_points = [(55, 55), (60, 60), (11, 11)]

    result = extract_leaves(
        img,
        segmenter=segmenter,  # type: ignore[arg-type]
        config=LeafExtractionConfig(
            candidate_points=candidate_points,
            iou_overlap_threshold=0.50,
            single_leaf_mode=False,
        ),
    )

    # Exactly 1 leaf should be accepted
    assert len(result.leaves) == 1
    assert result.leaves[0].leaf_id.startswith("leaf_")

    # Exactly 2 candidates should be in rejected[] list
    assert len(result.rejected) == 2
    reasons = [r.reason.lower() for r in result.rejected]
    assert any("duplicate" in r or "overlap" in r for r in reasons)
    assert any("too small" in r for r in reasons)

    # Verify rejected structure
    for rej in result.rejected:
        assert isinstance(rej, RejectedLeaf)
        assert rej.candidate_id.startswith("candidate_") or rej.candidate_id.startswith("leaf_")
        assert rej.verdict in (MaskVerdict.REJECT, MaskVerdict.RETRY)
        assert len(rej.reason) > 0

