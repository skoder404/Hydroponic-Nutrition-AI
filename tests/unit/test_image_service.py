"""Unit tests for ImageService pipeline (T6-006).

Acceptance Criteria:
1. Plant mode returns one result per valid leaf plus a count summary.
2. Rejected leaves are listed with reasons, not dropped.
3. No averaged or merged confidence appears in the response.
4. If no valid leaf exists return SEGMENTATION_FAILED (retryable), never a prediction.
5. Single-leaf mode returns per-leaf result with top-level confidence.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.app.core.errors import InvalidInputError, SegmentationFailedError
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.image_service import ImageService, get_image_service, reset_image_service
from ml.segmentation.leaf_extraction import (
    ExtractedLeaf,
    LeafExtractionConfig,
    LeafExtractionResult,
    RejectedLeaf as SegmenterRejectedLeaf,
    compute_bounding_box,
)
from ml.segmentation.mask_validation import MaskMetrics, MaskVerdict
from ml.segmentation.sam2_segmenter import SAM2Segmenter, SegmentationResult


def create_mock_segmenter_with_results(
    valid_count: int = 2,
    rejected_count: int = 1,
) -> SAM2Segmenter:
    """Create a mock segmenter that returns controlled valid and invalid leaf masks."""
    mock_seg = SAM2Segmenter.__new__(SAM2Segmenter)
    mock_seg._model = True
    mock_seg._predictor = True
    mock_seg._loaded_device = "cpu"
    mock_seg.checkpoint_path = "mock.pt"
    mock_seg.config_path = "mock.yaml"
    mock_seg.fallback_on_oom = False

    def mock_segment(image, point_coords=None, point_labels=None, box=None, simulate_oom=False):
        h, w, _ = image.shape
        # If point coordinate is in top-left, make a valid leaf mask
        if point_coords and point_coords[0][0] < w // 2:
            mask = np.zeros((h, w), dtype=bool)
            mask[20:60, 20:60] = True
            score = 0.95
        elif point_coords and point_coords[0][0] >= w // 2:
            # Second valid leaf
            mask = np.zeros((h, w), dtype=bool)
            mask[70:120, 70:120] = True
            score = 0.90
        else:
            # Center fallback
            mask = np.zeros((h, w), dtype=bool)
            mask[30:80, 30:80] = True
            score = 0.92

        return SegmentationResult(
            mask=mask,
            config_version="mock_sam2:v1",
            device_used="cpu",
            score=score,
            fallback_triggered=False,
        )

    mock_seg.segment = mock_segment
    return mock_seg


@pytest.fixture(autouse=True)
def clean_image_service():
    """Ensure clean singleton before and after each test."""
    reset_image_service()
    yield
    reset_image_service()


def test_plant_mode_returns_per_leaf_and_discrete_summary():
    """Criterion 1: Plant mode returns one result per valid leaf plus a count summary."""
    service = ImageService()

    # Create a 200x200 RGB image with two green vegetation spots
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 30, (25, 200, 25), -1)
    cv2.circle(img, (140, 140), 30, (25, 200, 25), -1)

    response = service.predict(
        image=img,
        crop="lettuce",
        input_mode="plant",
        provenance="manual",
    )

    assert isinstance(response, PredictionResponse)
    assert response.crop == "lettuce"
    assert response.modality == "image"
    assert response.input_mode == "plant"

    # Per-leaf results present
    assert len(response.leaves) >= 1
    for leaf in response.leaves:
        assert leaf.leaf_id.startswith("leaf_")
        assert len(leaf.bbox) == 4
        assert leaf.confidence is not None
        assert 0.0 <= leaf.confidence <= 1.0
        assert leaf.confidence_semantics is not None
        assert isinstance(leaf.uncertain, bool)

    # Discrete plant summary present
    assert response.summary is not None
    assert response.summary.leaf_count == len(response.leaves)
    assert isinstance(response.summary.class_counts, dict)
    assert sum(response.summary.class_counts.values()) == response.summary.leaf_count

    # Criterion 3: In plant mode, top-level confidence is None (never averaged)
    assert response.confidence is None
    assert response.confidence_semantics is None


def test_single_leaf_mode_returns_top_level_confidence():
    """Criterion 5: Single-leaf mode returns per-leaf result and top-level confidence."""
    service = ImageService()

    # Create a single green leaf spot
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 45, (30, 210, 30), -1)

    response = service.predict(
        image=img,
        crop="lettuce",
        input_mode="leaf",
        provenance="manual",
    )

    assert response.input_mode == "single_leaf"
    assert len(response.leaves) == 1
    assert response.confidence is not None
    assert response.confidence_semantics is not None
    assert response.summary is not None
    assert response.summary.leaf_count == 1
    assert response.summary.class_counts[response.predicted_class] == 1


def test_segmentation_failed_when_no_valid_leaf():
    """Criterion 4: If no valid leaf exists return SEGMENTATION_FAILED (retryable)."""
    service = ImageService()

    # All-black image with no vegetation
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(SegmentationFailedError) as exc_info:
        service.predict(
            image=black_img,
            crop="lettuce",
            input_mode="plant",
        )

    err = exc_info.value
    assert err.code == "SEGMENTATION_FAILED"
    assert err.status_code == 422
    assert err.retryable is True


def test_rejected_leaves_are_listed_with_reasons():
    """Criterion 2: Rejected leaves are listed with reasons, not dropped."""
    service = ImageService()

    # Image with one valid leaf and tiny noise specks that get rejected by area gate
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Valid leaf
    cv2.circle(img, (100, 100), 40, (20, 220, 20), -1)
    # Tiny noise dots (1-2 pixels) that will be rejected
    img[250, 250] = (20, 220, 20)
    img[280, 280] = (20, 220, 20)

    response = service.predict(
        image=img,
        crop="spinach",
        input_mode="plant",
    )

    assert len(response.leaves) >= 1
    # Check that rejected_leaves is a list of RejectedLeaf instances with non-empty reasons
    assert isinstance(response.rejected_leaves, list)
    for rej in response.rejected_leaves:
        assert rej.leaf_id != ""
        assert rej.reason != ""
        assert rej.verdict == "reject"


def test_no_averaged_confidence_invariant():
    """Criterion 3: No averaged or merged confidence appears in the response."""
    service = ImageService()

    img = np.zeros((250, 250, 3), dtype=np.uint8)
    cv2.circle(img, (75, 75), 35, (30, 210, 30), -1)
    cv2.circle(img, (175, 175), 35, (30, 210, 30), -1)

    response = service.predict(
        image=img,
        crop="tomato",
        input_mode="plant",
    )

    # In multi-leaf plant mode, top-level confidence must strictly be None
    assert response.confidence is None
    # Summary must have leaf_count and class_counts only, no 'mean_confidence'
    assert hasattr(response.summary, "leaf_count")
    assert hasattr(response.summary, "class_counts")
    assert not hasattr(response.summary, "average_confidence")
    assert not hasattr(response.summary, "confidence")


def test_unknown_crop_rejected():
    """Verify unknown crop is rejected with InvalidInputError."""
    service = ImageService()
    img = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(InvalidInputError) as exc_info:
        service.predict(
            image=img,
            crop="banana",
            input_mode="plant",
        )
    assert "Unsupported crop" in str(exc_info.value)
