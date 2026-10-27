"""Unit tests for SAM 2.1 Segmenter wrapper (T5-008)."""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

from ml.segmentation.sam2_segmenter import (
    SAM2Segmenter,
    SegmentationError,
)
from scripts.verify_sam2_segmenter import (
    verify_lazy_loading,
    verify_mask_and_config_version,
    verify_oom_handling_and_fallback,
)


def test_lazy_loading():
    ok, msg = verify_lazy_loading()
    assert ok, msg


def test_mask_and_config_version():
    ok, msg = verify_mask_and_config_version()
    assert ok, msg


def test_oom_handling_and_fallback():
    ok, msg = verify_oom_handling_and_fallback()
    assert ok, msg


def test_invalid_image_path():
    segmenter = SAM2Segmenter()
    with pytest.raises(SegmentationError):
        segmenter.segment("non_existent_image_path_12345.jpg")
    segmenter.unload()


def test_missing_checkpoint():
    segmenter = SAM2Segmenter(checkpoint_path="non_existent_checkpoint.pt")
    with pytest.raises(FileNotFoundError):
        segmenter.load_model()
