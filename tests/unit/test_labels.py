"""Unit tests for class-label registry and display mapping (T2-016)."""
from __future__ import annotations

import pytest

from ml.labels import (
    UnknownModalityError,
    get_class_from_index,
    get_class_index,
    get_classes,
    num_classes,
)
from scripts.verify_labels import (
    verify_display_mapping_deterministic,
    verify_labels_equal_disk_folders,
    verify_unknown_label_error,
)


def test_labels_equal_disk_folders():
    ok, msg = verify_labels_equal_disk_folders()
    assert ok, msg


def test_unknown_label_error():
    ok, msg = verify_unknown_label_error()
    assert ok, msg


def test_display_mapping_deterministic():
    ok, msg = verify_display_mapping_deterministic()
    assert ok, msg


def test_class_indexing():
    for crop in ("lettuce", "spinach", "tomato"):
        classes = get_classes(crop, "image")
        assert num_classes(crop, "image") == len(classes)
        for idx, cls in enumerate(classes):
            assert get_class_index(crop, cls, "image") == idx
            assert get_class_from_index(crop, idx, "image") == cls


def test_invalid_modality():
    with pytest.raises(UnknownModalityError):
        get_classes("tomato", "audio")
