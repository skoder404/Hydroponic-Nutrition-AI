"""Unit tests for Image Dataset Audit (T2-013)."""
from __future__ import annotations

from scripts.verify_image_audit import (
    verify_class_labels_and_differences,
    verify_corrupt_files_tracked,
    verify_presentation_status,
    verify_reports_exist,
)


def test_reports_exist():
    ok, msg = verify_reports_exist()
    assert ok, msg


def test_corrupt_files_tracked():
    ok, msg = verify_corrupt_files_tracked()
    assert ok, msg


def test_presentation_status():
    ok, msg = verify_presentation_status()
    assert ok, msg


def test_class_labels_and_differences():
    ok, msg = verify_class_labels_and_differences()
    assert ok, msg
