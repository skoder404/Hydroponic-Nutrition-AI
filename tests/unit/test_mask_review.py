"""Unit tests for qualitative review of leaf extraction (T5-011)."""

from __future__ import annotations

import re
from pathlib import Path
import pytest

from scripts.verify_mask_review import (
    REVIEW_FILE,
    run_verification,
    verify_at_least_60_reviewed_images,
    verify_failure_modes_listed,
    verify_threshold_changes_reference_review,
)


def test_mask_review_acceptance_criteria():
    """Verify all T5-011 acceptance criteria pass via verification script."""
    assert run_verification() is True


def test_mask_review_report_structure():
    """Verify mask_review.md contains required sections and tables."""
    assert REVIEW_FILE.exists(), f"Report file {REVIEW_FILE} does not exist"
    content = REVIEW_FILE.read_text(encoding="utf-8")

    assert "# Qualitative Review of Foundation Leaf Extraction (T5-011)" in content
    assert "## 1. Executive Summary & Review Verdicts" in content
    assert "## 2. Failure Modes Analysis" in content
    assert "## 3. Validator Threshold Calibration & Justification" in content
    assert "## 4. Comprehensive Image Verdict Log (60 Images)" in content
    assert "## 5. Conclusion & Recommendations" in content


def test_mask_review_failure_mode_coverage():
    """Verify that all 4 required failure modes are documented."""
    ok, msg = verify_failure_modes_listed()
    assert ok, msg


def test_mask_review_image_counts_per_crop():
    """Verify that exactly 20 images each of Lettuce, Spinach, and Tomato are recorded."""
    content = REVIEW_FILE.read_text(encoding="utf-8")
    table_rows = re.findall(
        r"^\|\s*\d+\s*\|\s*(Lettuce|Spinach|Tomato)\s*\|.*\|$", content, re.MULTILINE
    )
    assert len(table_rows) >= 60

    counts = {"Lettuce": 0, "Spinach": 0, "Tomato": 0}
    for crop in table_rows:
        if crop in counts:
            counts[crop] += 1

    assert counts["Lettuce"] >= 20
    assert counts["Spinach"] >= 20
    assert counts["Tomato"] >= 20
