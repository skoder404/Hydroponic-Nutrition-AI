"""Unit tests for T5-006: Decision on droop_angle (ADR 0007)."""

from __future__ import annotations

from pathlib import Path

from scripts.verify_droop_angle_decision import run_verification

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


def test_adr_0007_file_exists() -> None:
    adr_file = DOCS_DIR / "decisions" / "0007-droop-angle.md"
    assert adr_file.is_file(), "0007-droop-angle.md must exist under docs/decisions/"


def test_adr_0007_content_and_criteria() -> None:
    adr_file = DOCS_DIR / "decisions" / "0007-droop-angle.md"
    content = adr_file.read_text(encoding="utf-8")

    # Decision
    assert "REMOVE" in content
    # Empirical evidence
    assert "20" in content
    assert "cv2.fitEllipse" in content
    assert "flipTB" in content
    # Impact list
    assert "glcm_contrast" in content
    assert "glcm_entropy" in content
    assert "bounding_aspect_ratio" in content
    assert "solidity" in content
    # Agreement
    assert "Yashwant" in content
    assert "Sushil" in content


def test_droop_angle_verification_script() -> None:
    assert run_verification() is True
