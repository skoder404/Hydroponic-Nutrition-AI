"""Unit tests for ADR 0006 Sensor Validation Protocol (T4-002)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ADR_FILE = ROOT / "docs" / "decisions" / "0006-sensor-validation-protocol.md"


def test_adr_file_exists() -> None:
    """Verify ADR 0006 file exists and has content."""
    assert ADR_FILE.is_file(), f"Missing ADR file: {ADR_FILE}"
    content = ADR_FILE.read_text(encoding="utf-8")
    assert len(content) > 1000


def test_adr_contains_crop_group_keys() -> None:
    """Verify ADR names group keys for lettuce, spinach, and tomato."""
    content = ADR_FILE.read_text(encoding="utf-8")
    assert "experiment, date" in content or "(experiment, date)" in content
    assert "bench" in content and "hour // 6" in content
    assert "calendar_month" in content or "source_timestamp.str[:7]" in content


def test_adr_contains_oversampling_inside_fold_rule() -> None:
    """Verify ADR defines oversampling-inside-fold rule and explains lettuce flaw."""
    content = ADR_FILE.read_text(encoding="utf-8")
    assert "Oversampling-Inside-Fold Rule" in content or "oversampling inside each fold" in content.lower()
    assert "GridSearchCV" in content
    assert "pH_Low" in content or "duplicate" in content.lower()
    assert "Pipeline" in content


def test_adr_contains_sign_offs() -> None:
    """Verify both Sushil and Yashwant signed off."""
    content = ADR_FILE.read_text(encoding="utf-8")
    assert "Sushil" in content
    assert "Yashwant" in content
    assert "Signed off" in content
