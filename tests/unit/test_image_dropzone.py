"""Unit tests for T7-007: ImageDropzone and InputModeToggle client-side requirements."""

from __future__ import annotations

from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@pytest.fixture
def dropzone_content() -> str:
    dropzone_file = FRONTEND_DIR / "components" / "analyze" / "ImageDropzone.tsx"
    assert dropzone_file.exists(), "ImageDropzone.tsx file missing"
    return dropzone_file.read_text(encoding="utf-8")


@pytest.fixture
def toggle_content() -> str:
    toggle_file = FRONTEND_DIR / "components" / "analyze" / "InputModeToggle.tsx"
    assert toggle_file.exists(), "InputModeToggle.tsx file missing"
    return toggle_file.read_text(encoding="utf-8")


@pytest.fixture
def drawer_content() -> str:
    drawer_file = FRONTEND_DIR / "components" / "analyze" / "InputHelpDrawer.tsx"
    assert drawer_file.exists(), "InputHelpDrawer.tsx file missing"
    return drawer_file.read_text(encoding="utf-8")


def test_camera_capture_attribute_set(dropzone_content: str) -> None:
    """Mobile camera capture attribute must be explicitly set to 'environment'."""
    assert 'capture="environment"' in dropzone_content
    # Confirm it accepts jpeg, png, and webp
    assert 'accept="image/jpeg,image/png,image/webp"' in dropzone_content


def test_keyboard_navigation_attributes(dropzone_content: str) -> None:
    """Image dropzone must support keyboard-only interactions."""
    assert 'role="button"' in dropzone_content
    assert "tabIndex={" in dropzone_content or "tabIndex={0}" in dropzone_content
    assert "onKeyDown=" in dropzone_content
    assert 'e.key === "Enter"' in dropzone_content
    assert 'e.key === " "' in dropzone_content


def test_specific_validation_messages(dropzone_content: str) -> None:
    """Specific error messages for invalid formats and sizes must be defined."""
    assert "Invalid file format:" in dropzone_content
    assert "Only JPEG, PNG, and WebP images are supported" in dropzone_content
    assert "exceeds maximum allowed limit of 15 MB" in dropzone_content
    assert "15 * 1024 * 1024" in dropzone_content


def test_input_mode_toggle_accessibility(toggle_content: str) -> None:
    """InputModeToggle must provide clear accessibility semantics for screen readers."""
    assert 'role="radiogroup"' in toggle_content
    assert 'aria-label="Image Input Mode"' in toggle_content
    assert "aria-checked=" in toggle_content
    assert "single_leaf" in toggle_content
    assert "plant" in toggle_content


def test_input_help_drawer_accessibility(drawer_content: str) -> None:
    """InputHelpDrawer must conform to modal dialog accessibility patterns."""
    assert 'role="dialog"' in drawer_content
    assert 'aria-modal="true"' in drawer_content
    assert 'aria-labelledby="help-drawer-title"' in drawer_content
    assert 'e.key === "Escape"' in drawer_content
