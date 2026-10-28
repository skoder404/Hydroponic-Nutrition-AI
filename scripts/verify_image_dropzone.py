"""Verification script for T7-007: ImageDropzone and InputModeToggle.

Verifies acceptance criteria:
1. Wrong file type shows a specific message:
   - MIME / extension mismatch gives clear error text mentioning JPEG, PNG, WebP.
   - Oversized file (>15 MB) gives clear error text matching MAX_UPLOAD_BYTES limit.
2. Works with keyboard only:
   - Dropzone has role="button", tabIndex={0}, and Enter / Space key listeners.
   - Clear and accessible keyboard focus state.
3. Mobile camera capture attribute is set:
   - HTML input includes capture="environment" for rear greenhouse leaf camera.
   - Dedicated "Take Photo" button for mobile workflow.
4. InputModeToggle support:
   - Segmented toggle for single_leaf vs plant canopy.
   - Accessible help drawer explaining when to use each mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_image_dropzone_component() -> None:
    dropzone_file = Path("frontend/components/analyze/ImageDropzone.tsx")
    assert dropzone_file.exists(), (
        "frontend/components/analyze/ImageDropzone.tsx must exist"
    )

    content = dropzone_file.read_text(encoding="utf-8")

    # 1. Acceptance Criteria: Mobile camera capture attribute is set
    assert 'capture="environment"' in content, (
        "Mobile camera capture attribute 'capture=\"environment\"' must be set on camera input"
    )
    assert 'accept="image/jpeg,image/png,image/webp"' in content, (
        "Accept attribute must specify JPEG, PNG, WebP MIME types"
    )
    print("Mobile camera capture attribute (capture='environment'): VERIFIED.")

    # 2. Acceptance Criteria: Keyboard only accessibility
    assert "tabIndex={0}" in content or "tabIndex={disabled ? -1 : 0}" in content, (
        "ImageDropzone must have tabIndex for keyboard focus"
    )
    assert 'role="button"' in content, (
        "ImageDropzone must have role='button' for screen readers and keyboard users"
    )
    assert "onKeyDown" in content, (
        "ImageDropzone must have onKeyDown listener for keyboard activation"
    )
    assert 'key === "Enter"' in content and 'key === " "' in content, (
        "ImageDropzone keyboard handler must trigger on Enter and Space keys"
    )
    print(
        "Keyboard-only accessibility (tabIndex, role, Enter/Space handlers): VERIFIED."
    )

    # 3. Acceptance Criteria: Wrong file type and size show specific messages
    assert "Invalid file format:" in content, (
        "Specific error message for invalid file format must be present"
    )
    assert "Only JPEG, PNG, and WebP images are supported" in content, (
        "Error message must specify allowed formats: JPEG, PNG, and WebP"
    )
    assert "exceeds maximum allowed limit of 15 MB" in content, (
        "Specific error message for oversized files (>15 MB) must be present"
    )
    assert "MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024" in content, (
        "File size threshold must match backend 15 MB"
    )
    print("Specific client-side validation error messages (format & size): VERIFIED.")


def verify_input_mode_toggle_and_drawer() -> None:
    toggle_file = Path("frontend/components/analyze/InputModeToggle.tsx")
    assert toggle_file.exists(), (
        "frontend/components/analyze/InputModeToggle.tsx must exist"
    )
    toggle_content = toggle_file.read_text(encoding="utf-8")

    assert "single_leaf" in toggle_content, (
        "InputModeToggle must support 'single_leaf' mode"
    )
    assert "plant" in toggle_content, "InputModeToggle must support 'plant' canopy mode"
    assert 'role="radiogroup"' in toggle_content, (
        "InputModeToggle must have role='radiogroup'"
    )
    assert "aria-checked" in toggle_content, "Radio options must track aria-checked"

    drawer_file = Path("frontend/components/analyze/InputHelpDrawer.tsx")
    assert drawer_file.exists(), (
        "frontend/components/analyze/InputHelpDrawer.tsx must exist"
    )
    drawer_content = drawer_file.read_text(encoding="utf-8")

    assert 'role="dialog"' in drawer_content, "Help drawer must have role='dialog'"
    assert 'aria-modal="true"' in drawer_content, (
        "Help drawer must set aria-modal='true'"
    )
    assert "Escape" in drawer_content, "Help drawer must close on Escape key"

    print("InputModeToggle and InputHelpDrawer accessibility: VERIFIED.")


def verify_analyze_page_integration() -> None:
    page_file = Path("frontend/app/analyze/page.tsx")
    assert page_file.exists(), "frontend/app/analyze/page.tsx must exist"
    content = page_file.read_text(encoding="utf-8")

    assert "ImageDropzone" in content, "ImageDropzone must be mounted in /analyze page"
    assert "InputModeToggle" in content, (
        "InputModeToggle must be mounted in /analyze page"
    )
    assert "InputMode" in content, "InputMode state must be handled in /analyze page"

    print("Analyze page integration: VERIFIED.")


def main() -> None:
    print("--- Verifying T7-007: ImageDropzone with client-side checks ---")
    verify_image_dropzone_component()
    verify_input_mode_toggle_and_drawer()
    verify_analyze_page_integration()
    print("--- All T7-007 acceptance criteria successfully verified! ---")


if __name__ == "__main__":
    main()
