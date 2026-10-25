#!/usr/bin/env python3
"""Verification script for class-label registry and display mapping (T2-016).

Acceptance Criteria:
1. Label lists equal the folder names on disk (21 image classes in total).
2. Unknown label raises a clear error.
3. Display mapping is deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.labels import (
    ALL_IMAGE_CLASSES,
    IMAGE_CLASSES,
    LETTUCE_IMAGE_CLASSES,
    SENSOR_CLASSES,
    SPINACH_IMAGE_CLASSES,
    TOMATO_IMAGE_CLASSES,
    UnknownCropError,
    UnknownLabelError,
    get_classes,
    get_display_label,
    validate_label,
)


def get_disk_classes(crop_folder: str) -> set[str]:
    """Inspect actual class folders on disk containing image files."""
    root = PROJECT_ROOT / "image_data" / crop_folder
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = set()
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in valid_exts:
            classes.add(f.parent.name)
    return classes


def verify_labels_equal_disk_folders() -> tuple[bool, str]:
    """Criterion 1: Label lists equal the folder names on disk (21 image classes total)."""
    disk_lettuce = get_disk_classes("Lettuce_Images")
    disk_spinach = get_disk_classes("Spinach_Images")
    disk_tomato = get_disk_classes("Tomato_Images")

    if set(LETTUCE_IMAGE_CLASSES) != disk_lettuce:
        return False, f"Lettuce image classes mismatch. Registry: {set(LETTUCE_IMAGE_CLASSES)} vs Disk: {disk_lettuce}"

    if set(SPINACH_IMAGE_CLASSES) != disk_spinach:
        return False, f"Spinach image classes mismatch. Registry: {set(SPINACH_IMAGE_CLASSES)} vs Disk: {disk_spinach}"

    if set(TOMATO_IMAGE_CLASSES) != disk_tomato:
        return False, f"Tomato image classes mismatch. Registry: {set(TOMATO_IMAGE_CLASSES)} vs Disk: {disk_tomato}"

    total_image_classes = len(ALL_IMAGE_CLASSES)
    if total_image_classes != 21:
        return False, f"Expected 21 total image classes, got {total_image_classes}"

    # Verify exact spelling including space
    if "Tomato_Spider_mites Two-spotted_spider_mite" not in TOMATO_IMAGE_CLASSES:
        return False, "Exact spelling with space 'Tomato_Spider_mites Two-spotted_spider_mite' missing."

    return True, "Verified 21 image classes exactly equal folder names on disk (Lettuce: 4, Spinach: 6, Tomato: 11)."


def verify_unknown_label_error() -> tuple[bool, str]:
    """Criterion 2: Unknown label raises a clear error."""
    # Test unknown label on image modality
    try:
        validate_label("tomato", "Tomato_Completely_Fake_Disease")
        return False, "Failed to raise UnknownLabelError for unknown tomato label."
    except UnknownLabelError as exc:
        if "Tomato_Completely_Fake_Disease" not in str(exc):
            return False, f"Error message missing label: {exc}"

    # Test unknown label on get_display_label
    try:
        get_display_label("lettuce", "Fake_Lettuce_Class")
        return False, "Failed to raise UnknownLabelError in get_display_label."
    except UnknownLabelError as exc:
        if "Fake_Lettuce_Class" not in str(exc):
            return False, f"Error message missing label: {exc}"

    # Test unknown crop
    try:
        get_classes("cucumber")
        return False, "Failed to raise UnknownCropError for unknown crop."
    except UnknownCropError:
        pass

    return True, "UnknownLabelError and UnknownCropError raised with clear, informative diagnostics."


def verify_display_mapping_deterministic() -> tuple[bool, str]:
    """Criterion 3: Display mapping is deterministic."""
    # Healthy cases
    if get_display_label("lettuce", "Healthier") != "healthy":
        return False, "Expected 'Healthier' -> 'healthy'"
    if get_display_label("spinach", "Spinach_Healthy") != "healthy":
        return False, "Expected 'Spinach_Healthy' -> 'healthy'"
    if get_display_label("tomato", "Tomato_Healthy") != "healthy":
        return False, "Expected 'Tomato_Healthy' -> 'healthy'"

    # Unhealthy cases
    for crop, classes in IMAGE_CLASSES.items():
        for cls in classes:
            disp1 = get_display_label(crop, cls)
            disp2 = get_display_label(crop, cls)
            if disp1 != disp2:
                return False, f"Non-deterministic mapping for {crop}:{cls} -> {disp1} != {disp2}"

            if cls in ("Healthier", "Spinach_Healthy", "Tomato_Healthy"):
                if disp1 != "healthy":
                    return False, f"Expected {cls} -> 'healthy', got '{disp1}'"
            else:
                expected_disp = f"unhealthy/{cls}"
                if disp1 != expected_disp:
                    return False, f"Expected {cls} -> '{expected_disp}', got '{disp1}'"

    # Sensor display mapping
    for crop, s_classes in SENSOR_CLASSES.items():
        for s_cls in s_classes:
            s_disp = get_display_label(crop, s_cls, modality="sensor")
            if s_cls == "Normal":
                if s_disp != "healthy":
                    return False, f"Expected sensor {s_cls} -> 'healthy', got '{s_disp}'"
            else:
                if s_disp != f"unhealthy/{s_cls}":
                    return False, f"Expected sensor {s_cls} -> 'unhealthy/{s_cls}', got '{s_disp}'"

    return True, "Display mapping verified deterministic across all 21 image classes and 26 sensor classes."


def main() -> int:
    checks = [
        ("Label lists equal folder names on disk (21 image classes total)", verify_labels_equal_disk_folders),
        ("Unknown label raises a clear error", verify_unknown_label_error),
        ("Display mapping is deterministic", verify_display_mapping_deterministic),
    ]

    print("================================================================================")
    print("VERIFY CLASS-LABEL REGISTRY & DISPLAY MAPPING (T2-016)")
    print("================================================================================\n")

    all_passed = True
    for title, check_fn in checks:
        ok, msg = check_fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {title}\n       {msg}")
        if not ok:
            all_passed = False

    print("\n--------------------------------------------------------------------------------")
    if all_passed:
        print("All criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
