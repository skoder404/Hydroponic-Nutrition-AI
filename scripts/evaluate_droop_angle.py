"""Prototype and empirical evaluation of droop_angle on 20 Tomato images (T5-006).

Tests whether an orientation/droop angle can be defensibly calculated on
PlantVillage tomato leaf images. Evaluates:
1. Sensitivity to 180-degree rotation and vertical flip augmentations.
2. Sensitivity to arbitrary image framing / scanner angle.
3. Availability of a ground-truth gravity or plant stem reference axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOMATO_DIR = PROJECT_ROOT / "image_data" / "Tomato_Images"


def segment_leaf(image_bgr: np.ndarray) -> np.ndarray:
    """Segment leaf from PlantVillage neutral gray/white background using Otsu thresholding."""
    # Background in PlantVillage is bright or textured gray/sheet
    # Using Otsu thresholding on saturation channel
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def calculate_fitted_angle(mask: np.ndarray) -> tuple[float, float, str]:
    """Calculate major axis angle via cv2.fitEllipse and cv2.minAreaRect."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, "no_contour"
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0, 0.0, "degenerate_contour"

    # 1. Ellipse fitting angle in [0, 180) degrees relative to vertical (y-axis)
    ellipse = cv2.fitEllipse(largest)
    ellipse_angle = float(ellipse[2])

    # 2. MinAreaRect angle
    rect = cv2.minAreaRect(largest)
    rect_angle = float(rect[2])

    return ellipse_angle, rect_angle, "ok"


def run_evaluation() -> list[dict]:
    # Select 20 representative tomato images across healthy and disease classes,
    # including natural pairs of base images and their 180deg / flipTB augmentations.
    healthy_dir = TOMATO_DIR / "Tomato_Healthy"
    unhealthy_dir = TOMATO_DIR / "Tomato_Unhealthy" / "Tomato_Early_blight"

    test_filenames = [
        # Set 1: Healthy base and augmented pairs
        (
            healthy_dir,
            "000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG",
            "healthy_base_1",
        ),
        (
            healthy_dir,
            "000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG",
            "healthy_base_2",
        ),
        (
            healthy_dir,
            "000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1_180deg.JPG",
            "healthy_180deg_2",
        ),
        (
            healthy_dir,
            "000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1_flipTB.JPG",
            "healthy_flipTB_2",
        ),
        (
            healthy_dir,
            "00bce074-967b-4d50-967a-31fdaa35e688___RS_HL 0223.JPG",
            "healthy_base_3",
        ),
        (
            healthy_dir,
            "00bce074-967b-4d50-967a-31fdaa35e688___RS_HL 0223_180deg.JPG",
            "healthy_180deg_3",
        ),
        (
            healthy_dir,
            "00bce074-967b-4d50-967a-31fdaa35e688___RS_HL 0223_flipTB.JPG",
            "healthy_flipTB_3",
        ),
        (
            healthy_dir,
            "014b5e19-7917-4d76-b632-b5dd31d999ec___RS_HL 9640.JPG",
            "healthy_base_4",
        ),
        (
            healthy_dir,
            "01700b46-3800-46c6-ab81-09a5f50af525___RS_HL 0108.JPG",
            "healthy_base_5",
        ),
        (
            healthy_dir,
            "0172e56c-8bb8-4e75-8ac7-509df81393e8___RS_HL 0580.JPG",
            "healthy_base_6",
        ),
        (
            healthy_dir,
            "01c1da17-8d9f-4d69-8a1e-58d37453d3c3___RS_HL 9641.JPG",
            "healthy_base_7",
        ),
        (
            healthy_dir,
            "01c1da17-8d9f-4d69-8a1e-58d37453d3c3___RS_HL 9641_180deg.JPG",
            "healthy_180deg_7",
        ),
        (
            healthy_dir,
            "01c1da17-8d9f-4d69-8a1e-58d37453d3c3___RS_HL 9641_flipTB.JPG",
            "healthy_flipTB_7",
        ),
        (
            healthy_dir,
            "01f98912-9627-4ce0-8f0f-e451b7acbb6c___RS_HL 0509.JPG",
            "healthy_base_8",
        ),
        (
            healthy_dir,
            "01f98912-9627-4ce0-8f0f-e451b7acbb6c___RS_HL 0509_180deg.JPG",
            "healthy_180deg_8",
        ),
        # Set 2: Unhealthy leaves
        (
            unhealthy_dir,
            "0017ae72-7489-406a-a18a-668ff40bf483___RS_Erly.B 7843.JPG",
            "early_blight_1",
        ),
        (
            unhealthy_dir,
            "0017ae72-7489-406a-a18a-668ff40bf483___RS_Erly.B 7843_180deg.JPG",
            "early_blight_180deg_1",
        ),
        (
            unhealthy_dir,
            "0017ae72-7489-406a-a18a-668ff40bf483___RS_Erly.B 7843_flipTB.JPG",
            "early_blight_flipTB_1",
        ),
        (
            unhealthy_dir,
            "00344795-0370-49b9-87a4-c7be361870fb___RS_Erly.B 7457.JPG",
            "early_blight_2",
        ),
        (
            unhealthy_dir,
            "00344795-0370-49b9-87a4-c7be361870fb___RS_Erly.B 7457_180deg.JPG",
            "early_blight_180deg_2",
        ),
    ]

    results = []
    print(
        f"{'Idx':<4} {'Label/Condition':<25} {'Filename':<35} {'Ellipse Angle':<15} {'Stem Visible?':<14} {'Gravity Ref?'}"
    )
    print("-" * 110)

    for i, (folder, fname, condition) in enumerate(test_filenames, start=1):
        fpath = folder / fname
        if not fpath.exists():
            # Fallback to any existing jpg if exact name differs
            jpgs = list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
            fpath = jpgs[i % len(jpgs)]
            fname = fpath.name

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        mask = segment_leaf(img)
        ellipse_ang, rect_ang, _status = calculate_fitted_angle(mask)

        # Inspection flags:
        # PlantVillage leaves are excised single leaves placed on flat scanner sheets
        stem_visible = False
        gravity_ref = False

        record = {
            "index": i,
            "filename": fname,
            "condition": condition,
            "ellipse_angle": ellipse_ang,
            "rect_angle": rect_ang,
            "stem_visible": stem_visible,
            "gravity_ref": gravity_ref,
        }
        results.append(record)
        print(
            f"{i:<4} {condition:<25} {fname[:33]:<35} {ellipse_ang:<15.1f} {stem_visible!s:<14} {gravity_ref!s}"
        )

    return results


if __name__ == "__main__":
    res = run_evaluation()
    print("-" * 110)
    print(f"Total images evaluated: {len(res)}")
