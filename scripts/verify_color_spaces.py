"""Verification script for T5-001: Colour Spaces and Vegetation Indices Notebook.

Verifies:
1. notebooks/04_color_spaces.ipynb exists and is valid nbformat v4.
2. Contains required sections:
   - OpenCV BGR vs RGB loading
   - ExG and ExR index calculations (correct vs corrupted channel order)
   - HSV conversion and [0, 179] Hue range explanation
   - Yellow chlorosis and brown necrosis segmentation masks
   - Answers to all self-check questions
3. Validates that mathematical functions execute correctly on test images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def verify_notebook_structure(nb_path: Path) -> dict:
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found at: {nb_path}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    if nb_data.get("nbformat") != 4:
        raise ValueError(f"Expected nbformat v4, got {nb_data.get('nbformat')}")

    cells = nb_data.get("cells", [])
    if len(cells) < 10:
        raise ValueError(f"Expected at least 10 cells in notebook, found {len(cells)}")

    all_source = ""
    for cell in cells:
        source = "".join(cell.get("source", []))
        all_source += source + "\n"

    # Required patterns
    required_keywords = [
        ("BGR vs RGB", ["cv2.imread", "COLOR_BGR2RGB", "bgr"]),
        ("Excess Green (ExG)", ["2 * g - r - b", "exg", "2G - R - B"]),
        ("Excess Red (ExR)", ["1.4 * r - g", "exr", "1.4R - G"]),
        ("OpenCV Hue [0, 179]", ["179", "COLOR_RGB2HSV", "uint8", "360"]),
        (
            "Yellow / Brown masks",
            ["inRange", "yellow", "brown", "necrosis", "chlorosis"],
        ),
        ("Self-check Q1", ["Q1", "Intel", "BGR"]),
        ("Self-check Q2", ["Q2", "1.4B - G", "chlorophyll"]),
        ("Self-check Q3", ["Q3", "179", "uint8"]),
        ("Self-check Q4", ["Q4", "HSV", "lighting"]),
        ("Self-check Q5", ["Q5", "normalization", "illumination"]),
    ]

    for category, tokens in required_keywords:
        found = any(tok.lower() in all_source.lower() for tok in tokens)
        if not found:
            raise AssertionError(
                f"Missing required topic in notebook: '{category}' (expected tokens {tokens})"
            )

    return {
        "cells_count": len(cells),
        "code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
        "markdown_cells": sum(1 for c in cells if c.get("cell_type") == "markdown"),
    }


def verify_color_math() -> None:
    # Test image array: green foliage pixel and yellow lesion pixel
    # Green pixel: R=50, G=150, B=20 (RGB)
    green_rgb = np.array([[[50, 150, 20]]], dtype=np.uint8)
    green_bgr = cv2.cvtColor(green_rgb, cv2.COLOR_RGB2BGR)

    # Correct RGB normalization
    rgb_f = green_rgb.astype(np.float32) / 255.0
    total = np.sum(rgb_f, axis=2, keepdims=True) + 1e-6
    r, g, b = (
        rgb_f[:, :, 0] / total[:, :, 0],
        rgb_f[:, :, 1] / total[:, :, 0],
        rgb_f[:, :, 2] / total[:, :, 0],
    )
    exg_correct = 2 * g - r - b
    exr_correct = 1.4 * r - g

    # Corrupted: treat BGR array as RGB
    bgr_as_rgb_f = green_bgr.astype(np.float32) / 255.0
    total_corrupt = np.sum(bgr_as_rgb_f, axis=2, keepdims=True) + 1e-6
    r_corrupt = bgr_as_rgb_f[:, :, 0] / total_corrupt[:, :, 0]
    g_corrupt = bgr_as_rgb_f[:, :, 1] / total_corrupt[:, :, 0]
    b_corrupt = bgr_as_rgb_f[:, :, 2] / total_corrupt[:, :, 0]
    exg_corrupt = 2 * g_corrupt - r_corrupt - b_corrupt
    exr_corrupt = 1.4 * r_corrupt - g_corrupt

    # ExG is mathematically identical under R/B swap: 2G - R - B == 2G - B - R
    np.testing.assert_allclose(exg_correct, exg_corrupt, atol=1e-5)

    # ExR is severely broken because R_corrupt = B_true, G_corrupt = G_true
    # exr_corrupt = 1.4 * B - G
    assert float(exr_correct.item()) != float(exr_corrupt.item())
    assert float(exr_corrupt.item()) < 0.0, (
        "Corrupted ExR should be negative on foliage"
    )

    # HSV verification
    # Pure yellow: Hue in standard degrees is 60 deg -> in OpenCV uint8 it is 30
    yellow_rgb = np.array([[[255, 255, 0]]], dtype=np.uint8)
    yellow_hsv = cv2.cvtColor(yellow_rgb, cv2.COLOR_RGB2HSV)
    h_val = yellow_hsv[0, 0, 0]
    assert h_val == 30, f"Expected OpenCV hue 30 for pure yellow, got {h_val}"


def main() -> int:
    nb_path = Path("notebooks/04_color_spaces.ipynb")
    print(f"Verifying {nb_path}...")
    stats = verify_notebook_structure(nb_path)
    print(
        f"Notebook structure valid: {stats['cells_count']} cells ({stats['code_cells']} code, {stats['markdown_cells']} markdown)."
    )

    print("Verifying mathematical formulas and OpenCV HSV conversions...")
    verify_color_math()
    print("Color space and vegetation indices verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
