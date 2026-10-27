"""Unit test for color spaces and vegetation indices calculations and notebook content."""

import json
from pathlib import Path

import cv2
import numpy as np


def test_notebook_exists_and_executed():
    nb_path = Path("notebooks/04_color_spaces.ipynb")
    assert nb_path.exists(), "Notebook 04_color_spaces.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 10

    # Ensure code cells have execution outputs
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    has_outputs = any(len(c.get("outputs", [])) > 0 for c in code_cells)
    assert has_outputs, "Notebook code cells should contain executed outputs"


def test_bgr_vs_rgb_exg_and_exr_math():
    # Synthetic leaf patch: strong green channel, moderate red, low blue
    leaf_rgb = np.array([[[40, 160, 20]]], dtype=np.uint8)
    leaf_bgr = cv2.cvtColor(leaf_rgb, cv2.COLOR_RGB2BGR)

    rgb_f = leaf_rgb.astype(np.float32) / 255.0
    tot_rgb = np.sum(rgb_f, axis=2, keepdims=True) + 1e-6
    rgb_norm = rgb_f / tot_rgb
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    exg_rgb = 2 * g - r - b
    exr_rgb = 1.4 * r - g

    bgr_f = leaf_bgr.astype(np.float32) / 255.0
    tot_bgr = np.sum(bgr_f, axis=2, keepdims=True) + 1e-6
    bgr_norm = bgr_f / tot_bgr
    r_bad, g_bad, b_bad = bgr_norm[:, :, 0], bgr_norm[:, :, 1], bgr_norm[:, :, 2]

    exg_bad = 2 * g_bad - r_bad - b_bad
    exr_bad = 1.4 * r_bad - g_bad

    # ExG is invariant to R-B swap
    np.testing.assert_allclose(exg_rgb, exg_bad, atol=1e-5)

    # ExR is NOT invariant and corrupts drastically
    assert not np.isclose(exr_rgb, exr_bad)
    # Since r_bad is actually blue, and blue is minimal (20/220 ~ 0.09), exr_bad is deeply negative
    assert exr_bad.item() < 0.0


def test_opencv_hue_halved_range():
    # In HSV:
    # Yellow = 60 degrees in standard color wheel -> 30 in OpenCV uint8
    yellow = np.array([[[255, 255, 0]]], dtype=np.uint8)
    yellow_hsv = cv2.cvtColor(yellow, cv2.COLOR_RGB2HSV)
    assert yellow_hsv[0, 0, 0] == 30

    # Green = 120 degrees standard -> 60 in OpenCV uint8
    green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    green_hsv = cv2.cvtColor(green, cv2.COLOR_RGB2HSV)
    assert green_hsv[0, 0, 0] == 60

    # Cyan = 180 degrees standard -> 90 in OpenCV uint8
    cyan = np.array([[[0, 255, 255]]], dtype=np.uint8)
    cyan_hsv = cv2.cvtColor(cyan, cv2.COLOR_RGB2HSV)
    assert cyan_hsv[0, 0, 0] == 90
