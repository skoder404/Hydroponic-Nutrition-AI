"""Unit tests for from-scratch 2D convolution, pooling, and backpropagation (T5-017)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.verify_conv_notebook import (
    NOTEBOOK_PATH,
    conv2d_scratch,
    run_verification,
    verify_notebook_visualises_feature_maps,
    verify_numpy_matches_torch_conv2d,
    verify_self_check_questions_answered,
)


def test_conv_verification_script():
    """Verify all T5-017 criteria pass via verification script."""
    assert run_verification() is True


def test_conv2d_scratch_exactness():
    """Verify NumPy 2D convolution matches PyTorch conv2d across shapes."""
    np.random.seed(42)
    # 1. Single channel, 3x3 kernel, stride 1, padding 0
    X1 = np.random.randn(1, 6, 6).astype(np.float64)
    W1 = np.random.randn(1, 1, 3, 3).astype(np.float64)
    y1_np = conv2d_scratch(X1, W1, stride=1, padding=0)
    y1_pt = F.conv2d(torch.tensor(X1).unsqueeze(0), torch.tensor(W1), stride=1, padding=0).squeeze(0).numpy()
    np.testing.assert_allclose(y1_np, y1_pt, atol=1e-6)

    # 2. Multi-channel, 3 input, 4 output, stride 2, padding 1 with bias
    X2 = np.random.randn(3, 8, 8).astype(np.float64)
    W2 = np.random.randn(4, 3, 3, 3).astype(np.float64)
    b2 = np.random.randn(4).astype(np.float64)
    y2_np = conv2d_scratch(X2, W2, bias=b2, stride=2, padding=1)
    y2_pt = F.conv2d(torch.tensor(X2).unsqueeze(0), torch.tensor(W2), torch.tensor(b2), stride=2, padding=1).squeeze(0).numpy()
    np.testing.assert_allclose(y2_np, y2_pt, atol=1e-6)


def test_notebook_structure_and_executed_visualizations():
    """Verify notebook exists, is valid nbformat v4, and contains executed image outputs."""
    assert NOTEBOOK_PATH.exists(), f"Notebook {NOTEBOOK_PATH} does not exist"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 10

    # Ensure code cells have outputs
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert any(len(c.get("outputs", [])) > 0 for c in code_cells)

    ok, msg = verify_notebook_visualises_feature_maps()
    assert ok, msg


def test_self_check_questions_and_concept_note():
    """Verify self-check questions and learning concept note."""
    ok, msg = verify_self_check_questions_answered()
    assert ok, msg

    concept_note = Path("docs/learning/convolution-pooling-and-backpropagation.md")
    assert concept_note.exists()
    content = concept_note.read_text(encoding="utf-8")
    assert "## 1. Intuition" in content
    assert "## 2. Mathematical Formulation & Formal Definition" in content
    assert "## 3. From-Scratch Implementation (NumPy / Pure Python)" in content
    assert "## 7. Self-Check Questions (3–5 Questions)" in content
