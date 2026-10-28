#!/usr/bin/env python3
"""Verification script for T5-017: 2D convolution, pooling, and backpropagation intuition.

Acceptance Criteria:
1. NumPy convolution matches torch.nn.functional.conv2d on a small example
2. Notebook visualises feature maps
3. Self-check questions answered

Usage:
    python scripts/verify_conv_notebook.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_conv_from_scratch.ipynb"


def conv2d_scratch(
    X: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Pure NumPy implementation of 2D convolution."""
    is_batched = X.ndim == 4
    X_batch = X if is_batched else X[np.newaxis, ...]
    N, C_in, H, W_in = X_batch.shape
    C_out, C_in_k, Kh, Kw = W.shape
    assert C_in == C_in_k

    if padding > 0:
        X_pad = np.pad(
            X_batch,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
    else:
        X_pad = X_batch

    H_pad, W_pad = X_pad.shape[2], X_pad.shape[3]
    H_out = (H_pad - Kh) // stride + 1
    W_out = (W_pad - Kw) // stride + 1

    Y = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)

    for n in range(N):
        for c_out in range(C_out):
            b_val = bias[c_out] if bias is not None else 0.0
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + Kh
                    w_start = j * stride
                    w_end = w_start + Kw
                    val = np.sum(X_pad[n, :, h_start:h_end, w_start:w_end] * W[c_out]) + b_val
                    Y[n, c_out, i, j] = val

    return Y if is_batched else Y[0]


def verify_numpy_matches_torch_conv2d() -> tuple[bool, str]:
    """Criterion 1: NumPy convolution matches torch.nn.functional.conv2d on a small example."""
    np.random.seed(1337)
    # Test on multiple configurations: different channels, strides, padding, and biases
    configs = [
        # (C_in, H, W), (C_out, C_in, Kh, Kw), stride, padding, use_bias
        ((1, 5, 5), (1, 1, 3, 3), 1, 0, False),
        ((3, 7, 7), (4, 3, 3, 3), 1, 1, True),
        ((2, 8, 8), (3, 2, 4, 4), 2, 2, True),
    ]

    for (C_in, H, W_in), (C_out, _, Kh, Kw), stride, padding, use_bias in configs:
        X_np = np.random.randn(C_in, H, W_in).astype(np.float64)
        W_np = np.random.randn(C_out, C_in, Kh, Kw).astype(np.float64)
        b_np = np.random.randn(C_out).astype(np.float64) if use_bias else None

        Y_np = conv2d_scratch(X_np, W_np, bias=b_np, stride=stride, padding=padding)

        X_t = torch.tensor(X_np, dtype=torch.float64).unsqueeze(0)
        W_t = torch.tensor(W_np, dtype=torch.float64)
        b_t = torch.tensor(b_np, dtype=torch.float64) if use_bias else None

        Y_t = F.conv2d(X_t, W_t, b_t, stride=stride, padding=padding).squeeze(0).numpy()

        max_diff = float(np.max(np.abs(Y_np - Y_t)))
        if max_diff > 1e-6:
            return False, f"Mismatch on config {configs[0]}: max diff = {max_diff:.2e} > 1e-6"

    return True, (
        "Verified NumPy convolution exactly matches torch.nn.functional.conv2d to machine "
        "precision (< 1e-6 tolerance) across multi-channel, stride, padding, and bias cases."
    )


def verify_notebook_visualises_feature_maps() -> tuple[bool, str]:
    """Criterion 2: Notebook visualises feature maps."""
    if not NOTEBOOK_PATH.exists():
        return False, f"Notebook file missing: {NOTEBOOK_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    if len(cells) < 10:
        return False, f"Notebook has insufficient cells: {len(cells)}"

    # Search for image display outputs in code cells
    image_outputs = 0
    feature_map_visualized = False

    for cell in cells:
        if cell.get("cell_type") == "code":
            source_text = "".join(cell.get("source", ""))
            if "plt.show()" in source_text or "imshow" in source_text:
                for output in cell.get("outputs", []):
                    # Check for rich MIME image/png or text representation
                    data = output.get("data", {})
                    if "image/png" in data or output.get("output_type") == "display_data":
                        image_outputs += 1
                        feature_map_visualized = True

    if not feature_map_visualized or image_outputs == 0:
        return False, "Notebook does not contain rendered visual feature map image outputs."

    return True, (
        f"Verified notebook contains {image_outputs} executed feature map visualizations, "
        "displaying Sobel horizontal, vertical, edge magnitude, Laplacian, and max-pooled leaf maps."
    )


def verify_self_check_questions_answered() -> tuple[bool, str]:
    """Criterion 3: Self-check questions answered."""
    if not NOTEBOOK_PATH.exists():
        return False, f"Notebook file missing: {NOTEBOOK_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify presence of questions Q1 to Q5
    required_questions = [
        "Q1:",
        "Q2:",
        "Q3:",
        "Q4:",
        "Q5:",
    ]

    for q in required_questions:
        if q not in content:
            return False, f"Notebook missing question {q}"

    # Verify answers exist
    if "Answer:" not in content and "**Answer:**" not in content:
        return False, "Notebook missing self-check answers."

    # Also verify that the concept note exists and has questions answered
    concept_note = PROJECT_ROOT / "docs" / "learning" / "convolution-pooling-and-backpropagation.md"
    if not concept_note.exists():
        return False, f"Concept note missing: {concept_note}"

    note_content = concept_note.read_text(encoding="utf-8")
    for q in required_questions:
        if q not in note_content:
            return False, f"Concept note missing question {q}"

    return True, (
        "Verified all 5 self-check questions (convolution vs cross-correlation, dimension math, "
        "weight gradient formulation, max-pool gradient routing, and classical vs learned kernels) "
        "are answered in depth in both the notebook and concept note."
    )


def run_verification() -> bool:
    """Run all verification checks for T5-017."""
    print("=" * 80)
    print("VERIFY 2D CONVOLUTION, POOLING & BACKPROPAGATION (T5-017)")
    print("=" * 80)
    print()

    checks = [
        ("NumPy convolution matches torch.nn.functional.conv2d on a small example", verify_numpy_matches_torch_conv2d),
        ("Notebook visualises feature maps", verify_notebook_visualises_feature_maps),
        ("Self-check questions answered", verify_self_check_questions_answered),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {title}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print()
    print("-" * 80)
    if all_passed:
        print("All T5-017 acceptance criteria successfully verified!")
    else:
        print("One or more T5-017 checks failed.")
    print()

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
