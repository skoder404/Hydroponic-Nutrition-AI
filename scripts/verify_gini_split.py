#!/usr/bin/env python3
"""Verification script for from-scratch Gini impurity and greedy best-split (T4-003).

Acceptance Criteria:
1. Your threshold equals scikit-learn's (or the difference is explained).
2. Notebook includes a hand-worked example.
3. Verification can be independently executed via this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass


def gini_impurity(y: np.ndarray) -> float:
    """Calculate Gini impurity for an array of discrete class labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs**2))


def greedy_best_split(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, Any]:
    """Find the single greedy best split threshold across all features maximizing Gini gain."""
    n_samples = len(y)
    parent_gini = gini_impurity(y)
    best_gain = -1.0
    best_feat_idx = -1
    best_threshold = 0.0
    best_left_gini = 0.0
    best_right_gini = 0.0
    best_left_count = 0
    best_right_count = 0

    for feat_idx in range(X.shape[1]):
        vals = X[:, feat_idx]
        unique_vals = np.unique(vals)
        if len(unique_vals) <= 1:
            continue

        # Candidate thresholds are midpoints between sorted adjacent distinct values
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for t in thresholds:
            left_mask = vals <= t
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            g_left = gini_impurity(y_left)
            g_right = gini_impurity(y_right)

            weighted_child_gini = (len(y_left) / n_samples) * g_left + (
                len(y_right) / n_samples
            ) * g_right
            gain = parent_gini - weighted_child_gini

            # Greedy maximization
            if gain > best_gain:
                best_gain = gain
                best_feat_idx = feat_idx
                best_threshold = float(t)
                best_left_gini = g_left
                best_right_gini = g_right
                best_left_count = len(y_left)
                best_right_count = len(y_right)

    return {
        "best_feature": feature_names[best_feat_idx],
        "best_feature_idx": best_feat_idx,
        "best_threshold": best_threshold,
        "best_gain": best_gain,
        "parent_gini": parent_gini,
        "left_gini": best_left_gini,
        "right_gini": best_right_gini,
        "left_count": best_left_count,
        "right_count": best_right_count,
    }


def verify_criterion_2_hand_worked_example() -> None:
    """Verify that the hand-worked arithmetic example computes correct Gini impurities and split."""
    # 6 samples: 3 Stress (S) with pH [5.2, 5.4, 5.6], 3 Healthy (H) with pH [6.0, 6.2, 6.8]
    pH = np.array([5.2, 5.4, 5.6, 6.0, 6.2, 6.8]).reshape(-1, 1)
    labels = np.array(["S", "S", "S", "H", "H", "H"])

    # 1. Parent Gini: 1 - ((3/6)^2 + (3/6)^2) = 1 - 0.5 = 0.5
    parent_g = gini_impurity(labels)
    assert np.isclose(parent_g, 0.5), f"Expected parent Gini 0.5, got {parent_g}"

    # 2. Threshold between 5.6 and 6.0 is 5.8
    res = greedy_best_split(pH, labels, ["pH"])
    assert np.isclose(res["best_threshold"], 5.8), (
        f"Expected threshold 5.8, got {res['best_threshold']}"
    )
    assert np.isclose(res["left_gini"], 0.0), "Left node should be pure"
    assert np.isclose(res["right_gini"], 0.0), "Right node should be pure"
    assert np.isclose(res["best_gain"], 0.5), (
        f"Expected gain 0.5, got {res['best_gain']}"
    )

    print("[PASS] Criterion 2: Hand-worked arithmetic example verified.")
    print(
        f"       Parent Gini={parent_g:.3f}, Chosen Threshold={res['best_threshold']}, Split Gain={res['best_gain']:.3f}"
    )


def verify_criterion_1_parity_with_sklearn() -> None:
    """Verify that from-scratch threshold on 2-feature lettuce training data matches scikit-learn."""
    data_path = ROOT / "processed_data" / "sensor" / "lettuce" / "train.csv"
    assert data_path.exists(), f"Lettuce train dataset not found at {data_path}"

    df = pd.read_csv(data_path)
    feature_names = ["ph", "ec"]
    X = df[feature_names].values
    y = df["label"].values

    # 1. From-scratch computation
    scratch = greedy_best_split(X, y, feature_names)

    # 2. scikit-learn DecisionTreeClassifier(max_depth=1)
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(X, y)

    sk_feat_idx = int(clf.tree_.feature[0])
    sk_feat_name = feature_names[sk_feat_idx]
    sk_threshold = float(clf.tree_.threshold[0])
    sk_parent_gini = float(clf.tree_.impurity[0])
    sk_left_gini = float(clf.tree_.impurity[1])
    sk_right_gini = float(clf.tree_.impurity[2])

    print("\n--- Model Parity Comparison on Lettuce (ph, ec) ---")
    print(
        f"From-scratch best feature : {scratch['best_feature']} (idx: {scratch['best_feature_idx']})"
    )
    print(f"scikit-learn best feature : {sk_feat_name} (idx: {sk_feat_idx})")
    print(f"From-scratch threshold    : {scratch['best_threshold']:.6f}")
    print(f"scikit-learn threshold    : {sk_threshold:.6f}")
    print(f"From-scratch parent Gini  : {scratch['parent_gini']:.6f}")
    print(f"scikit-learn parent Gini  : {sk_parent_gini:.6f}")
    print(f"From-scratch left Gini    : {scratch['left_gini']:.6f}")
    print(f"scikit-learn left Gini    : {sk_left_gini:.6f}")
    print(f"From-scratch right Gini   : {scratch['right_gini']:.6f}")
    print(f"scikit-learn right Gini   : {sk_right_gini:.6f}")

    # Assertions
    assert scratch["best_feature"] == sk_feat_name, (
        "Selected split feature does not match scikit-learn!"
    )
    # scikit-learn internally casts thresholds to float32 (5.450000047683716 vs 5.45)
    assert np.isclose(scratch["best_threshold"], sk_threshold, atol=1e-5), (
        f"Threshold mismatch: {scratch['best_threshold']} vs {sk_threshold}"
    )
    assert np.isclose(scratch["parent_gini"], sk_parent_gini, atol=1e-5), (
        "Parent Gini mismatch!"
    )
    assert np.isclose(scratch["left_gini"], sk_left_gini, atol=1e-5), (
        "Left child Gini mismatch!"
    )
    assert np.isclose(scratch["right_gini"], sk_right_gini, atol=1e-5), (
        "Right child Gini mismatch!"
    )

    print(
        "\n[PASS] Criterion 1: From-scratch threshold equals scikit-learn's exact decision stump threshold (5.45)."
    )


def verify_notebook_file_exists() -> None:
    """Verify that notebooks/02_gini_from_scratch.ipynb exists and is valid JSON."""
    nb_path = ROOT / "notebooks" / "02_gini_from_scratch.ipynb"
    assert nb_path.exists(), f"Notebook file missing at '{nb_path}'"

    with open(nb_path, encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, "Invalid notebook format: missing 'cells'"
    assert len(nb_data["cells"]) >= 6, (
        "Notebook should have at least 6 cells covering all steps"
    )

    content_str = json.dumps(nb_data)
    assert "gini_impurity" in content_str
    assert "greedy_best_split" in content_str
    assert "hand-worked" in content_str.lower()
    assert "DecisionTreeClassifier" in content_str

    print(
        "[PASS] Notebook format verified: notebooks/02_gini_from_scratch.ipynb is valid and complete."
    )


def main() -> None:
    """Run all T4-003 verifications."""
    print("=" * 70)
    print("T4-003 Verification: Gini Impurity and Best-Split From Scratch")
    print("=" * 70)

    verify_criterion_2_hand_worked_example()
    verify_criterion_1_parity_with_sklearn()
    verify_notebook_file_exists()

    print("\nALL CRITERIA PASSED! T4-003 is fully verified.")


if __name__ == "__main__":
    main()
