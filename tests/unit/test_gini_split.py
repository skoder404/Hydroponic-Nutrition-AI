"""Unit tests for from-scratch Gini impurity and greedy best-split (T4-003).

Acceptance Criteria:
1. Your threshold equals scikit-learn's (or the difference is explained).
2. Notebook includes a hand-worked example.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from scripts.verify_gini_split import (
    gini_impurity,
    greedy_best_split,
)

ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# Gini Impurity Function Unit Tests
# ==============================================================================


def test_gini_impurity_empty_array() -> None:
    """Empty array should return 0.0 impurity."""
    assert gini_impurity(np.array([])) == 0.0


def test_gini_impurity_pure_nodes() -> None:
    """Pure nodes containing a single class should have 0.0 Gini impurity."""
    assert gini_impurity(np.array(["A", "A", "A", "A"])) == 0.0
    assert gini_impurity(np.array([1, 1, 1])) == 0.0


def test_gini_impurity_binary_uniform() -> None:
    """Balanced binary node should have Gini impurity = 0.5."""
    y = np.array(["A", "A", "B", "B"])
    assert np.isclose(gini_impurity(y), 0.5)


def test_gini_impurity_multi_class_uniform() -> None:
    """K-class uniform distribution should have Gini = 1 - 1/K."""
    # 4 classes, equal representation: 1 - 1/4 = 0.75
    y = np.array([0, 1, 2, 3] * 10)
    assert np.isclose(gini_impurity(y), 0.75)

    # 7 classes: 1 - 1/7 = 6/7 = 0.857142857...
    y7 = np.array(list(range(7)) * 10)
    assert np.isclose(gini_impurity(y7), 6.0 / 7.0)


# ==============================================================================
# Hand-Worked Example Test (Criterion 2)
# ==============================================================================


def test_hand_worked_arithmetic_example() -> None:
    """Verify manual calculation: 6 samples with pH values [5.2, 5.4, 5.6, 6.0, 6.2, 6.8]."""
    X = np.array([5.2, 5.4, 5.6, 6.0, 6.2, 6.8]).reshape(-1, 1)
    y = np.array(["S", "S", "S", "H", "H", "H"])

    parent_g = gini_impurity(y)
    assert np.isclose(parent_g, 0.5)

    split = greedy_best_split(X, y, ["pH"])
    assert np.isclose(split["best_threshold"], 5.8)
    assert np.isclose(split["left_gini"], 0.0)
    assert np.isclose(split["right_gini"], 0.0)
    assert np.isclose(split["best_gain"], 0.5)


# ==============================================================================
# Parity with scikit-learn on Lettuce Training Data (Criterion 1)
# ==============================================================================


def test_parity_with_sklearn_stump_on_lettuce() -> None:
    """Verify that from-scratch greedy split threshold equals scikit-learn's DecisionTreeClassifier."""
    data_path = ROOT / "processed_data" / "sensor" / "lettuce" / "train.csv"
    if not data_path.exists():
        pytest.skip(f"Dataset missing: {data_path}")

    df = pd.read_csv(data_path)
    features = ["ph", "ec"]
    X = df[features].values
    y = df["label"].values

    scratch = greedy_best_split(X, y, features)

    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(X, y)

    sk_feat_idx = int(clf.tree_.feature[0])
    sk_feat_name = features[sk_feat_idx]
    sk_threshold = float(clf.tree_.threshold[0])
    sk_parent_gini = float(clf.tree_.impurity[0])
    sk_left_gini = float(clf.tree_.impurity[1])
    sk_right_gini = float(clf.tree_.impurity[2])

    assert scratch["best_feature"] == sk_feat_name
    assert scratch["best_feature"] == "ph"
    assert np.isclose(scratch["best_threshold"], sk_threshold, atol=1e-5)
    assert np.isclose(scratch["best_threshold"], 5.45)
    assert np.isclose(scratch["parent_gini"], sk_parent_gini, atol=1e-5)
    assert np.isclose(scratch["left_gini"], sk_left_gini, atol=1e-5)
    assert np.isclose(scratch["right_gini"], sk_right_gini, atol=1e-5)


# ==============================================================================
# Notebook Structure & Content Checks
# ==============================================================================


def test_notebook_structure_and_completeness() -> None:
    """Verify that notebooks/02_gini_from_scratch.ipynb exists and is valid JSON."""
    nb_path = ROOT / "notebooks" / "02_gini_from_scratch.ipynb"
    assert nb_path.exists()

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 6

    content = json.dumps(nb)
    assert "Gini Impurity" in content
    assert "Hand-Worked Arithmetic Example" in content
    assert "find_best_split" in content or "greedy_best_split" in content
    assert "DecisionTreeClassifier" in content
