"""Unit tests for grouped cross-validation with in-fold oversampling (T4-007, T8-004).

Acceptance Criteria:
- [x] Group overlap raises an error (GroupOverlapError).
- [x] Oversampling never touches the validation fold (raw distributions preserved).
- [x] Unit test cases listed for T8 tasks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.evaluation import (
    CVValidationError,
    GroupOverlapError,
    GuardedGroupKFold,
    assert_no_group_overlap,
    evaluate_grouped_cv,
    get_fold_data,
    oversample_minority_classes,
)


def test_assert_no_group_overlap_raises_on_leakage() -> None:
    """Criterion 1: Group overlap raises GroupOverlapError."""
    train_groups = ["day_1", "day_2", "day_3", "day_4"]
    val_groups = ["day_4", "day_5"]  # 'day_4' overlaps!

    with pytest.raises(GroupOverlapError, match="Group overlap detected"):
        assert_no_group_overlap(train_groups, val_groups, fold_idx=1)


def test_assert_no_group_overlap_passes_on_disjoint_groups() -> None:
    """Disjoint group identifiers pass without exception."""
    train_groups = ["day_1", "day_2", "day_3"]
    val_groups = ["day_4", "day_5"]
    assert_no_group_overlap(train_groups, val_groups, fold_idx=0)


def test_guarded_group_kfold_split_enforces_group_separation() -> None:
    """GuardedGroupKFold partitions data cleanly across groups without leakage."""
    n_samples = 100
    groups = np.repeat([f"batch_{i}" for i in range(10)], 10)  # 10 batches of 10
    X = np.random.randn(n_samples, 4)
    y = np.random.choice(["Normal", "EC_High"], size=n_samples)

    cv = GuardedGroupKFold(n_splits=5)
    splits = list(cv.split(X, y, groups=groups))

    assert len(splits) == 5
    for fold_idx, (t_idx, v_idx) in enumerate(splits):
        t_groups = set(groups[t_idx])
        v_groups = set(groups[v_idx])
        assert len(t_groups.intersection(v_groups)) == 0, f"Overlap in fold {fold_idx}"
        assert len(t_idx) + len(v_idx) == n_samples


def test_guarded_group_kfold_validation_errors() -> None:
    """Precondition violations raise CVValidationError."""
    cv = GuardedGroupKFold(n_splits=5)
    X = np.ones((10, 2))

    # groups is None
    with pytest.raises(CVValidationError, match="groups cannot be None"):
        list(cv.split(X, groups=None))

    # mismatched length
    with pytest.raises(CVValidationError, match="does not match sample count"):
        list(cv.split(X, groups=["g1", "g2"]))

    # fewer groups than n_splits
    with pytest.raises(CVValidationError, match="less than n_splits"):
        list(cv.split(X, groups=["g1"] * 5 + ["g2"] * 5))


def test_oversample_minority_classes_balances_training_fold() -> None:
    """In-fold oversampler equalizes minority class counts in training data."""
    # Imbalanced setup: Normal has 50 rows, pH_Low has 5 rows
    df = pd.DataFrame(
        {
            "feature_1": np.arange(55),
            "feature_2": np.arange(55) * 2,
        }
    )
    y = pd.Series(["Normal"] * 50 + ["pH_Low"] * 5)

    resampled_X, resampled_y = oversample_minority_classes(
        df,
        y,
        target_count="max",
        random_state=42,
    )

    counts = resampled_y.value_counts()
    assert counts["Normal"] == 50
    assert counts["pH_Low"] == 50
    assert len(resampled_X) == 100
    assert isinstance(resampled_X, pd.DataFrame)
    assert list(resampled_X.columns) == ["feature_1", "feature_2"]


def test_oversampling_never_touches_validation_fold() -> None:
    """Criterion 2: Oversampling never touches the validation fold."""
    # Synthetic dataset with severe class imbalance
    # Group A (30 rows Normal), Group B (20 rows Normal), Group C (5 rows pH_Low)
    records = []
    groups = []
    targets = []

    for i in range(30):
        records.append({"val": float(i), "id": f"A_{i}"})
        groups.append("Group_A")
        targets.append("Normal")

    for i in range(20):
        records.append({"val": float(100 + i), "id": f"B_{i}"})
        groups.append("Group_B")
        targets.append("Normal")

    for i in range(5):
        records.append({"val": float(200 + i), "id": f"C_{i}"})
        groups.append("Group_C")
        targets.append("pH_Low")

    X = pd.DataFrame(records)
    y = pd.Series(targets)
    groups = np.array(groups)

    # Let Group_C (with 5 pH_Low rows) be the validation fold
    val_idx = np.where(groups == "Group_C")[0]
    train_idx = np.where(groups != "Group_C")[0]

    original_val_x = X.iloc[val_idx].copy()
    original_val_y = y.iloc[val_idx].copy()

    x_tr_resampled, y_tr_resampled, x_val_out, y_val_out = get_fold_data(
        X=X,
        y=y,
        groups=groups,
        train_idx=train_idx,
        val_idx=val_idx,
        oversample_train=True,
        random_state=42,
    )

    # Validation fold must be 100% identical to original unresampled slice
    assert len(x_val_out) == len(original_val_x) == 5
    assert len(y_val_out) == len(original_val_y) == 5
    assert (y_val_out.values == "pH_Low").all()
    pd.testing.assert_frame_equal(
        x_val_out.reset_index(drop=True), original_val_x.reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        y_val_out.reset_index(drop=True), original_val_y.reset_index(drop=True)
    )

    # And training fold must be balanced
    assert len(x_tr_resampled) == len(y_tr_resampled) == 50


def test_evaluate_grouped_cv_end_to_end() -> None:
    """Verify end-to-end grouped CV with feature scaling and macro-F1 evaluation."""
    rng = np.random.RandomState(42)
    n_samples = 120
    groups = np.repeat(
        [f"exp_day_{i}" for i in range(12)], 10
    )  # 12 days, 10 readings each
    labels = ["Normal", "EC_High", "pH_Low"]

    # Generate synthetic telemetry
    X = pd.DataFrame(
        {
            "ph": rng.uniform(5.5, 7.5, size=n_samples),
            "ec": rng.uniform(1.0, 3.0, size=n_samples),
            "temp": rng.uniform(18.0, 26.0, size=n_samples),
        }
    )
    y = pd.Series(rng.choice(labels, size=n_samples))

    clf = LogisticRegression(max_iter=100)
    scaler = StandardScaler()

    results = evaluate_grouped_cv(
        estimator=clf,
        X=X,
        y=y,
        groups=groups,
        labels=labels,
        n_splits=4,
        oversample_train=True,
        scaler=scaler,
        random_state=42,
    )

    assert "mean_macro_f1" in results
    assert "std_macro_f1" in results
    assert "mean_accuracy" in results
    assert len(results["fold_metrics"]) == 4
    assert results["n_splits"] == 4
    assert results["scaler_used"] is True
    assert 0.0 <= results["mean_macro_f1"] <= 1.0

    # Ensure fold metrics contain Section 18.2 required fields
    for fm in results["fold_metrics"]:
        assert "accuracy" in fm
        assert "macro_f1" in fm
        assert "per_class" in fm
        assert "confusion_matrix" in fm
