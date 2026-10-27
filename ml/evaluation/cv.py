"""Grouped and time-aware cross-validation with in-fold oversampling (T4-007).

Adheres to:
- Overview Section 18.4 (Cross-validation on correlated sensor time series).
- ADR 0006: Sensor Validation Protocol and Grouped Cross-Validation.
- Section 4.2: The Oversampling-Inside-Fold Rule.

Invariants:
1. Zero group overlap between training fold and validation fold:
   G(train_fold) ∩ G(val_fold) == ∅.
2. Oversampling and scaling never touch the validation fold:
   Validation folds remain strictly real, un-oversampled, un-interpolated distributions.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, GroupKFold

from ml.evaluation.metrics import calculate_metrics


class GroupOverlapError(ValueError):
    """Raised when any group appears in both the training and validation fold."""


class CVValidationError(ValueError):
    """Raised when cross-validation inputs violate structure or split preconditions."""


def assert_no_group_overlap(
    train_groups: Sequence[Any] | np.ndarray,
    val_groups: Sequence[Any] | np.ndarray,
    fold_idx: int | None = None,
) -> None:
    """Verify that train and validation folds share zero group identifiers.

    Parameters
    ----------
    train_groups:
        Group identifiers assigned to training fold samples.
    val_groups:
        Group identifiers assigned to validation fold samples.
    fold_idx:
        Optional fold index for descriptive error messaging.

    Raises
    ------
    GroupOverlapError:
        If set(train_groups) & set(val_groups) is non-empty.
    """
    train_set = set(train_groups)
    val_set = set(val_groups)
    overlap = train_set.intersection(val_set)

    if overlap:
        sorted_overlap = sorted(str(item) for item in overlap)
        fold_str = f" in fold {fold_idx}" if fold_idx is not None else ""
        raise GroupOverlapError(
            f"Group overlap detected{fold_str}! "
            f"{len(overlap)} group(s) appeared in both train and validation: {sorted_overlap[:10]}. "
            f"Per ADR 0006, grouped splits must be strictly disjoint: G(train) ∩ G(val) = ∅."
        )


class GuardedGroupKFold(BaseCrossValidator):
    """Guarded GroupKFold validator enforcing zero group leakage across all folds.

    Wraps scikit-learn's GroupKFold and performs an immutable assertion on each
    fold's partition to guarantee zero group overlap before yielding indices.
    """

    def __init__(self, n_splits: int = 5) -> None:
        if n_splits < 2:
            raise CVValidationError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self._gkf = GroupKFold(n_splits=n_splits)

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/val indices with strict group disjointness verification.

        Parameters
        ----------
        X:
            Feature data array or DataFrame.
        y:
            Target labels array or Series (optional).
        groups:
            Group keys array or Series matching length of X. Must not be None.

        Yields
        ------
        (train_indices, val_indices):
            Tuple of ndarray containing fold split indices.

        Raises
        ------
        CVValidationError:
            If groups is None, length differs from X, or unique groups < n_splits.
        GroupOverlapError:
            If any group ID appears in both train and validation folds.
        """
        if groups is None:
            raise CVValidationError("groups cannot be None for GuardedGroupKFold.")

        groups_arr = np.asarray(groups)
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]

        if len(groups_arr) != n_samples:
            raise CVValidationError(
                f"Length of groups ({len(groups_arr)}) does not match sample count ({n_samples})."
            )

        unique_groups = np.unique(groups_arr)
        if len(unique_groups) < self.n_splits:
            raise CVValidationError(
                f"Number of distinct groups ({len(unique_groups)}) is less than n_splits ({self.n_splits})."
            )

        for fold_idx, (train_idx, val_idx) in enumerate(
            self._gkf.split(X, y, groups=groups_arr)
        ):
            train_g = groups_arr[train_idx]
            val_g = groups_arr[val_idx]
            assert_no_group_overlap(train_g, val_g, fold_idx=fold_idx)
            yield train_idx, val_idx


def oversample_minority_classes(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence[Any],
    target_count: int | str = "max",
    random_state: int = 42,
    min_keep: int = 1,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    """Oversample minority classes with replacement inside a training fold.

    Guarantees:
    - Pure NumPy / pandas implementation requiring zero unpinned C-extensions.
    - Deterministic replication using explicit random_state seed.
    - Preserves column names, dtypes, and index structure if pandas inputs are provided.

    Parameters
    ----------
    X:
        Training feature matrix (pd.DataFrame or np.ndarray).
    y:
        Training target vector (pd.Series, np.ndarray, or sequence).
    target_count:
        'max' (default) to equalize all classes to majority class count,
        or an explicit integer count.
    random_state:
        Seed for deterministic random sampling.
    min_keep:
        Minimum count threshold for a class to be retained.

    Returns
    -------
    (X_resampled, y_resampled):
        Balanced training feature and target containers.
    """
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, pd.Series)

    if is_df:
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(np.asarray(X))

    y_series = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    if len(X_df) != len(y_series):
        raise CVValidationError(
            f"X and y must have identical length: got {len(X_df)} vs {len(y_series)}."
        )

    class_counts = y_series.value_counts()
    class_counts = class_counts[class_counts >= min_keep]

    if len(class_counts) == 0:
        raise CVValidationError("No classes satisfy min_keep threshold in y.")

    if target_count == "max":
        target_n = int(class_counts.max())
    elif isinstance(target_count, (int, np.integer)):
        target_n = int(target_count)
    else:
        raise ValueError(
            f"Invalid target_count: {target_count}. Expected 'max' or int."
        )

    rng = np.random.RandomState(random_state)
    parts_x: list[pd.DataFrame] = []
    parts_y: list[pd.Series] = []

    for cls, count in class_counts.items():
        mask = (y_series == cls).to_numpy()
        sub_x = X_df.iloc[mask]
        sub_y = y_series.iloc[mask]

        if count < target_n:
            needed = target_n - count
            draw_indices = rng.choice(np.arange(count), size=needed, replace=True)
            extra_x = sub_x.iloc[draw_indices]
            extra_y = sub_y.iloc[draw_indices]
            sub_x = pd.concat([sub_x, extra_x], axis=0)
            sub_y = pd.concat([sub_y, extra_y], axis=0)

        parts_x.append(sub_x)
        parts_y.append(sub_y)

    full_x = pd.concat(parts_x, axis=0)
    full_y = pd.concat(parts_y, axis=0)

    # Deterministic final shuffle of balanced fold
    shuffle_idx = rng.permutation(len(full_x))
    resampled_x = full_x.iloc[shuffle_idx].reset_index(drop=True)
    resampled_y = full_y.iloc[shuffle_idx].reset_index(drop=True)

    if is_df:
        final_x: pd.DataFrame | np.ndarray = resampled_x
    else:
        final_x = resampled_x.to_numpy()

    if is_series:
        final_y: pd.Series | np.ndarray = resampled_y
    else:
        final_y = resampled_y.to_numpy()

    return final_x, final_y


def get_fold_data(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence[Any],
    groups: Sequence[Any] | np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    oversample_train: bool = True,
    target_count: int | str = "max",
    random_state: int = 42,
) -> tuple[Any, Any, Any, Any]:
    """Slice fold data, assert zero group leakage, and optionally oversample training fold only.

    Guarantees:
    - Validation fold (X_val, y_val) is NEVER touched by oversampling.
    - Zero group overlap is verified before returning.

    Parameters
    ----------
    X:
        Full feature dataset.
    y:
        Full target labels.
    groups:
        Group identifiers for all rows.
    train_idx:
        Indices for the training fold.
    val_idx:
        Indices for the validation fold.
    oversample_train:
        Whether to oversample minority classes inside the training fold only.
    target_count:
        Target count per class for oversampling.
    random_state:
        Seed for deterministic oversampling.

    Returns
    -------
    (X_train_resampled, y_train_resampled, X_val_raw, y_val_raw):
        Fold partitions ready for model fitting and evaluation.
    """
    groups_arr = np.asarray(groups)
    assert_no_group_overlap(groups_arr[train_idx], groups_arr[val_idx])

    # Slice validation partition (strictly raw, untouched)
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
    else:
        X_arr = np.asarray(X)
        X_train = X_arr[train_idx].copy()
        X_val = X_arr[val_idx].copy()

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()
    else:
        y_arr = np.asarray(y)
        y_train = y_arr[train_idx].copy()
        y_val = y_arr[val_idx].copy()

    # Oversample training partition only
    if oversample_train:
        X_train_out, y_train_out = oversample_minority_classes(
            X_train,
            y_train,
            target_count=target_count,
            random_state=random_state,
        )
    else:
        X_train_out, y_train_out = X_train, y_train

    return X_train_out, y_train_out, X_val, y_val


def iterate_grouped_folds(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence[Any],
    groups: Sequence[Any] | np.ndarray,
    n_splits: int = 5,
    oversample_train: bool = True,
    random_state: int = 42,
) -> Generator[tuple[int, Any, Any, Any, Any], None, None]:
    """Iterate through grouped folds yielding disjoint train and validation slices.

    Yields
    ------
    (fold_idx, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
    splitter = GuardedGroupKFold(n_splits=n_splits)
    for fold_idx, (t_idx, v_idx) in enumerate(splitter.split(X, y, groups=groups)):
        x_tr, y_tr, x_v, y_v = get_fold_data(
            X=X,
            y=y,
            groups=groups,
            train_idx=t_idx,
            val_idx=v_idx,
            oversample_train=oversample_train,
            random_state=random_state + fold_idx,
        )
        yield fold_idx, x_tr, y_tr, x_v, y_v


def evaluate_grouped_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence[Any],
    groups: Sequence[Any] | np.ndarray,
    labels: Sequence[str | int],
    n_splits: int = 5,
    oversample_train: bool = True,
    scaler: Any | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run grouped cross-validation with in-fold preprocessing and metric computation.

    Adheres strictly to ADR 0006:
    - Preprocessing (scaler and oversampler) fit ONLY on training fold data.
    - Zero group overlap enforced on every fold.
    - Evaluation performed on raw, un-oversampled validation folds.

    Parameters
    ----------
    estimator:
        Scikit-learn compatible classifier implementing fit and predict.
    X:
        Training pool features.
    y:
        Training pool targets.
    groups:
        Group identifiers matching X.
    labels:
        Fixed class taxonomy order for metric calculation per Section 18.2.
    n_splits:
        Number of cross-validation folds.
    oversample_train:
        Whether to oversample minority classes inside each training fold.
    scaler:
        Optional feature scaler instance (e.g. StandardScaler) to fit on train only.
    random_state:
        Seed for reproducibility across folds.

    Returns
    -------
    dict[str, Any]:
        Dictionary containing:
        - "mean_macro_f1": float
        - "std_macro_f1": float
        - "mean_accuracy": float
        - "std_accuracy": float
        - "mean_macro_precision": float
        - "mean_macro_recall": float
        - "fold_metrics": list of per-fold metric dictionaries
        - "n_splits": int
        - "oversample_train": bool
        - "scaler_used": bool
    """
    fold_metrics_list: list[dict[str, Any]] = []

    for fold_idx, x_tr, y_tr, x_v, y_v in iterate_grouped_folds(
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits,
        oversample_train=oversample_train,
        random_state=random_state,
    ):
        model = clone(estimator)

        # Scale features if scaler provided (fit on train fold only!)
        if scaler is not None:
            sc = clone(scaler)
            if hasattr(x_tr, "values"):
                x_tr_fit = sc.fit_transform(x_tr.to_numpy())
                x_v_trans = sc.transform(x_v.to_numpy())
            else:
                x_tr_fit = sc.fit_transform(x_tr)
                x_v_trans = sc.transform(x_v)
        else:
            x_tr_fit = x_tr
            x_v_trans = x_v

        # Fit model on balanced training fold
        y_tr_fit = np.asarray(y_tr)
        model.fit(x_tr_fit, y_tr_fit)

        # Predict on untouched raw validation fold
        y_pred = model.predict(x_v_trans)
        y_v_arr = np.asarray(y_v)

        # Compute Section 18.2 metrics
        m = calculate_metrics(y_true=y_v_arr, y_pred=y_pred, labels=labels)
        m["fold"] = fold_idx
        m["train_samples"] = len(y_tr_fit)
        m["val_samples"] = len(y_v_arr)
        fold_metrics_list.append(m)

    macro_f1_scores = [f["macro_f1"] for f in fold_metrics_list]
    accuracy_scores = [f["accuracy"] for f in fold_metrics_list]
    precision_scores = [f["macro_precision"] for f in fold_metrics_list]
    recall_scores = [f["macro_recall"] for f in fold_metrics_list]

    return {
        "mean_macro_f1": float(np.mean(macro_f1_scores)),
        "std_macro_f1": float(np.std(macro_f1_scores)),
        "mean_accuracy": float(np.mean(accuracy_scores)),
        "std_accuracy": float(np.std(accuracy_scores)),
        "mean_macro_precision": float(np.mean(precision_scores)),
        "mean_macro_recall": float(np.mean(recall_scores)),
        "fold_metrics": fold_metrics_list,
        "n_splits": n_splits,
        "oversample_train": oversample_train,
        "scaler_used": scaler is not None,
    }
