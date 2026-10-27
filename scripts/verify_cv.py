#!/usr/bin/env python3
"""Verification script for grouped cross-validation with in-fold oversampling (T4-007).

Acceptance Criteria:
1. Group overlap raises an error (GroupOverlapError).
2. Oversampling never touches the validation fold.
3. Unit test cases are listed for T8 tasks.
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001, S110
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation import (
    GroupOverlapError,
    GuardedGroupKFold,
    assert_no_group_overlap,
    get_fold_data,
)


def verify_group_overlap_raises_error() -> tuple[bool, str]:
    """Criterion 1: Group overlap raises an error."""
    train_g = ["exp1_day1", "exp1_day2", "exp1_day3"]
    val_g = ["exp1_day3", "exp1_day4"]  # Overlap on exp1_day3

    try:
        assert_no_group_overlap(train_g, val_g, fold_idx=0)
        return (
            False,
            "assert_no_group_overlap failed to raise GroupOverlapError on overlapping groups.",
        )
    except GroupOverlapError as exc:
        if "Group overlap detected" not in str(exc):
            return False, f"Unexpected error message: {exc}"

    # Verify GuardedGroupKFold on disjoint groups
    n_samples = 60
    groups = np.repeat([f"day_{i}" for i in range(6)], 10)
    X = np.ones((n_samples, 2))
    cv = GuardedGroupKFold(n_splits=3)
    splits = list(cv.split(X, groups=groups))

    for t_idx, v_idx in splits:
        t_set = set(groups[t_idx])
        v_set = set(groups[v_idx])
        if len(t_set.intersection(v_set)) > 0:
            return (
                False,
                f"GuardedGroupKFold yielded overlapping groups: {t_set & v_set}",
            )

    return (
        True,
        "Criterion 1 satisfied: Group overlap strictly raises GroupOverlapError and GuardedGroupKFold enforces disjointness.",
    )


def verify_oversampling_never_touches_validation_fold() -> tuple[bool, str]:
    """Criterion 2: Oversampling never touches the validation fold."""
    records = []
    groups = []
    targets = []

    # Majority class: 40 rows
    for i in range(40):
        records.append({"feat1": float(i), "feat2": float(i * 10)})
        groups.append("Group_Train_A")
        targets.append("Normal")

    # Minority class in train: 4 rows
    for i in range(4):
        records.append({"feat1": float(100 + i), "feat2": float(200 + i)})
        groups.append("Group_Train_B")
        targets.append("pH_Low")

    # Validation group with minority class: 3 rows
    for i in range(3):
        records.append({"feat1": float(500 + i), "feat2": float(600 + i)})
        groups.append("Group_Val")
        targets.append("pH_Low")

    X = pd.DataFrame(records)
    y = pd.Series(targets)
    groups = np.array(groups)

    val_idx = np.where(groups == "Group_Val")[0]
    train_idx = np.where(groups != "Group_Val")[0]

    raw_val_x = X.iloc[val_idx].copy()
    raw_val_y = y.iloc[val_idx].copy()

    _x_tr, y_tr, x_val, y_val = get_fold_data(
        X=X,
        y=y,
        groups=groups,
        train_idx=train_idx,
        val_idx=val_idx,
        oversample_train=True,
        random_state=42,
    )

    # Invariants for validation partition
    if len(x_val) != 3 or len(y_val) != 3:
        return False, f"Validation fold length changed! Expected 3, got {len(x_val)}"

    if not (x_val.reset_index(drop=True).equals(raw_val_x.reset_index(drop=True))):
        return False, "Validation features were altered by oversampling!"

    if not (y_val.reset_index(drop=True).equals(raw_val_y.reset_index(drop=True))):
        return False, "Validation target labels were altered by oversampling!"

    # Training partition MUST be balanced to majority count (40)
    counts_tr = pd.Series(y_tr).value_counts()
    if counts_tr["Normal"] != 40 or counts_tr["pH_Low"] != 40:
        return False, f"Training partition was not properly balanced: {dict(counts_tr)}"

    return (
        True,
        f"Criterion 2 satisfied: Validation fold remains strictly untouched ({len(x_val)} rows) while train fold is balanced (40 -> 80 rows).",
    )


def verify_unit_test_cases_listed() -> tuple[bool, str]:
    """Criterion 3: Unit test cases are listed for T8 tasks."""
    test_file = ROOT / "tests" / "unit" / "test_cv.py"
    if not test_file.exists():
        return False, f"Missing test file: {test_file}"

    content = test_file.read_text(encoding="utf-8")
    required_tests = [
        "test_assert_no_group_overlap_raises_on_leakage",
        "test_assert_no_group_overlap_passes_on_disjoint_groups",
        "test_guarded_group_kfold_split_enforces_group_separation",
        "test_guarded_group_kfold_validation_errors",
        "test_oversample_minority_classes_balances_training_fold",
        "test_oversampling_never_touches_validation_fold",
        "test_evaluate_grouped_cv_end_to_end",
    ]

    missing = [t for t in required_tests if t not in content]
    if missing:
        return False, f"Missing unit tests in test_cv.py: {missing}"

    return (
        True,
        f"Criterion 3 satisfied: All {len(required_tests)} unit test cases listed and implemented for T8-004.",
    )


def main() -> int:
    print("=" * 70)
    print("T4-007 Verification: Grouped CV with In-Fold Oversampling")
    print("=" * 70)

    checks = [
        (
            "Criterion 1: Group overlap raises an error",
            verify_group_overlap_raises_error,
        ),
        (
            "Criterion 2: Oversampling never touches validation fold",
            verify_oversampling_never_touches_validation_fold,
        ),
        (
            "Criterion 3: Unit test cases listed for T8 tasks",
            verify_unit_test_cases_listed,
        ),
    ]

    all_passed = True
    for label, fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        print(f"       {msg}\n")
        if not ok:
            all_passed = False

    if all_passed:
        print("ALL CRITERIA PASSED! T4-007 is fully verified.")
        return 0
    else:
        print("VERIFICATION FAILED! Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
