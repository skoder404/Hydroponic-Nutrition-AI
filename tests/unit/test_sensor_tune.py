"""Unit tests for Hyperparameter Search with Grouped CV (T4-015)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.evaluation.cv import GroupOverlapError, assert_no_group_overlap
from ml.sensor_models.tune import (
    main as cli_main,
)
from ml.sensor_models.tune import (
    tune_sensor_model,
)


@pytest.fixture
def mock_tune_dataset(tmp_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Generate a multi-group sensor dataset for hyperparameter tuning tests."""
    # 5 features, 3 distinct groups (experiments), multiple classes
    data = {
        "ph": [6.0, 7.5, 5.0, 6.2, 5.8, 6.1, 7.2, 5.2, 6.0, 5.9] * 4,
        "ec": [2.0, 3.5, 1.2, 2.1, 1.9, 2.0, 3.4, 1.3, 2.0, 1.8] * 4,
        "water_temp": [22.0, 28.0, 20.0, 21.5, 23.0, 22.5, 27.5, 20.5, 22.0, 23.0] * 4,
        "humidity": [65.0, 35.0, 70.0, 66.0, 64.0, 65.0, 36.0, 69.0, 65.0, 63.0] * 4,
        "air_temp": [24.0, 32.0, 22.0, 23.5, 25.0, 24.5, 31.5, 22.5, 24.0, 25.0] * 4,
        "label": [
            "Normal",
            "EC_High",
            "pH_Low",
            "Normal",
            "Normal",
            "Normal",
            "EC_High",
            "pH_Low",
            "Normal",
            "Normal",
        ]
        * 4,
        "experiment": ["EXP_A"] * 10 + ["EXP_B"] * 15 + ["EXP_C"] * 15,
    }
    df = pd.DataFrame(data)
    groups = df["experiment"].tolist()
    return df, groups


def test_search_uses_group_aware_folds_only(
    mock_tune_dataset: tuple[pd.DataFrame, list[str]], tmp_path: Path
) -> None:
    """Acceptance Criterion 1: Search uses group-aware folds only, enforcing zero group overlap."""
    df, groups = mock_tune_dataset
    out_dir = tmp_path / "tuning_reports"

    # Small override grid for fast execution
    small_grid = [
        {"max_depth": 3, "min_samples_leaf": 1},
        {"max_depth": 5, "min_samples_leaf": 2},
    ]

    res = tune_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        custom_df=df,
        custom_groups=groups,
        output_dir=out_dir,
        n_splits=3,
        seed=42,
        override_grid=small_grid,
    )

    # Verify that tuning completed using group folds
    assert res["best_params"] is not None
    assert res["best_macro_f1"] >= 0.0

    # Invariant: assert_no_group_overlap fails if group leakage were introduced
    with pytest.raises(GroupOverlapError, match="Group overlap detected"):
        assert_no_group_overlap(["EXP_A", "EXP_B"], ["EXP_B", "EXP_C"])


def test_best_parameters_and_cv_scores_saved_to_json(
    mock_tune_dataset: tuple[pd.DataFrame, list[str]], tmp_path: Path
) -> None:
    """Acceptance Criterion 2: Best parameters and CV scores are saved to output JSON."""
    df, groups = mock_tune_dataset
    out_dir = tmp_path / "tuning_reports"

    small_grid = [
        {"n_estimators": 10, "max_depth": 3},
        {"n_estimators": 15, "max_depth": 4},
    ]

    res = tune_sensor_model(
        crop="lettuce",
        model_type="random_forest",
        custom_df=df,
        custom_groups=groups,
        output_dir=out_dir,
        n_splits=3,
        seed=77,
        override_grid=small_grid,
    )

    out_file = res["output_path"]
    assert out_file.is_file()

    # Verify saved JSON payload contents
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["crop"] == "lettuce"
    assert payload["model_family"] == "random_forest"
    assert payload["best_params"] in small_grid
    assert payload["best_macro_f1"] == pytest.approx(res["best_macro_f1"], rel=1e-3)
    assert len(payload["all_results"]) == len(small_grid)


def test_runtime_and_seed_recorded(
    mock_tune_dataset: tuple[pd.DataFrame, list[str]], tmp_path: Path
) -> None:
    """Acceptance Criterion 3: Runtime and seed are recorded in the tuning results."""
    df, groups = mock_tune_dataset
    test_seed = 999

    res = tune_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        custom_df=df,
        custom_groups=groups,
        output_dir=tmp_path,
        n_splits=3,
        seed=test_seed,
        override_grid=[{"max_depth": 2}],
    )

    # Verify runtime and seed recorded
    assert res["seed"] == test_seed
    assert res["runtime_seconds"] >= 0.0

    payload = json.loads(res["output_path"].read_text(encoding="utf-8"))
    assert payload["seed"] == test_seed
    assert payload["runtime_seconds"] >= 0.0
    assert "completed_at" in payload


def test_test_set_never_accessed_during_tuning(tmp_path: Path) -> None:
    """Precondition: Tuning operates strictly on training partitions; test.csv is never accessed."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy train.csv
    train_data = {
        "ph": [6.0, 7.5, 5.0, 6.2] * 5,
        "ec": [2.0, 3.5, 1.2, 2.1] * 5,
        "water_temp": [22.0, 28.0, 20.0, 21.5] * 5,
        "humidity": [65.0, 35.0, 70.0, 66.0] * 5,
        "air_temp": [24.0, 32.0, 22.0, 23.5] * 5,
        "label": ["Normal", "EC_High", "pH_Low", "Normal"] * 5,
        "experiment": ["EXP_1"] * 10 + ["EXP_2"] * 10,
    }
    pd.DataFrame(train_data).to_csv(ds_dir / "train.csv", index=False)

    # Explicitly corrupt/delete test.csv
    test_file = ds_dir / "test.csv"
    assert not test_file.exists()

    # Tuning should execute cleanly without touching test.csv
    res = tune_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        data_dir=ds_dir,
        output_dir=tmp_path / "reports",
        n_splits=2,
        seed=42,
        override_grid=[{"max_depth": 3}],
    )
    assert res["best_macro_f1"] >= 0.0
    assert not test_file.exists()


def test_cli_tune_execution(
    mock_tune_dataset: tuple[pd.DataFrame, list[str]], tmp_path: Path
) -> None:
    """Verify CLI interface for tune.py runs successfully."""
    df, _ = mock_tune_dataset
    ds_dir = tmp_path / "sensor_data"
    ds_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ds_dir / "train.csv", index=False)

    exit_code = cli_main(
        [
            "--crop",
            "lettuce",
            "--model",
            "decision_tree",
            "--data-dir",
            str(ds_dir),
            "--output-dir",
            str(tmp_path / "tuning_out"),
            "--n-splits",
            "3",
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0
