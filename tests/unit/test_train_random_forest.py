"""Unit tests for Random Forest trainer and feature importance saver (T4-012)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.registry import ModelRegistry
from ml.sensor_models.train_sensor import (
    main as cli_main,
)
from ml.sensor_models.train_sensor import (
    train_sensor_model,
)


def test_random_forest_runs_for_lettuce_end_to_end(tmp_path: Path) -> None:
    """Acceptance Criterion 1: Runs for lettuce end to end on prepared dataset."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="random_forest",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        n_estimators=30,
        max_depth=6,
        min_samples_leaf=2,
    )

    # Output verification
    assert res["crop"] == "lettuce"
    assert res["model_type"] == "random_forest"
    assert res["val_macro_f1"] > 0.50
    assert res["artifact_path"].is_file()
    assert res["metadata_path"].is_file()
    assert res["md_report"].is_file()


def test_feature_importances_saved(tmp_path: Path) -> None:
    """Acceptance Criterion 2: Feature importances are saved next to the artifact."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="random_forest",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        n_estimators=25,
    )

    # Acceptance Criterion: Feature importances saved
    imp_path = res["feature_importances_path"]
    assert imp_path is not None
    assert imp_path.is_file()
    assert imp_path.parent == res["artifact_path"].parent

    # Content verification: check that JSON contains all expected features
    content = json.loads(imp_path.read_text(encoding="utf-8"))
    expected_feats = ["ph", "ec", "water_temp", "humidity", "air_temp"]
    for feat in expected_feats:
        assert feat in content
        assert isinstance(content[feat], float)
        assert content[feat] >= 0.0

    # Total feature importance in Random Forest sums to ~1.0
    total_importance = sum(content.values())
    assert total_importance == pytest.approx(1.0, rel=1e-2)


def test_rf_seeds_and_hyperparameters_recorded(tmp_path: Path) -> None:
    """Acceptance Criterion 3: Seeds and configurable hyperparameters are recorded."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"
    test_seed = 88
    test_n_trees = 40
    test_depth = 5
    test_leaf = 3
    test_crit = "entropy"
    test_weight = "balanced"

    res = train_sensor_model(
        crop="lettuce",
        model_type="random_forest",
        seed=test_seed,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.1",
        n_estimators=test_n_trees,
        max_depth=test_depth,
        min_samples_leaf=test_leaf,
        criterion=test_crit,
        class_weight=test_weight,
    )

    registry = ModelRegistry(root_dir=reg_dir)
    _pipe, meta = registry.load_model(
        crop="lettuce",
        modality="sensor",
        version="1.0.1",
        expected_features=["ph", "ec", "water_temp", "humidity", "air_temp"],
    )

    # Verify seed and hyperparameters recorded in metadata
    assert meta.hyperparameters["random_state"] == test_seed
    assert meta.hyperparameters["n_estimators"] == test_n_trees
    assert meta.hyperparameters["max_depth"] == test_depth
    assert meta.hyperparameters["min_samples_leaf"] == test_leaf
    assert meta.hyperparameters["criterion"] == test_crit
    assert meta.hyperparameters["class_weight"] == test_weight

    # Verify seed recorded in Section 18.2 evaluation report
    with open(res["json_report"], encoding="utf-8") as f:
        report_data = json.load(f)
    assert report_data["seeds"] == {"seed": test_seed}


def test_cli_random_forest_execution(tmp_path: Path) -> None:
    """Verify CLI interface runs end-to-end for random_forest with hyperparameter flags."""
    exit_code = cli_main(
        [
            "--crop",
            "lettuce",
            "--model",
            "random_forest",
            "--seed",
            "42",
            "--version",
            "1.0.2",
            "--n-estimators",
            "20",
            "--max-depth",
            "4",
            "--min-samples-leaf",
            "2",
            "--criterion",
            "gini",
            "--class-weight",
            "balanced",
            "--registry-root",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
