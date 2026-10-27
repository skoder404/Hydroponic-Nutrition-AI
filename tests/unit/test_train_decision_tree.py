"""Unit tests for Decision Tree trainer and rule exporter (T4-011)."""

from __future__ import annotations

from pathlib import Path

from ml.registry import ModelRegistry
from ml.sensor_models.train_sensor import (
    main as cli_main,
)
from ml.sensor_models.train_sensor import (
    train_sensor_model,
)


def test_decision_tree_runs_for_lettuce_end_to_end(tmp_path: Path) -> None:
    """Acceptance Criterion 1: Runs for lettuce end to end on prepared dataset."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        max_depth=6,
        min_samples_leaf=2,
        criterion="gini",
    )

    # Output verification
    assert res["crop"] == "lettuce"
    assert res["model_type"] == "decision_tree"
    assert res["val_macro_f1"] > 0.50
    assert res["artifact_path"].is_file()
    assert res["metadata_path"].is_file()
    assert res["md_report"].is_file()


def test_exported_rules_file_is_written(tmp_path: Path) -> None:
    """Acceptance Criterion 2: Exported text rules file is written next to the artifact."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        max_depth=5,
    )

    rules_path = res["rules_path"]
    assert rules_path is not None
    assert rules_path.is_file()

    # Rule file sits in the same version directory next to the model artifact
    assert rules_path.parent == res["artifact_path"].parent
    assert rules_path.name.endswith("_rules.txt")

    content = rules_path.read_text(encoding="utf-8")
    assert len(content) > 50
    # Rule contains feature names from lettuce sensor schema
    assert any(
        feat in content for feat in ("ph", "ec", "water_temp", "humidity", "air_temp")
    )


def test_seeds_and_hyperparameters_recorded(tmp_path: Path) -> None:
    """Acceptance Criterion 3: Seeds and configurable hyperparameters are recorded in metadata."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"
    test_seed = 99
    test_depth = 4
    test_leaf = 3
    test_crit = "entropy"

    res = train_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        seed=test_seed,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.1",
        max_depth=test_depth,
        min_samples_leaf=test_leaf,
        criterion=test_crit,
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
    assert meta.hyperparameters["max_depth"] == test_depth
    assert meta.hyperparameters["min_samples_leaf"] == test_leaf
    assert meta.hyperparameters["criterion"] == test_crit

    # Verify seed recorded in Section 18.2 evaluation report
    import json

    with open(res["json_report"], encoding="utf-8") as f:
        report_data = json.load(f)
    assert report_data["seeds"] == {"seed": test_seed}


def test_cli_decision_tree_execution(tmp_path: Path) -> None:
    """Verify CLI interface runs end-to-end for decision_tree with hyperparameter flags."""
    exit_code = cli_main(
        [
            "--crop",
            "lettuce",
            "--model",
            "decision_tree",
            "--seed",
            "42",
            "--version",
            "1.0.2",
            "--max-depth",
            "5",
            "--min-samples-leaf",
            "2",
            "--criterion",
            "gini",
            "--registry-root",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
