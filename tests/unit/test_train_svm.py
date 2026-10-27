"""Unit tests for Support Vector Machine (SVM) trainer with scaling (T4-013)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ml.registry import ModelRegistry
from ml.sensor_models.train_sensor import (
    main as cli_main,
)
from ml.sensor_models.train_sensor import (
    train_sensor_model,
)


def test_svm_runs_for_lettuce_end_to_end_and_scaler_is_saved(tmp_path: Path) -> None:
    """Acceptance Criterion 2: Scaler is bundled as part of the saved artifact and usable."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="svm",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        C=1.0,
        gamma="scale",
        kernel="rbf",
    )

    # Output verification
    assert res["crop"] == "lettuce"
    assert res["model_type"] == "svm"
    assert res["val_macro_f1"] > 0.60
    assert res["artifact_path"].is_file()
    assert res["metadata_path"].is_file()
    assert res["md_report"].is_file()

    # Load from registry and verify scaler
    registry = ModelRegistry(root_dir=reg_dir)
    loaded_pipe, _meta = registry.load_model(
        crop="lettuce",
        modality="sensor",
        version="1.0.0",
        expected_features=["ph", "ec", "water_temp", "humidity", "air_temp"],
    )

    # Acceptance Criterion: Scaler is part of the saved artifact
    assert loaded_pipe.scaler is not None
    assert isinstance(loaded_pipe.scaler, StandardScaler)
    assert hasattr(loaded_pipe.scaler, "mean_")
    assert hasattr(loaded_pipe.scaler, "scale_")


def test_svm_hyperparameters_and_probability_output(tmp_path: Path) -> None:
    """Acceptance Criterion 3: Probability output setting and hyperparameters are recorded."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"
    test_seed = 66
    test_c = 2.5
    test_gamma = 0.05
    test_kernel = "rbf"

    res = train_sensor_model(
        crop="lettuce",
        model_type="svm",
        seed=test_seed,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.1",
        C=test_c,
        gamma=test_gamma,
        kernel=test_kernel,
        probability=True,
    )

    registry = ModelRegistry(root_dir=reg_dir)
    loaded_pipe, meta = registry.load_model(
        crop="lettuce",
        modality="sensor",
        version="1.0.1",
        expected_features=["ph", "ec", "water_temp", "humidity", "air_temp"],
    )

    # Verify hyperparameters
    assert meta.hyperparameters["C"] == test_c
    assert meta.hyperparameters["gamma"] == test_gamma
    assert meta.hyperparameters["kernel"] == test_kernel
    assert meta.hyperparameters["probability"] is True
    assert meta.hyperparameters["random_state"] == test_seed

    # Verify probability output setting is enabled and outputs valid probabilities
    sample_input = np.array([[6.0, 2.0, 22.0, 65.0, 24.0]])
    probs = loaded_pipe.predict_proba(sample_input)
    assert probs.shape == (1, len(loaded_pipe.classifier.classes_))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert float(np.sum(probs)) == pytest.approx(1.0, rel=1e-3)

    # Verify seed recorded in evaluation report
    with open(res["json_report"], encoding="utf-8") as f:
        report_data = json.load(f)
    assert report_data["seeds"] == {"seed": test_seed}


def test_svm_scaling_comparison_experiment(tmp_path: Path) -> None:
    """Acceptance Criterion 1: SVM with scaling drastically outperforms unscaled baseline."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    # Train with scaling
    res_scaled = train_sensor_model(
        crop="lettuce",
        model_type="svm",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.2",
        force_scale=True,
    )

    # Train without scaling
    res_unscaled = train_sensor_model(
        crop="lettuce",
        model_type="svm",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.3",
        force_scale=False,
    )

    # Scaling comparison invariant: scaled model outperforms unscaled model by > 0.20 Macro-F1
    assert res_scaled["val_macro_f1"] > 0.50
    assert res_scaled["val_macro_f1"] > res_unscaled["val_macro_f1"] + 0.20


def test_cli_svm_execution(tmp_path: Path) -> None:
    """Verify CLI interface runs end-to-end for svm with hyperparameter flags."""
    exit_code = cli_main(
        [
            "--crop",
            "lettuce",
            "--model",
            "svm",
            "--seed",
            "42",
            "--version",
            "1.0.4",
            "--C",
            "2.0",
            "--kernel",
            "rbf",
            "--gamma",
            "scale",
            "--registry-root",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
