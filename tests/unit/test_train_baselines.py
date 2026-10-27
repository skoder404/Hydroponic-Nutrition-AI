"""Unit tests for Baseline Classifiers (KNN and Logistic Regression) (T4-014)."""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ml.registry import ModelRegistry
from ml.sensor_models.train_sensor import (
    main as cli_main,
)
from ml.sensor_models.train_sensor import (
    train_sensor_model,
)


def test_knn_trains_end_to_end_with_scaler(tmp_path: Path) -> None:
    """Acceptance Criterion 1a: KNN model trains end to end and bundles scaler."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="knn",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        n_neighbors=7,
        weights="distance",
    )

    # Output verification
    assert res["crop"] == "lettuce"
    assert res["model_type"] == "knn"
    assert res["val_macro_f1"] > 0.40
    assert res["artifact_path"].is_file()
    assert res["metadata_path"].is_file()
    assert res["md_report"].is_file()

    # Load from registry and verify scaler and estimator
    registry = ModelRegistry(root_dir=reg_dir)
    loaded_pipe, meta = registry.load_model(
        crop="lettuce",
        modality="sensor",
        version="1.0.0",
        expected_features=["ph", "ec", "water_temp", "humidity", "air_temp"],
    )

    assert isinstance(loaded_pipe.scaler, StandardScaler)
    assert isinstance(loaded_pipe.classifier, KNeighborsClassifier)
    assert meta.family == "KNeighborsClassifier"
    assert meta.hyperparameters["n_neighbors"] == 7
    assert meta.hyperparameters["weights"] == "distance"


def test_logistic_regression_trains_end_to_end_with_scaler(tmp_path: Path) -> None:
    """Acceptance Criterion 1b: Logistic Regression trains end to end and bundles scaler."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"

    res = train_sensor_model(
        crop="lettuce",
        model_type="logistic_regression",
        seed=42,
        registry_root=reg_dir,
        reports_dir=rep_dir,
        artifact_version="1.0.0",
        C=0.5,
        class_weight="balanced",
    )

    assert res["crop"] == "lettuce"
    assert res["model_type"] == "logistic_regression"
    assert res["val_macro_f1"] > 0.40
    assert res["artifact_path"].is_file()
    assert res["metadata_path"].is_file()

    registry = ModelRegistry(root_dir=reg_dir)
    loaded_pipe, meta = registry.load_model(
        crop="lettuce",
        modality="sensor",
        version="1.0.0",
        expected_features=["ph", "ec", "water_temp", "humidity", "air_temp"],
    )

    assert isinstance(loaded_pipe.scaler, StandardScaler)
    assert isinstance(loaded_pipe.classifier, LogisticRegression)
    assert meta.family == "LogisticRegression"
    assert meta.hyperparameters["C"] == 0.5
    assert meta.hyperparameters["class_weight"] == "balanced"


def test_five_model_families_comparison_table(tmp_path: Path) -> None:
    """Acceptance Criterion 2: Baselines appear in comparison table alongside DT, RF, and SVM."""
    reg_dir = tmp_path / "models"
    rep_dir = tmp_path / "reports"
    families = ["decision_tree", "random_forest", "svm", "knn", "logistic_regression"]

    results = {}
    for fam in families:
        res = train_sensor_model(
            crop="lettuce",
            model_type=fam,
            seed=42,
            registry_root=reg_dir,
            reports_dir=rep_dir,
            artifact_version=f"1.0.{fam}",
        )
        results[fam] = {
            "macro_f1": res["val_macro_f1"],
            "accuracy": res["val_accuracy"],
            "artifact": res["artifact_path"].name,
        }

    # Verify all 5 families evaluated successfully
    assert len(results) == 5
    for fam in families:
        assert results[fam]["macro_f1"] > 0.0
        assert results[fam]["accuracy"] > 0.0


def test_cli_baselines_execution(tmp_path: Path) -> None:
    """Verify CLI interface runs end-to-end for knn and logistic_regression."""
    # Test KNN CLI
    exit_knn = cli_main(
        [
            "--crop",
            "lettuce",
            "--model",
            "knn",
            "--seed",
            "42",
            "--version",
            "1.0.1",
            "--n-neighbors",
            "5",
            "--weights",
            "uniform",
            "--registry-root",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_knn == 0

    # Test Logistic Regression CLI
    exit_lr = cli_main(
        [
            "--crop",
            "lettuce",
            "--model",
            "logistic_regression",
            "--seed",
            "42",
            "--version",
            "1.0.1",
            "--C",
            "1.0",
            "--class-weight",
            "balanced",
            "--registry-root",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_lr == 0
