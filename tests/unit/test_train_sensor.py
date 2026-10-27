"""Unit tests for ml/sensor_models/train_sensor.py (T4-010)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.registry import ModelRegistry
from ml.sensor_models.schemas import DataLeakageError
from ml.sensor_models.train_sensor import (
    main as cli_main,
)
from ml.sensor_models.train_sensor import (
    train_sensor_model,
)


@pytest.fixture
def mock_lettuce_dataset_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary dataset directory with train, val, and test splits."""
    ds_dir = tmp_path / "sensor_data"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 5 features for lettuce: ph, ec, water_temp, humidity, air_temp
    data = {
        "ph": [6.0, 7.5, 5.0, 6.2, 5.8] * 10,
        "ec": [2.0, 3.5, 1.2, 2.1, 1.9] * 10,
        "water_temp": [22.0, 28.0, 20.0, 21.5, 23.0] * 10,
        "humidity": [65.0, 35.0, 70.0, 66.0, 64.0] * 10,
        "air_temp": [24.0, 32.0, 22.0, 23.5, 25.0] * 10,
        "label": ["Normal", "EC_High", "pH_Low", "Normal", "Normal"] * 10,
    }
    df = pd.DataFrame(data)

    df.to_csv(ds_dir / "train.csv", index=False)
    df.to_csv(ds_dir / "val.csv", index=False)
    df.to_csv(ds_dir / "test.csv", index=False)

    return ds_dir


def test_running_with_unreadable_test_file_succeeds(mock_lettuce_dataset_dir: Path, tmp_path: Path) -> None:
    """Acceptance Criterion 1: Running with test.csv unreadable or absent still succeeds."""
    test_csv = mock_lettuce_dataset_dir / "test.csv"
    assert test_csv.is_file()

    # Rename test.csv to simulate being absent / unreadable / deleted
    hidden_test_csv = mock_lettuce_dataset_dir / "test.csv.unreadable"
    shutil.move(test_csv, hidden_test_csv)
    assert not test_csv.exists()

    reg_dir = tmp_path / "registry"
    # Should execute and succeed without ever touching test.csv
    res = train_sensor_model(
        crop="lettuce",
        model_type="decision_tree",
        seed=42,
        data_dir=mock_lettuce_dataset_dir,
        registry_root=reg_dir,
        reports_dir=tmp_path / "reports",
        verify_splits=False,
    )

    assert res["val_accuracy"] >= 0.0
    assert res["artifact_path"].is_file()
    assert res["metadata_path"].is_file()
    assert res["md_report"].is_file()


def test_leakage_blacklist_causes_hard_failure(mock_lettuce_dataset_dir: Path, tmp_path: Path) -> None:
    """Acceptance Criterion 2: Leakage-blacklist columns cause a hard failure (DataLeakageError)."""
    # Create train dataframe contaminated with leakage columns (e.g. 'experiment', 'ph_prev', 'label_next')
    contaminated_data = {
        "ph": [6.0, 7.5],
        "ec": [2.0, 3.5],
        "water_temp": [22.0, 28.0],
        "humidity": [65.0, 35.0],
        "air_temp": [24.0, 32.0],
        "experiment": ["EXP.1", "EXP.1"],  # LEAKAGE!
        "ph_prev": [5.9, 7.4],  # LEAKAGE!
        "label": ["Normal", "EC_High"],
    }
    bad_train_df = pd.DataFrame(contaminated_data)
    clean_val_df = pd.DataFrame({
        "ph": [6.0], "ec": [2.0], "water_temp": [22.0], "humidity": [65.0], "air_temp": [24.0],
        "label": ["Normal"],
    })

    with pytest.raises(DataLeakageError, match="Data leakage detected"):
        train_sensor_model(
            crop="lettuce",
            model_type="random_forest",
            data_dir=mock_lettuce_dataset_dir,
            registry_root=tmp_path / "registry",
            verify_splits=False,
            custom_train_df=bad_train_df,
            custom_val_df=clean_val_df,
        )


def test_scaler_is_saved_inside_the_artifact(mock_lettuce_dataset_dir: Path, tmp_path: Path) -> None:
    """Acceptance Criterion 3: Scaler is saved inside the artifact and accessible."""
    reg_dir = tmp_path / "registry"

    # Train SVM model which uses scaler by default
    res = train_sensor_model(
        crop="lettuce",
        model_type="svm",
        seed=42,
        data_dir=mock_lettuce_dataset_dir,
        registry_root=reg_dir,
        reports_dir=tmp_path / "reports",
        verify_splits=False,
        force_scale=True,
    )

    # 1. Pipeline has scaler
    assert res["scaler"] is not None
    assert isinstance(res["scaler"], StandardScaler)

    # 2. Loaded from registry has scaler intact
    registry = ModelRegistry(root_dir=reg_dir)
    loaded_pipe, _meta = registry.load_model(
        crop="lettuce",
        modality="sensor",
        version="1.0.0",
        expected_features=["ph", "ec", "water_temp", "humidity", "air_temp"],
    )

    # Accessible via .scaler property
    assert loaded_pipe.scaler is not None
    assert isinstance(loaded_pipe.scaler, StandardScaler)
    assert hasattr(loaded_pipe.scaler, "mean_")

    # Accessible via named_steps
    assert "scaler" in loaded_pipe.named_steps
    assert isinstance(loaded_pipe.named_steps["scaler"], StandardScaler)

    # Accessible via dict-style __getitem__
    assert isinstance(loaded_pipe["scaler"], StandardScaler)


def test_cli_execution_succeeds(mock_lettuce_dataset_dir: Path, tmp_path: Path) -> None:
    """Verify CLI interface runs end-to-end and returns code 0."""
    exit_code = cli_main([
        "--crop", "lettuce",
        "--model", "decision_tree",
        "--seed", "123",
        "--version", "2.0.0",
        "--data-dir", str(mock_lettuce_dataset_dir),
        "--registry-root", str(tmp_path / "reg"),
        "--reports-dir", str(tmp_path / "rep"),
        "--no-verify-splits",
    ])
    assert exit_code == 0
