"""Unit tests for ml.registry.metadata_schema adhering to Overview Section 24.1 (T4-008)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ml.registry import ModelMetadata

MANIFEST_PATH = Path("models/manifests/example_lettuce_sensor_rf.json")


@pytest.fixture
def valid_sensor_metadata_dict() -> dict:
    """Fixture providing a valid dictionary representation of ModelMetadata."""
    return {
        "model_id": "test_sensor_model_v1",
        "crop": "spinach",
        "task": "classification",
        "modality": "sensor",
        "family": "DecisionTreeClassifier",
        "artifact_version": "1.0.0",
        "artifact_file": "models/sensor/spinach_dt_v1.joblib",
        "commit": "8133572a9b407fcf8f0cf52011b66df870197d02",
        "training_date": "2026-10-27T04:00:00+00:00",
        "dataset_manifest_hash": "c62bf6a127fce1c4efdfa88d752ad755a7201b2fba9eb73b306b9b1e92d6e3c5",
        "rule_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "experimental": True,
        "status": "candidate",
        "classes": ["Normal", "EC_High_Toxicity"],
        "display_mapping": {
            "Normal": "healthy",
            "EC_High_Toxicity": "unhealthy/EC_High_Toxicity",
        },
        "split_protocol": "Time-block stratified split across greenhouse benches",
        "split_counts": {"train": 1120, "validation": 140, "test": 140},
        "ordered_features": ["air_temperature_C", "ph"],
        "feature_units": {"air_temperature_C": "degC", "ph": "pH_units"},
        "input_unit": "sensor_reading",
        "hyperparameters": {"max_depth": 5, "criterion": "gini"},
        "confidence_settings": {"uncertainty_threshold": 0.35},
        "expected_input_schema": {"type": "object", "required": ["air_temperature_C", "ph"]},
        "dependency_versions": {"python": "3.13.5", "scikit-learn": "1.6.1"},
        "validation_metrics": {"accuracy": 0.98, "macro_f1": 0.97},
        "final_test_metrics": None,
    }


def test_schema_validation_fails_on_missing_mandatory_field(valid_sensor_metadata_dict: dict) -> None:
    """Acceptance Criterion 1: Schema validation fails on any missing mandatory field."""
    # List of required fields
    mandatory_fields = [
        "model_id",
        "crop",
        "task",
        "modality",
        "family",
        "artifact_version",
        "artifact_file",
        "commit",
        "training_date",
        "dataset_manifest_hash",
        "rule_version",
        "preprocessing_version",
        "experimental",
        "status",
        "classes",
        "display_mapping",
        "split_protocol",
        "split_counts",
        "input_unit",
        "hyperparameters",
        "confidence_settings",
        "expected_input_schema",
        "dependency_versions",
        "validation_metrics",
    ]

    for field in mandatory_fields:
        tampered = valid_sensor_metadata_dict.copy()
        del tampered[field]
        with pytest.raises(ValidationError, match=field):
            ModelMetadata.model_validate(tampered)


def test_experimental_and_status_fields_exist_and_validated(valid_sensor_metadata_dict: dict) -> None:
    """Acceptance Criterion 2: 'experimental' and 'status' fields exist and enforce valid types/values."""
    # Test valid values
    meta = ModelMetadata.model_validate(valid_sensor_metadata_dict)
    assert meta.experimental is True
    assert meta.status == "candidate"

    # Test status transitions
    for valid_status in ("candidate", "promoted", "deprecated", "archived"):
        updated = valid_sensor_metadata_dict.copy()
        updated["status"] = valid_status
        m = ModelMetadata.model_validate(updated)
        assert m.status == valid_status

    # Test invalid status raises ValidationError
    invalid_status_dict = valid_sensor_metadata_dict.copy()
    invalid_status_dict["status"] = "deployed_production"  # Not in allowed enum
    with pytest.raises(ValidationError, match="status"):
        ModelMetadata.model_validate(invalid_status_dict)

    # Test invalid experimental type raises ValidationError
    invalid_exp_dict = valid_sensor_metadata_dict.copy()
    invalid_exp_dict["experimental"] = "not_a_bool"
    with pytest.raises(ValidationError, match="experimental"):
        ModelMetadata.model_validate(invalid_exp_dict)


def test_example_metadata_file_committed_and_valid() -> None:
    """Acceptance Criterion 3: Example metadata file exists under models/manifests/ and validates."""
    assert MANIFEST_PATH.is_file(), f"Missing example metadata manifest: {MANIFEST_PATH}"

    # Load with classmethod
    meta = ModelMetadata.load(MANIFEST_PATH)
    assert meta.model_id == "lettuce_sensor_random_forest_v1"
    assert meta.crop == "lettuce"
    assert meta.modality == "sensor"
    assert meta.status == "candidate"
    assert meta.experimental is False
    assert len(meta.classes) == 9
    assert meta.validation_metrics["macro_f1"] == 0.986


def test_image_modality_constraints() -> None:
    """Verify Overview Section 24.1 image constraints: input_unit must be 'leaf' or 'single_leaf'."""
    image_dict = {
        "model_id": "lettuce_image_cnn_v1",
        "crop": "lettuce",
        "task": "classification",
        "modality": "image",
        "family": "CustomCNN",
        "artifact_version": "1.0.0",
        "artifact_file": "models/image/lettuce_cnn.pth",
        "commit": "8133572a9b407fcf8f0cf52011b66df870197d02",
        "training_date": "2026-10-27T04:00:00+00:00",
        "dataset_manifest_hash": "c62bf6a127fce1c4efdfa88d752ad755a7201b2fba9eb73b306b9b1e92d6e3c5",
        "rule_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "experimental": False,
        "status": "promoted",
        "classes": ["Healthier", "NitrogenDeficiency"],
        "display_mapping": {"Healthier": "healthy", "NitrogenDeficiency": "unhealthy/NitrogenDeficiency"},
        "split_protocol": "Stratified image partition with dHash deduplication",
        "split_counts": {"train": 700, "validation": 150, "test": 150},
        "input_unit": "whole_tray",  # VIOLATION: Must be leaf or single_leaf per §24.1
        "image_size": [224, 224],
        "channel_order": "RGB",
        "hyperparameters": {"lr": 1e-3, "batch_size": 32},
        "confidence_settings": {"uncertainty_threshold": 0.5},
        "expected_input_schema": {"shape": [3, 224, 224]},
        "dependency_versions": {"python": "3.13.5", "torch": "2.5.1"},
        "validation_metrics": {"accuracy": 0.95, "macro_f1": 0.94},
    }

    # Fails when input_unit is not leaf/single_leaf
    with pytest.raises(ValidationError, match="input_unit for image/feature models must be 'leaf' or 'single_leaf'"):
        ModelMetadata.model_validate(image_dict)

    # Passes when input_unit is single_leaf
    image_dict["input_unit"] = "single_leaf"
    meta = ModelMetadata.model_validate(image_dict)
    assert meta.input_unit == "single_leaf"


def test_missing_display_mapping_fails_validation(valid_sensor_metadata_dict: dict) -> None:
    """Verify that unmapped class labels in display_mapping are rejected."""
    tampered = valid_sensor_metadata_dict.copy()
    # Remove one class from display_mapping
    tampered["display_mapping"] = {"Normal": "healthy"}
    with pytest.raises(ValidationError, match="display_mapping is missing entries for classes.*EC_High_Toxicity"):
        ModelMetadata.model_validate(tampered)
