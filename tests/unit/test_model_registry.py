"""Unit tests for ModelRegistry and promote_model.py CLI (T4-009)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.registry import (
    ModelMetadata,
    ModelRegistry,
    PromotionPermissionError,
    SchemaMismatchError,
)
from scripts.promote_model import main as cli_main


def _create_sample_metadata(
    crop: str = "lettuce",
    modality: str = "sensor",
    version: str = "1.0.0",
    features: list[str] | None = None,
) -> ModelMetadata:
    """Helper to construct valid ModelMetadata for registry tests."""
    ordered_feats = features if features is not None else ["air_temp", "humidity", "ph"]
    return ModelMetadata.model_validate({
        "model_id": f"{crop}_{modality}_model_v{version}",
        "crop": crop,
        "task": "classification",
        "modality": modality,
        "family": "RandomForestClassifier",
        "artifact_version": version,
        "artifact_file": f"model_v{version}.joblib",
        "commit": "8133572a9b407fcf8f0cf52011b66df870197d02",
        "training_date": "2026-10-27T04:00:00+00:00",
        "dataset_manifest_hash": "c62bf6a127fce1c4efdfa88d752ad755a7201b2fba9eb73b306b9b1e92d6e3c5",
        "rule_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "experimental": False,
        "status": "candidate",
        "classes": ["Normal", "EC_High"],
        "display_mapping": {"Normal": "healthy", "EC_High": "unhealthy/EC_High"},
        "split_protocol": "Calendar holdout",
        "split_counts": {"train": 100, "validation": 20, "test": 20},
        "ordered_features": ordered_feats,
        "feature_units": {f: "units" for f in ordered_feats},
        "input_unit": "sensor_reading",
        "hyperparameters": {"n_estimators": 50},
        "confidence_settings": {"uncertainty_threshold": 0.4},
        "expected_input_schema": {"type": "object"},
        "dependency_versions": {"python": "3.13.5"},
        "validation_metrics": {"accuracy": 0.98, "macro_f1": 0.97},
    })


def test_promote_then_rollback_restores_previous_version(tmp_path: Path) -> None:
    """Acceptance Criterion 1: Promote then rollback restores the previous version."""
    registry = ModelRegistry(root_dir=tmp_path)

    # 1. Save v1.0.0
    meta_v1 = _create_sample_metadata(version="1.0.0")
    dummy_model_v1 = {"weights": [1, 2, 3], "version": "1.0.0"}
    registry.save_model(dummy_model_v1, meta_v1)

    # Promote v1.0.0
    res_p1 = registry.promote("lettuce", "sensor", "1.0.0", called_from_cli=True)
    assert res_p1["active_version"] == "1.0.0"
    assert res_p1["previous_version"] is None

    # Verify active is v1.0.0
    loaded_obj, loaded_meta = registry.load_model("lettuce", "sensor")
    assert loaded_meta.artifact_version == "1.0.0"
    assert loaded_obj["version"] == "1.0.0"

    # 2. Save v1.1.0
    meta_v2 = _create_sample_metadata(version="1.1.0")
    dummy_model_v2 = {"weights": [4, 5, 6], "version": "1.1.0"}
    registry.save_model(dummy_model_v2, meta_v2)

    # Promote v1.1.0
    res_p2 = registry.promote("lettuce", "sensor", "1.1.0", called_from_cli=True)
    assert res_p2["active_version"] == "1.1.0"
    assert res_p2["previous_version"] == "1.0.0"

    # Verify active is now v1.1.0
    loaded_obj, loaded_meta = registry.load_model("lettuce", "sensor")
    assert loaded_meta.artifact_version == "1.1.0"
    assert loaded_obj["version"] == "1.1.0"

    # 3. Rollback to v1.0.0
    res_rb = registry.rollback("lettuce", "sensor", called_from_cli=True)
    assert res_rb["active_version"] == "1.0.0"
    assert res_rb["previous_version"] == "1.1.0"

    # Verify active is restored to v1.0.0
    loaded_obj, loaded_meta = registry.load_model("lettuce", "sensor")
    assert loaded_meta.artifact_version == "1.0.0"
    assert loaded_obj["version"] == "1.0.0"


def test_loading_artifact_with_mismatched_feature_list_fails_schema_mismatch(tmp_path: Path) -> None:
    """Acceptance Criterion 2: Loading an artifact with a mismatched feature list fails with SCHEMA_MISMATCH."""
    registry = ModelRegistry(root_dir=tmp_path)

    # Model trained with features: ["air_temp", "humidity", "ph"]
    meta = _create_sample_metadata(version="1.0.0", features=["air_temp", "humidity", "ph"])
    dummy_model = {"model": "sample"}
    registry.save_model(dummy_model, meta)
    registry.promote("lettuce", "sensor", "1.0.0", called_from_cli=True)

    # Case A: Correct features succeed
    loaded_obj, _ = registry.load_model(
        "lettuce", "sensor", expected_features=["air_temp", "humidity", "ph"]
    )
    assert loaded_obj["model"] == "sample"

    # Case B: Missing feature fails with SCHEMA_MISMATCH
    with pytest.raises(SchemaMismatchError) as exc_info:
        registry.load_model(
            "lettuce", "sensor", expected_features=["air_temp", "humidity"]
        )
    assert "SCHEMA_MISMATCH" in str(exc_info.value)
    assert exc_info.value.error_code == "SCHEMA_MISMATCH"

    # Case C: Permuted/different feature order fails with SCHEMA_MISMATCH (order matters!)
    with pytest.raises(SchemaMismatchError) as exc_info:
        registry.load_model(
            "lettuce", "sensor", expected_features=["humidity", "air_temp", "ph"]
        )
    assert "SCHEMA_MISMATCH" in str(exc_info.value)


def test_promotion_is_cli_only(tmp_path: Path) -> None:
    """Acceptance Criterion 3: Promotion and rollback are CLI-only."""
    registry = ModelRegistry(root_dir=tmp_path)
    meta = _create_sample_metadata(version="1.0.0")
    registry.save_model({"m": 1}, meta)

    # Direct code call with called_from_cli=False fails with PromotionPermissionError
    with pytest.raises(PromotionPermissionError, match="restricted to CLI commands only"):
        registry.promote("lettuce", "sensor", "1.0.0", called_from_cli=False)

    with pytest.raises(PromotionPermissionError, match="restricted to CLI commands only"):
        registry.rollback("lettuce", "sensor", called_from_cli=False)

    # CLI execution via scripts/promote_model.py succeeds
    exit_code = cli_main([
        "promote",
        "--crop", "lettuce",
        "--modality", "sensor",
        "--version", "1.0.0",
        "--root", str(tmp_path),
    ])
    assert exit_code == 0

    # CLI show command succeeds
    exit_code_show = cli_main([
        "show",
        "--crop", "lettuce",
        "--modality", "sensor",
        "--root", str(tmp_path),
    ])
    assert exit_code_show == 0


def test_versioned_directory_structure_preserved(tmp_path: Path) -> None:
    """Verify artifacts are persisted under versioned directories models/<modality>/<crop>/v<version>/."""
    registry = ModelRegistry(root_dir=tmp_path)
    meta = _create_sample_metadata(crop="spinach", modality="sensor", version="2.3.1")
    art_path, meta_path = registry.save_model({"spinach": 1}, meta)

    expected_dir = tmp_path / "sensor" / "spinach" / "v2.3.1"
    assert art_path.parent == expected_dir
    assert meta_path.parent == expected_dir
    assert art_path.is_file()
    assert meta_path.is_file()
