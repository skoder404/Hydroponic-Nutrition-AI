"""Unit tests for promoted model loading and GET /api/v1/models endpoint (T6-002).

Acceptance Criteria:
- Removing one artifact breaks only that crop (status=MODEL_UNAVAILABLE).
- Incompatible feature list is reported as SCHEMA_MISMATCH.
- /health reflects registry state ('ok', 'degraded', 'unavailable').
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.errors import ModelUnavailableError, SchemaMismatchError
from backend.app.main import create_app
from backend.app.services.model_service import (
    ModelRegistryService,
    get_model_registry_service,
    reset_model_registry_service,
)


@pytest.fixture(autouse=True)
def clean_registry_service():
    """Ensure a clean registry singleton before and after each test."""
    reset_model_registry_service()
    yield
    reset_model_registry_service()


def test_models_endpoint_exposes_promoted_models():
    """Verify GET /api/v1/models returns promoted models with required fields."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    response = client.get("/api/v1/models")
    assert response.status_code == 200

    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 3

    crops_found = {m["crop"]: m for m in models}
    assert "lettuce" in crops_found
    assert "spinach" in crops_found
    assert "tomato" in crops_found

    # Validate mandatory fields per T6-002: crop, task, family, version, status, val macro-F1, experimental
    for crop in ["lettuce", "spinach", "tomato"]:
        m = crops_found[crop]
        assert m["crop"] == crop
        assert m["task"] == "classification"
        assert m["modality"] == "sensor"
        assert m["family"] in ("DecisionTreeClassifier", "RandomForestClassifier")
        assert m["version"] == "1.0.0"
        assert m["status"] == "promoted"
        assert isinstance(m["validation_macro_f1"], float)
        assert m["validation_macro_f1"] > 0.6
        assert m["error_detail"] is None

    # Check tomato experimental flag
    assert crops_found["tomato"]["experimental"] is True
    assert crops_found["lettuce"]["experimental"] is False
    assert crops_found["spinach"]["experimental"] is False


def test_models_endpoint_filtering_and_root_alias():
    """Verify crop and modality query filtering, and root /models alias."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    # Filter by crop
    res_lettuce = client.get("/api/v1/models?crop=lettuce")
    assert res_lettuce.status_code == 200
    data_lettuce = res_lettuce.json()
    assert len(data_lettuce) == 1
    assert data_lettuce[0]["crop"] == "lettuce"

    # Filter by modality
    res_sensor = client.get("/api/v1/models?modality=sensor")
    assert res_sensor.status_code == 200
    data_sensor = res_sensor.json()
    assert len(data_sensor) >= 3
    assert all(m["modality"] == "sensor" for m in data_sensor)

    # Root alias /models
    res_root = client.get("/models")
    assert res_root.status_code == 200
    assert len(res_root.json()) == len(client.get("/api/v1/models").json())


def test_health_reflects_ok_when_all_models_healthy():
    """Verify /health reports registry='ok' and status='ok' when all models load."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["registry"] == "ok"


def test_criterion_1_removing_one_artifact_breaks_only_that_crop(tmp_path: Path):
    """Acceptance Criterion 1: Removing one artifact breaks only that crop.

    Corrupting or deleting lettuce artifact must:
    - Mark lettuce as MODEL_UNAVAILABLE.
    - Leave spinach and tomato as promoted and fully operational.
    - Reflect 'degraded' in /health.
    """
    # Create isolated copy of current models directory
    src_models = Path("models").resolve()
    mock_models_root = tmp_path / "models"
    shutil.copytree(src_models, mock_models_root)

    # Delete lettuce artifact
    lettuce_pointer_path = mock_models_root / "manifests" / "active_lettuce_sensor.json"
    lettuce_pointer = json.loads(lettuce_pointer_path.read_text(encoding="utf-8"))
    lettuce_art_path = mock_models_root / lettuce_pointer["active_artifact_path"]
    assert lettuce_art_path.is_file()
    lettuce_art_path.unlink()  # Remove lettuce artifact
    assert not lettuce_art_path.is_file()

    # Initialize service pointing to isolated test root
    service = ModelRegistryService(registry_root=mock_models_root)
    service.load_promoted_models()

    # Verify summaries
    summaries = {s.crop: s for s in service.get_model_summaries()}
    assert summaries["lettuce"].status == "MODEL_UNAVAILABLE"
    assert summaries["lettuce"].error_detail is not None
    assert "not found" in summaries["lettuce"].error_detail.lower()

    # Spinach and tomato remain intact (no cascade failure, no silent substitution)
    assert summaries["spinach"].status == "promoted"
    assert summaries["tomato"].status == "promoted"

    # Runtime access: lettuce raises ModelUnavailableError
    with pytest.raises(ModelUnavailableError) as exc_info:
        service.get_model("lettuce", "sensor")
    assert exc_info.value.code == "MODEL_UNAVAILABLE"

    # Runtime access: spinach succeeds without error
    spinach_obj, spinach_meta = service.get_model("spinach", "sensor")
    assert spinach_obj is not None
    assert spinach_meta.crop == "spinach"

    # Verify health status is degraded
    assert service.get_health_status() == "degraded"

    # Verify via TestClient endpoint
    app = create_app(configure_logging=False, model_registry_root=str(mock_models_root))
    client = TestClient(app)

    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "degraded"
    assert health_data["registry"] == "degraded"

    models_resp = client.get("/api/v1/models")
    assert models_resp.status_code == 200
    crops_map = {m["crop"]: m for m in models_resp.json()}
    assert crops_map["lettuce"]["status"] == "MODEL_UNAVAILABLE"
    assert crops_map["spinach"]["status"] == "promoted"
    assert crops_map["tomato"]["status"] == "promoted"


def test_criterion_2_incompatible_feature_list_reported_as_schema_mismatch(tmp_path: Path):
    """Acceptance Criterion 2: Incompatible feature list is reported as SCHEMA_MISMATCH.

    Altering spinach feature schema in metadata must:
    - Mark spinach as SCHEMA_MISMATCH.
    - Leave lettuce and tomato as promoted and fully operational.
    - Raise SchemaMismatchError on inference attempt.
    """
    src_models = Path("models").resolve()
    mock_models_root = tmp_path / "models"
    shutil.copytree(src_models, mock_models_root)

    # Corrupt spinach feature list in its metadata
    spinach_meta_path = (
        mock_models_root / "sensor" / "spinach" / "v1.0.0" / "metadata.json"
    )
    meta_data = json.loads(spinach_meta_path.read_text(encoding="utf-8"))
    meta_data["ordered_features"] = ["ph", "ec", "bogus_unsupported_sensor"]
    spinach_meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

    # Initialize service on altered tree
    service = ModelRegistryService(registry_root=mock_models_root)
    service.load_promoted_models()

    summaries = {s.crop: s for s in service.get_model_summaries()}
    assert summaries["spinach"].status == "SCHEMA_MISMATCH"
    assert "schema mismatch" in summaries["spinach"].error_detail.lower()

    # Lettuce and tomato remain valid
    assert summaries["lettuce"].status == "promoted"
    assert summaries["tomato"].status == "promoted"

    # Runtime access: spinach raises SchemaMismatchError
    with pytest.raises(SchemaMismatchError) as exc_info:
        service.get_model("spinach", "sensor")
    assert exc_info.value.code == "SCHEMA_MISMATCH"

    # Runtime access: lettuce works cleanly
    lettuce_obj, lettuce_meta = service.get_model("lettuce", "sensor")
    assert lettuce_obj is not None
    assert lettuce_meta.crop == "lettuce"

    # Verify endpoint reflects SCHEMA_MISMATCH
    app = create_app(configure_logging=False, model_registry_root=str(mock_models_root))
    client = TestClient(app)

    models_resp = client.get("/api/v1/models")
    assert models_resp.status_code == 200
    crops_map = {m["crop"]: m for m in models_resp.json()}
    assert crops_map["spinach"]["status"] == "SCHEMA_MISMATCH"
    assert crops_map["lettuce"]["status"] == "promoted"


def test_criterion_3_empty_registry_reflects_unavailable(tmp_path: Path):
    """Acceptance Criterion 3: Empty registry yields unavailable in health."""
    mock_models_root = tmp_path / "models"
    (mock_models_root / "manifests").mkdir(parents=True)

    service = ModelRegistryService(registry_root=mock_models_root)
    service.load_promoted_models()

    assert service.get_health_status() == "unavailable"
    assert service.get_model_summaries() == []

    app = create_app(configure_logging=False, model_registry_root=str(mock_models_root))
    client = TestClient(app)

    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "degraded"
    assert health_resp.json()["registry"] == "unavailable"
