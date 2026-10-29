"""Unit tests for POST /analyze multi-modal endpoint (T6-011).

Acceptance Criteria:
- T6-011 AC1: Image failure does not block the sensor result
- T6-011 AC2: not_assessed explains what was skipped
- T6-011 AC3: Response has no merged score
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.errors import SegmentationFailedError
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.devices.adapter import reset_device_adapter
from backend.app.main import create_app
from backend.app.schemas.prediction import LeafPrediction, PredictionResponse, PredictionSummary
from backend.app.schemas.recommendation import RecommendationRead
from backend.app.services.image_service import reset_image_service
from backend.app.services.model_service import reset_model_registry_service
from backend.app.services.sensor_service import reset_sensor_service


@pytest.fixture(autouse=True)
def clean_environment():
    """Reset singletons and rate limiter before and after each test."""
    reset_model_registry_service()
    reset_sensor_service()
    reset_image_service()
    reset_device_adapter()
    get_rate_limiter().reset()
    yield
    reset_model_registry_service()
    reset_sensor_service()
    reset_image_service()
    reset_device_adapter()
    get_rate_limiter().reset()


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    app = create_app(configure_logging=False)
    with TestClient(app) as test_client:
        yield test_client


def _make_dummy_image_bytes(format: str = "PNG") -> bytes:
    img = Image.new("RGB", (200, 200), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_analyze_image_failure_does_not_block_sensor_result(client: TestClient):
    """T6-011 AC1: Image failure does not block the sensor result (partial success)."""
    sensor_readings = json.dumps({
        "ph": 6.2,
        "ec": 1.8,
        "water_temp": 21.0,
        "humidity": 65.0,
        "air_temp": 23.0,
    })

    # Simulate segmentation failure on image
    with patch(
        "backend.app.services.image_service.ImageService.predict",
        side_effect=SegmentationFailedError("No distinct leaves could be segmented."),
    ):
        img_bytes = _make_dummy_image_bytes()
        response = client.post(
            "/api/v1/analyze",
            data={"crop": "lettuce", "sensor_readings": sensor_readings},
            files={"file": ("corrupt.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()

        # Sensor prediction returned successfully
        assert data["sensor"] is not None
        assert data["sensor"]["crop"] == "lettuce"
        assert data["sensor"]["predicted_class"] == "Normal"

        # Image is None, and failure is documented in not_assessed
        assert data["image"] is None
        assert len(data["not_assessed"]) >= 1
        img_not_assessed = next(
            (item for item in data["not_assessed"] if item["modality"] == "image"), None
        )
        assert img_not_assessed is not None
        assert "No distinct leaves" in img_not_assessed["reason"]


def test_analyze_not_assessed_explains_what_was_skipped(client: TestClient):
    """T6-011 AC2: not_assessed explains what was skipped when a modality is omitted."""
    # Sensor only (no image provided)
    sensor_readings = json.dumps({
        "ph": 6.2,
        "ec": 1.8,
        "water_temp": 21.0,
        "humidity": 65.0,
        "air_temp": 23.0,
    })

    response = client.post(
        "/api/v1/analyze",
        data={"crop": "lettuce", "sensor_readings": sensor_readings},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sensor"] is not None
    assert data["image"] is None

    # not_assessed explains omitted image
    assert len(data["not_assessed"]) >= 1
    skipped_item = next(
        (item for item in data["not_assessed"] if item["modality"] == "image"), None
    )
    assert skipped_item is not None
    assert "No foliar plant image provided" in skipped_item["reason"]


def test_analyze_response_has_no_merged_score(client: TestClient):
    """T6-011 AC3: Response has no merged score or contradiction field (Overview §19)."""
    sensor_readings = json.dumps({
        "ph": 6.2,
        "ec": 1.8,
        "water_temp": 21.0,
        "humidity": 65.0,
        "air_temp": 23.0,
    })
    img_bytes = _make_dummy_image_bytes()

    dummy_img_pred = PredictionResponse(
        crop="lettuce",
        modality="image",
        predicted_class="Healthy",
        display_label="Healthy Foliage",
        confidence=None,
        confidence_semantics=None,
        uncertain=False,
        experimental=False,
        model_id="lettuce_image_v1.0.0",
        model_version="1.0.0",
        timestamp="2026-10-27T18:00:00Z",
        provenance="manual",
        warnings=[],
        limitations=[],
        input_mode="plant",
        leaves=[],
        summary=PredictionSummary(leaf_count=0, class_counts={}),
    )

    with patch("backend.app.services.image_service.ImageService.predict", return_value=dummy_img_pred):
        response = client.post(
            "/api/v1/analyze",
            data={"crop": "lettuce", "sensor_readings": sensor_readings},
            files={"file": ("plant.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()

        # Check invariants
        assert "merged_score" not in data
        assert "combined_score" not in data
        assert "contradiction" not in data
        assert "contradiction_alert" not in data

        # Modalities are strictly separate
        assert data["sensor"] is not None
        assert data["image"] is not None
        assert data["crop"] == "lettuce"
