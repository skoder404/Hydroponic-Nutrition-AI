"""Unit tests for POST /predict/sensor endpoint (T6-005).

Acceptance Criteria:
- [x] Valid request returns display label and limitations
- [x] Invalid unit returns 422 INVALID_INPUT
- [x] Response includes model version and timestamp
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.main import create_app
from backend.app.services.model_service import reset_model_registry_service
from backend.app.services.sensor_service import reset_sensor_service


@pytest.fixture(autouse=True)
def clean_environment():
    """Reset singletons and rate limiter before and after each test."""
    reset_model_registry_service()
    reset_sensor_service()
    get_rate_limiter().reset()
    yield
    reset_model_registry_service()
    reset_sensor_service()
    get_rate_limiter().reset()


@pytest.fixture
def client():
    """Create a FastAPI test client with loaded promoted models."""
    app = create_app(configure_logging=False)
    with TestClient(app) as test_client:
        yield test_client


def test_valid_sensor_prediction_lettuce(client: TestClient):
    """AC 1 & AC 3: Valid request returns display label, limitations, model version, and timestamp."""
    payload = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.0,
            "ec": 1.8,
            "water_temp": 21.0,
            "humidity": 65.0,
            "air_temp": 23.0,
        },
        "units": {
            "ph": "pH",
            "ec": "mS/cm",
            "water_temp": "degC",
            "humidity": "%",
            "air_temp": "degC",
        },
        "provenance": "manual",
    }

    response = client.post("/api/v1/predict/sensor", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # AC 1: Valid request returns display label and limitations
    assert "display_label" in data
    assert isinstance(data["display_label"], str)
    assert len(data["display_label"]) > 0

    assert "limitations" in data
    assert isinstance(data["limitations"], list)
    assert len(data["limitations"]) > 0
    # Must include 'rule-derived labels'
    assert any("rule-derived labels" in lim for lim in data["limitations"])

    # AC 3: Response includes model version and timestamp
    assert "model_version" in data
    assert data["model_version"].startswith("v")
    assert "timestamp" in data
    assert isinstance(data["timestamp"], str)

    # Core response envelope fields
    assert data["crop"] == "lettuce"
    assert data["modality"] == "sensor"
    assert data["predicted_class"] is not None
    assert "confidence" in data
    assert "uncertain" in data
    assert "experimental" in data
    assert data["experimental"] is False


def test_valid_sensor_prediction_nested_units(client: TestClient):
    """Verify nested {value, unit} readings format."""
    payload = {
        "crop": "spinach",
        "readings": {
            "air_temperature_C": {"value": 21.0, "unit": "degC"},
            "relative_humidity_percent": {"value": 60.0, "unit": "%"},
            "water_temperature_C": {"value": 19.5, "unit": "degC"},
            "ph": {"value": 6.5, "unit": "pH"},
            "ec_mS_cm": {"value": 1.7, "unit": "mS/cm"},
        },
    }

    # Route mounted without /api/v1 prefix as well
    response = client.post("/predict/sensor", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["crop"] == "spinach"
    assert "display_label" in data
    assert any("rule-derived labels" in lim for lim in data["limitations"])
    assert data["model_version"] is not None
    assert data["timestamp"] is not None


def test_invalid_unit_returns_422_invalid_input(client: TestClient):
    """AC 2: Invalid unit returns 422 INVALID_INPUT."""
    # Top-level units invalid unit test (e.g. Fahrenheit instead of Celsius)
    payload_top = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.0,
            "ec": 1.8,
            "water_temp": 21.0,
            "humidity": 65.0,
            "air_temp": 23.0,
        },
        "units": {
            "water_temp": "degF",  # Invalid! Expected degC
        },
    }

    response = client.post("/api/v1/predict/sensor", json=payload_top)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "INVALID_INPUT"
    assert "Invalid unit" in body["error"]["message"]
    assert "expected_unit" in body["error"]["details"]
    assert body["error"]["details"]["expected_unit"] == "degC"


def test_invalid_nested_unit_returns_422_invalid_input(client: TestClient):
    """AC 2: Nested unit mismatch also returns 422 INVALID_INPUT."""
    payload_nested = {
        "crop": "tomato",
        "readings": {
            "air_temperature_C": {"value": 24.0, "unit": "degC"},
            "relative_humidity_percent": {"value": 70.0, "unit": "%"},
            "water_temperature_C": {"value": 22.0, "unit": "degC"},
            "ph": {"value": 6.2, "unit": "pH"},
            "ec_mS_cm": {"value": 2.2, "unit": "uS/cm"},  # Invalid! Expected mS/cm
            "nitrogen_mg_L": {"value": 150.0, "unit": "mg/L"},
            "phosphorus_mg_L": {"value": 50.0, "unit": "mg/L"},
            "potassium_mg_L": {"value": 200.0, "unit": "mg/L"},
        },
    }

    response = client.post("/api/v1/predict/sensor", json=payload_nested)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "INVALID_INPUT"
    assert "Invalid unit 'uS/cm'" in body["error"]["message"]


def test_missing_required_field_returns_422_missing_field(client: TestClient):
    """Verify that omitting a mandatory sensor reading returns 422 MISSING_FIELD."""
    payload = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.0,
            "ec": 1.8,
            # water_temp is missing
            "humidity": 65.0,
            "air_temp": 23.0,
        },
    }

    response = client.post("/api/v1/predict/sensor", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "MISSING_FIELD"
    assert "water_temp" in body["error"]["message"]


def test_unknown_crop_returns_422_invalid_input(client: TestClient):
    """Verify that an unsupported crop returns 422 INVALID_INPUT."""
    payload = {
        "crop": "kale",
        "readings": {"ph": 6.0},
    }

    response = client.post("/api/v1/predict/sensor", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "INVALID_INPUT"
    assert "Unknown crop 'kale'" in body["error"]["message"]


def test_out_of_bounds_reading_returns_422_invalid_input(client: TestClient):
    """Verify physical bounds validation returns 422 INVALID_INPUT."""
    payload = {
        "crop": "lettuce",
        "readings": {
            "ph": 16.0,  # Physically impossible pH > 14
            "ec": 1.8,
            "water_temp": 21.0,
            "humidity": 65.0,
            "air_temp": 23.0,
        },
    }

    response = client.post("/api/v1/predict/sensor", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "INVALID_INPUT"


def test_tomato_prediction_carries_experimental_flag(client: TestClient):
    """Verify tomato prediction returns experimental=True per project specification."""
    payload = {
        "crop": "tomato",
        "readings": {
            "air_temperature_C": 24.0,
            "relative_humidity_percent": 70.0,
            "water_temperature_C": 22.0,
            "ph": 6.2,
            "ec_mS_cm": 2.2,
            "nitrogen_mg_L": 150.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 200.0,
        },
    }

    response = client.post("/api/v1/predict/sensor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "tomato"
    assert data["experimental"] is True
    assert "display_label" in data
    assert any("rule-derived labels" in lim for lim in data["limitations"])


def test_rate_limit_burst_returns_429(monkeypatch: pytest.MonkeyPatch):
    """Verify rate limiting applies to /predict/sensor and returns 429 when quota exceeded."""
    monkeypatch.setenv("RATE_LIMIT_STRICT", "2/minute")
    get_settings.cache_clear()

    app = create_app(configure_logging=False)
    payload = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.0,
            "ec": 1.8,
            "water_temp": 21.0,
            "humidity": 65.0,
            "air_temp": 23.0,
        },
    }

    with TestClient(app) as test_client:
        headers = {"X-Forwarded-For": "198.51.100.42"}

        # First 2 requests succeed
        res1 = test_client.post("/api/v1/predict/sensor", json=payload, headers=headers)
        assert res1.status_code == 200

        res2 = test_client.post("/api/v1/predict/sensor", json=payload, headers=headers)
        assert res2.status_code == 200

        # Third burst request exceeds quota -> 429 RATE_LIMIT_EXCEEDED
        res3 = test_client.post("/api/v1/predict/sensor", json=payload, headers=headers)
        assert res3.status_code == 429
        body = res3.json()
        assert "error" in body
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert res3.headers.get("Retry-After") is not None

    get_settings.cache_clear()
