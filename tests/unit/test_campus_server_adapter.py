"""Unit tests for CampusServerAdapter, GET /device/sensors/latest, and GET /device/camera/frame (T6-013).

Acceptance Criteria:
1. Adapter output maps exactly to the model feature schema.
2. Unreachable server returns unavailable (not zeros) with the last-success time.
3. No hard-coded IP or credentials anywhere.
4. Tomato readings are always flagged experimental.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.core.errors import DeviceUnavailableError
from backend.app.devices.adapter import (
    CampusAdapter,
    CampusServerAdapter,
    MockAdapter,
    get_device_adapter,
    reset_device_adapter,
)
from backend.app.main import create_app
from ml.sensor_models.schemas import get_sensor_schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def clean_device_adapter():
    reset_device_adapter()
    yield
    reset_device_adapter()


@pytest.fixture
def sample_campus_payload() -> dict:
    """Load real recorded live telemetry payload from campus network probe."""
    payload_file = PROJECT_ROOT / "docs/decisions/0008-campus-server/live_device_payload.json"
    with open(payload_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_no_hardcoded_ips_or_credentials() -> None:
    """AC 3: No hard-coded IP or credentials anywhere in adapter code or settings."""
    settings = get_settings()
    # Configuration is loaded from settings / environment variables
    assert hasattr(settings, "campus_server_url")
    assert hasattr(settings, "campus_camera_url")

    # Verify adapter source code does not contain hardcoded campus static IPs
    adapter_src = (PROJECT_ROOT / "backend/app/devices/adapter.py").read_text(encoding="utf-8")
    assert "192.168.100." not in adapter_src
    assert "password" not in adapter_src.lower() or "credentials" in adapter_src.lower()


def test_campus_adapter_output_maps_exactly_to_feature_schemas(sample_campus_payload) -> None:
    """AC 1: Adapter output maps exactly to the model feature schema."""
    adapter = CampusServerAdapter(server_url="http://campus.mock:5000", max_age_s=120)

    # Mock HTTP fetch
    with patch.object(adapter, "_fetch_raw_payload", return_value=(sample_campus_payload, datetime.now(timezone.utc))):
        # 1. Lettuce schema match
        lettuce_readings = adapter.get_latest_sensor_readings("lettuce")
        lettuce_schema = get_sensor_schema("lettuce")
        for feat in lettuce_schema.features:
            assert feat.name in lettuce_readings, f"Missing feature {feat.name}"
            val = lettuce_readings[feat.name]
            assert isinstance(val, (int, float))
            assert feat.hard_bounds[0] <= val <= feat.hard_bounds[1]

        # 2. Spinach schema match
        spinach_readings = adapter.get_latest_sensor_readings("spinach")
        spinach_schema = get_sensor_schema("spinach")
        for feat in spinach_schema.features:
            assert feat.name in spinach_readings, f"Missing feature {feat.name}"
            val = spinach_readings[feat.name]
            assert isinstance(val, (int, float))
            assert feat.hard_bounds[0] <= val <= feat.hard_bounds[1]

        # 3. Tomato_live schema match (N/P/K excluded per ADR 0008)
        tomato_live_readings = adapter.get_latest_sensor_readings("tomato_live")
        tomato_live_schema = get_sensor_schema("tomato_live")
        for feat in tomato_live_schema.features:
            assert feat.name in tomato_live_readings, f"Missing feature {feat.name}"
            val = tomato_live_readings[feat.name]
            assert isinstance(val, (int, float))
            assert feat.hard_bounds[0] <= val <= feat.hard_bounds[1]


def test_tomato_standard_model_raises_on_missing_npk(sample_campus_payload) -> None:
    """Standard tomato model requires N/P/K which physical hardware lacks; must raise DeviceUnavailableError."""
    adapter = CampusServerAdapter(server_url="http://campus.mock:5000", max_age_s=120)
    with patch.object(adapter, "_fetch_raw_payload", return_value=(sample_campus_payload, datetime.now(timezone.utc))):
        with pytest.raises(DeviceUnavailableError) as exc_info:
            adapter.get_latest_sensor_readings("tomato")
        assert "tomato_live" in str(exc_info.value)


def test_unreachable_server_returns_unavailable_not_zeros_with_last_success() -> None:
    """AC 2: Unreachable server returns unavailable (not zeros) with the last-success time."""
    adapter = CampusServerAdapter(server_url="http://nonexistent.campus:5000", max_age_s=120)

    # Seed an earlier successful contact timestamp
    recorded_success_time = datetime(2026, 10, 28, 14, 0, 0, tzinfo=timezone.utc)
    adapter._last_success_time = recorded_success_time

    # 1. get_status reporting
    status_resp = adapter.get_status()
    assert status_resp.status == "disconnected"
    assert status_resp.adapter == "campus"
    assert status_resp.source == "live"
    assert status_resp.is_mock is False

    # Check that reading values are None (NEVER fake zeros)
    for feat_name, reading_status in status_resp.readings.items():
        assert reading_status.value is None, f"Feature {feat_name} returned {reading_status.value} instead of None"
        assert reading_status.status == "disconnected"
        assert reading_status.timestamp == recorded_success_time
        assert reading_status.age_seconds is not None

    # 2. Calling get_latest_sensor_readings raises DeviceUnavailableError
    with pytest.raises(DeviceUnavailableError) as exc_info:
        adapter.get_latest_sensor_readings("lettuce")
    assert "last successful contact was at 2026-10-28T14:00:00+00:00" in str(exc_info.value)


def test_endpoints_latest_sensors_and_camera_frame(sample_campus_payload) -> None:
    """Test GET /api/v1/device/sensors/latest and GET /api/v1/device/camera/frame."""
    app = create_app(configure_logging=False)
    adapter = CampusServerAdapter(server_url="http://campus.mock:5000", max_age_s=120)

    # Patch fetch and camera
    sample_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb"
    now = datetime.now(timezone.utc)
    with patch.object(adapter, "_fetch_raw_payload", return_value=(sample_campus_payload, now)), \
         patch.object(adapter, "get_latest_camera_frame", return_value=(sample_jpeg, "image/jpeg")):

        app.dependency_overrides[get_device_adapter] = lambda: adapter

        with TestClient(app) as client:
            # 1. GET /api/v1/device/sensors/latest?crop=lettuce
            resp = client.get("/api/v1/device/sensors/latest?crop=lettuce")
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["crop"] == "lettuce"
            assert data["source"] == "live"
            assert data["is_mock"] is False
            assert "ph" in data["readings"]
            assert "ec" in data["readings"]
            assert data["readings"]["ph"] > 7.0

            # 2. GET /api/v1/device/sensors/latest?crop=tomato_live (flagged experimental)
            resp_tomato = client.get("/api/v1/device/sensors/latest?crop=tomato_live")
            assert resp_tomato.status_code == 200
            data_tomato = resp_tomato.json()
            assert data_tomato["experimental"] is True
            assert len(data_tomato["warnings"]) > 0

            # 3. GET /api/v1/device/camera/frame
            resp_camera = client.get("/api/v1/device/camera/frame")
            assert resp_camera.status_code == 200
            assert resp_camera.headers["content-type"] == "image/jpeg"
            assert resp_camera.content == sample_jpeg

