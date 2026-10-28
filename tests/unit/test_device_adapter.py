"""Unit tests for DeviceAdapter, MockAdapter, ManualAdapter, and GET /device/status (T6-012).

Acceptance Criteria:
- [x] Stale payload is reported stale and never used
- [x] Mock data is never labelled live
- [x] Manual entry works with DEVICE_ADAPTER=manual
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.core.errors import DeviceUnavailableError
from backend.app.devices.adapter import (
    DeviceAdapter,
    ManualAdapter,
    MockAdapter,
    get_device_adapter,
    reset_device_adapter,
)
from backend.app.main import create_app


@pytest.fixture(autouse=True)
def clean_device_adapter():
    """Ensure clean device adapter singleton before and after each test."""
    reset_device_adapter()
    yield
    reset_device_adapter()


def test_mock_data_is_never_labelled_live() -> None:
    """AC 2: Mock data is never labelled live."""
    adapter = MockAdapter(max_age_s=120)
    assert adapter.source_label == "mock"
    assert adapter.source_label != "live"
    assert adapter.is_mock is True

    status = adapter.get_status()
    assert status.source == "mock"
    assert status.source != "live"
    assert status.is_mock is True
    assert status.adapter == "mock"

    # Verify through API endpoint GET /api/v1/device/status
    app = create_app(configure_logging=False)
    app.dependency_overrides[get_device_adapter] = lambda: adapter
    with TestClient(app) as client:
        resp = client.get("/api/v1/device/status")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["source"] == "mock"
        assert data["source"] != "live"
        assert data["is_mock"] is True
        assert data["adapter"] == "mock"


def test_stale_payload_is_reported_stale_and_never_used() -> None:
    """AC 1: Stale payload is reported stale and never used."""
    adapter = MockAdapter(max_age_s=60)
    adapter.set_simulate_stale(True)

    # 1. Reported as stale in status
    status = adapter.get_status()
    assert status.status == "stale"
    assert status.readings["ph"].status == "stale"
    assert status.readings["ph"].age_seconds > 60

    # 2. Never used: calling get_latest_sensor_readings() raises DeviceUnavailableError
    with pytest.raises(DeviceUnavailableError) as exc_info:
        adapter.get_latest_sensor_readings("lettuce")
    assert "stale" in str(exc_info.value).lower()

    # 3. Verify via API endpoint
    app = create_app(configure_logging=False)
    app.dependency_overrides[get_device_adapter] = lambda: adapter
    with TestClient(app) as client:
        resp = client.get("/device/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stale"


def test_manual_entry_works_with_manual_adapter() -> None:
    """AC 3: Manual entry works with DEVICE_ADAPTER=manual."""
    manual_adapter = ManualAdapter(max_age_s=120)

    # Before submitting any readings, status is 'missing'
    initial_status = manual_adapter.get_status()
    assert initial_status.status == "missing"
    with pytest.raises(DeviceUnavailableError):
        manual_adapter.get_latest_sensor_readings("lettuce")

    # Submit valid manual measurements
    manual_adapter.submit_readings(
        crop="lettuce",
        readings={
            "ph": 6.2,
            "ec": 1.8,
            "water_temp": 20.5,
            "humidity": 65.0,
            "air_temp": 22.0,
        },
    )

    # Now status is 'ok'
    status = manual_adapter.get_status()
    assert status.status == "ok"
    assert status.adapter == "manual"
    assert status.source == "manual"
    assert status.is_mock is False
    assert status.readings["ph"].value == 6.2
    assert status.readings["ph"].status == "ok"

    # Latest readings are returned cleanly
    readings = manual_adapter.get_latest_sensor_readings("lettuce")
    assert readings["ph"] == 6.2
    assert readings["ec"] == 1.8

    # Verify via POST /device/manual endpoint
    app = create_app(configure_logging=False)
    app.dependency_overrides[get_device_adapter] = lambda: manual_adapter
    with TestClient(app) as client:
        payload = {
            "crop": "lettuce",
            "readings": {
                "ph": 6.4,
                "ec": 1.9,
                "water_temp": 21.0,
                "humidity": 62.0,
                "air_temp": 23.0,
            },
        }
        post_resp = client.post("/device/manual", json=payload)
        assert post_resp.status_code == 200, post_resp.text
        data = post_resp.json()
        assert data["status"] == "ok"
        assert data["readings"]["ph"]["value"] == 6.4

        # Read back via GET /device/status
        get_resp = client.get("/device/status")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["readings"]["ph"]["value"] == 6.4



def test_invalid_and_disconnected_reporting() -> None:
    """Verify missing, disconnected, and invalid readings are reported accurately."""
    mock = MockAdapter()

    # Disconnected state
    mock.set_simulate_disconnected(True)
    assert mock.get_status().status == "disconnected"
    with pytest.raises(DeviceUnavailableError):
        mock.get_latest_sensor_readings("lettuce")

    # Invalid reading (out of physical bounds)
    mock.set_simulate_disconnected(False)
    mock.set_mock_telemetry("lettuce", {"ph": 13.5, "ec": 1.8, "water_temp": 20.0, "humidity": 60.0, "air_temp": 22.0})
    status = mock.get_status()
    assert status.readings["ph"].status == "invalid"
    assert status.status == "invalid"
