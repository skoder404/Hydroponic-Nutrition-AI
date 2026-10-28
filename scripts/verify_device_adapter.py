#!/usr/bin/env python3
"""Verification script for T6-012: DeviceAdapter, MockAdapter, ManualAdapter, and GET /device/status.

Acceptance Criteria:
1. Stale payload is reported stale and never used
2. Mock data is never labelled live
3. Manual entry works with DEVICE_ADAPTER=manual
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from backend.app.core.errors import DeviceUnavailableError
from backend.app.devices.adapter import (
    ManualAdapter,
    MockAdapter,
    get_device_adapter,
    reset_device_adapter,
)
from backend.app.main import create_app


def run_verification() -> bool:
    print("[1/3] Verifying mock data is NEVER labelled live...")
    reset_device_adapter()
    mock_adapter = MockAdapter(max_age_s=120)
    assert mock_adapter.source_label == "mock", f"Expected 'mock', got {mock_adapter.source_label}"
    assert mock_adapter.source_label != "live", "MockAdapter MUST NOT be labelled live!"
    assert mock_adapter.is_mock is True, "MockAdapter must have is_mock=True"

    status = mock_adapter.get_status()
    assert status.source == "mock", f"Expected status source 'mock', got {status.source}"
    assert status.source != "live", "Mock status MUST NOT be labelled live!"
    assert status.is_mock is True

    # Test via API
    app = create_app(configure_logging=False)
    app.dependency_overrides[get_device_adapter] = lambda: mock_adapter
    client = TestClient(app)

    res = client.get("/device/status")
    assert res.status_code == 200, f"GET /device/status failed: {res.text}"
    api_data = res.json()
    assert api_data["source"] == "mock", "API response labelled mock as non-mock!"
    assert api_data["source"] != "live", "API response labelled mock as live!"
    assert api_data["is_mock"] is True
    print("  -> Passed mock data live isolation!")

    print("[2/3] Verifying stale payload is reported stale and NEVER used...")
    mock_adapter.set_simulate_stale(True)
    stale_status = mock_adapter.get_status()
    assert stale_status.status == "stale", f"Expected 'stale', got {stale_status.status}"
    assert stale_status.readings["ph"].status == "stale"
    print(f"  -> Reported status: {stale_status.status} (age: {stale_status.readings['ph'].age_seconds:.1f}s)")

    # Verify that get_latest_sensor_readings() raises DeviceUnavailableError
    stale_blocked = False
    try:
        mock_adapter.get_latest_sensor_readings("lettuce")
    except DeviceUnavailableError as exc:
        stale_blocked = True
        print(f"  -> Successfully prevented stale usage: {exc}")

    assert stale_blocked, "Stale payload was permitted to be used! Must be blocked."

    print("[3/3] Verifying manual entry works with DEVICE_ADAPTER=manual...")
    manual_adapter = ManualAdapter(max_age_s=120)
    app_manual = create_app(configure_logging=False)
    app_manual.dependency_overrides[get_device_adapter] = lambda: manual_adapter
    client_manual = TestClient(app_manual)

    # Initial check: missing
    init_res = client_manual.get("/device/status")
    assert init_res.status_code == 200
    assert init_res.json()["status"] == "missing"

    # Submit manual measurements
    post_res = client_manual.post(
        "/device/manual",
        json={
            "crop": "lettuce",
            "readings": {
                "ph": 6.3,
                "ec": 1.7,
                "water_temp": 20.0,
                "humidity": 65.0,
                "air_temp": 22.5,
            },
        },
    )
    assert post_res.status_code == 200, f"Manual submit failed: {post_res.text}"
    manual_data = post_res.json()
    assert manual_data["status"] == "ok"
    assert manual_data["source"] == "manual"
    assert manual_data["readings"]["ph"]["value"] == 6.3
    assert manual_data["readings"]["ph"]["status"] == "ok"

    # Read latest sensor readings
    readings = manual_adapter.get_latest_sensor_readings("lettuce")
    assert readings["ph"] == 6.3
    assert readings["ec"] == 1.7
    print(f"  -> Manual entry operational! Retrieved readings: {readings}")

    print("\nAll T6-012 device adapter criteria verified successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
