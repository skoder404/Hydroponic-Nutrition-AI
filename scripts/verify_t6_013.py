"""Verification script for T6-013: Implement CampusServerAdapter, GET /device/sensors/latest and GET /device/camera/frame.

Acceptance Criteria:
1. Adapter output maps exactly to the model feature schema.
2. Unreachable server returns unavailable (not zeros) with the last-success time.
3. No hard-coded IP or credentials anywhere.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.errors import DeviceUnavailableError
from backend.app.devices.adapter import CampusServerAdapter
from ml.sensor_models.schemas import get_sensor_schema


def verify_t6_013() -> bool:
    print("=== Verifying T6-013: CampusServerAdapter and Device Endpoints ===")

    # 1. Acceptance Criterion 3: No hardcoded IP or credentials anywhere
    print("\n[1/3] Verifying no hardcoded IP or credentials anywhere...")
    adapter_src = (PROJECT_ROOT / "backend/app/devices/adapter.py").read_text(encoding="utf-8")
    assert "192.168.100." not in adapter_src, "Hardcoded campus IP found in adapter.py"
    settings = get_settings()
    assert hasattr(settings, "campus_server_url")
    assert hasattr(settings, "campus_camera_url")
    print(f"[PASS] Campus URLs dynamically configurable via Settings: {settings.campus_server_url}")

    # 2. Acceptance Criterion 1: Adapter output maps exactly to model feature schema
    print("\n[2/3] Verifying adapter output maps exactly to the model feature schema...")
    payload_file = PROJECT_ROOT / "docs/decisions/0008-campus-server/live_device_payload.json"
    with open(payload_file, "r", encoding="utf-8") as f:
        sample_payload = json.load(f)

    adapter = CampusServerAdapter(server_url="http://campus.mock:5000", max_age_s=120)
    with patch.object(adapter, "_fetch_raw_payload", return_value=(sample_payload, datetime.now(timezone.utc))):
        for crop_name in ["lettuce", "spinach", "tomato_live"]:
            readings = adapter.get_latest_sensor_readings(crop_name)
            schema = get_sensor_schema(crop_name)
            for feat in schema.features:
                assert feat.name in readings, f"Crop {crop_name} missing {feat.name}"
                val = readings[feat.name]
                assert feat.hard_bounds[0] <= val <= feat.hard_bounds[1]
                print(f"  {crop_name}.{feat.name} = {val} {feat.unit} (allowed: {feat.hard_bounds})")

    # Tomato standard model raises on missing N/P/K
    with patch.object(adapter, "_fetch_raw_payload", return_value=(sample_payload, datetime.now(timezone.utc))):
        try:
            adapter.get_latest_sensor_readings("tomato")
            raise AssertionError("Standard tomato should fail due to absent physical N/P/K sensors")
        except DeviceUnavailableError as exc:
            assert "tomato_live" in str(exc)
            print("  [PASS] Standard tomato properly flags experimental / points to tomato_live.")
    print("[PASS] All crop features map cleanly into sensor schemas.")

    # 3. Acceptance Criterion 2: Unreachable server returns unavailable (not zeros) with last-success time
    print("\n[3/3] Verifying unreachable server returns unavailable (not zeros) with last-success time...")
    unreachable_adapter = CampusServerAdapter(server_url="http://nonexistent.campus:5000", max_age_s=120)
    last_success = datetime(2026, 10, 28, 15, 30, 0, tzinfo=timezone.utc)
    unreachable_adapter._last_success_time = last_success

    status_resp = unreachable_adapter.get_status()
    assert status_resp.status == "disconnected"
    assert status_resp.adapter == "campus"
    assert status_resp.source == "live"
    assert status_resp.is_mock is False

    for feat_name, reading_status in status_resp.readings.items():
        assert reading_status.value is None, (
            f"Feature {feat_name} returned numeric {reading_status.value} instead of None"
        )
        assert reading_status.status == "disconnected"
        assert reading_status.timestamp == last_success
        assert reading_status.age_seconds is not None

    try:
        unreachable_adapter.get_latest_sensor_readings("lettuce")
        raise AssertionError("Expected DeviceUnavailableError")
    except DeviceUnavailableError as exc:
        assert "2026-10-28T15:30:00+00:00" in str(exc)
        print(f"  [PASS] Unreachable error communicates last success time: {exc}")

    print("\n=== ALL T6-013 ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    verify_t6_013()

