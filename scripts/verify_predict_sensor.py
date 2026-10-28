#!/usr/bin/env python3
"""Verification script for T6-005: Add POST /predict/sensor.

Validates the acceptance criteria:
1. Valid request returns display label and limitations (including 'rule-derived labels')
2. Invalid unit returns 422 INVALID_INPUT
3. Response includes model version and timestamp

Usage:
    python scripts/verify_predict_sensor.py
"""

from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.core.rate_limit import get_rate_limiter
from backend.app.main import create_app
from backend.app.services.model_service import (
    reset_model_registry_service,
)
from backend.app.services.sensor_service import reset_sensor_service


def verify_criterion_1_valid_request(client: TestClient) -> tuple[bool, str]:
    """Criterion 1: Valid request returns display label and limitations (including 'rule-derived labels')."""
    payload = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.2,
            "ec": 1.6,
            "water_temp": 20.0,
            "humidity": 65.0,
            "air_temp": 22.0,
        },
        "units": {
            "ph": "pH",
            "ec": "mS/cm",
            "water_temp": "degC",
            "humidity": "%",
            "air_temp": "degC",
        },
    }

    res = client.post("/api/v1/predict/sensor", json=payload)
    if res.status_code != 200:
        return False, f"FAIL: Expected status 200, got {res.status_code}: {res.text}"

    data = res.json()
    if not data.get("display_label"):
        return False, "FAIL: Response missing 'display_label'."

    limitations = data.get("limitations", [])
    if not isinstance(limitations, list) or len(limitations) == 0:
        return False, "FAIL: Response missing 'limitations' list."

    if not any("rule-derived labels" in lim for lim in limitations):
        return (
            False,
            f"FAIL: Limitations list does not contain 'rule-derived labels'. Got: {limitations}",
        )

    return (
        True,
        f"PASS: Valid request returns display_label='{data['display_label']}' and limitations (contains 'rule-derived labels').",
    )


def verify_criterion_2_invalid_unit(client: TestClient) -> tuple[bool, str]:
    """Criterion 2: Invalid unit returns 422 INVALID_INPUT."""
    # Test top-level unit mismatch
    payload_top = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.2,
            "ec": 1.6,
            "water_temp": 20.0,
            "humidity": 65.0,
            "air_temp": 22.0,
        },
        "units": {
            "water_temp": "degF",  # Expected degC
        },
    }

    res_top = client.post("/api/v1/predict/sensor", json=payload_top)
    if res_top.status_code != 422:
        return (
            False,
            f"FAIL: Expected status 422 for invalid unit, got {res_top.status_code}: {res_top.text}",
        )

    body_top = res_top.json()
    error_top = body_top.get("error", {})
    if error_top.get("code") != "INVALID_INPUT":
        return (
            False,
            f"FAIL: Expected error code INVALID_INPUT, got: {error_top.get('code')}",
        )

    # Test nested unit mismatch
    payload_nested = {
        "crop": "lettuce",
        "readings": {
            "ph": 6.2,
            "ec": {"value": 1.6, "unit": "dS/m"},  # Expected mS/cm
            "water_temp": 20.0,
            "humidity": 65.0,
            "air_temp": 22.0,
        },
    }

    res_nested = client.post("/api/v1/predict/sensor", json=payload_nested)
    if res_nested.status_code != 422:
        return (
            False,
            f"FAIL: Expected status 422 for nested invalid unit, got {res_nested.status_code}: {res_nested.text}",
        )

    body_nested = res_nested.json()
    error_nested = body_nested.get("error", {})
    if error_nested.get("code") != "INVALID_INPUT":
        return (
            False,
            f"FAIL: Expected error code INVALID_INPUT for nested unit, got: {error_nested.get('code')}",
        )

    return (
        True,
        "PASS: Invalid unit returns HTTP 422 with error code INVALID_INPUT for both top-level and nested units.",
    )


def verify_criterion_3_metadata(client: TestClient) -> tuple[bool, str]:
    """Criterion 3: Response includes model version and timestamp."""
    payload = {
        "crop": "spinach",
        "readings": {
            "air_temperature_C": 21.0,
            "relative_humidity_percent": 60.0,
            "water_temperature_C": 19.5,
            "ph": 6.5,
            "ec_mS_cm": 1.7,
        },
    }

    res = client.post("/predict/sensor", json=payload)
    if res.status_code != 200:
        return False, f"FAIL: Expected status 200, got {res.status_code}: {res.text}"

    data = res.json()
    model_version = data.get("model_version")
    if not model_version or not str(model_version).strip():
        return False, "FAIL: Response missing 'model_version'."

    timestamp = data.get("timestamp")
    if not timestamp or not str(timestamp).strip():
        return False, "FAIL: Response missing 'timestamp'."

    return (
        True,
        f"PASS: Response includes model_version='{model_version}' and timestamp='{timestamp}'.",
    )


def main() -> int:
    """Run all verification checks for T6-005."""
    print("=" * 70)
    print("Verifying T6-005: Add POST /predict/sensor")
    print("=" * 70)

    reset_model_registry_service()
    reset_sensor_service()
    get_rate_limiter().reset()

    app = create_app(configure_logging=False)
    client = TestClient(app)

    checks = [
        (
            "Criterion 1: Valid request returns display label and limitations",
            verify_criterion_1_valid_request,
        ),
        (
            "Criterion 2: Invalid unit returns 422 INVALID_INPUT",
            verify_criterion_2_invalid_unit,
        ),
        (
            "Criterion 3: Response includes model version and timestamp",
            verify_criterion_3_metadata,
        ),
    ]

    all_passed = True
    for title, check_func in checks:
        print(f"\nChecking {title}...")
        passed, msg = check_func(client)
        print(f"  {msg}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY (T6-005).")
        print("=" * 70)
        return 0
    else:
        print("VERIFICATION FAILED FOR ONE OR MORE CRITERIA.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
