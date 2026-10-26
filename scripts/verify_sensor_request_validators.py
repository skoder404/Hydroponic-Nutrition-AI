#!/usr/bin/env python3
"""Verification script for Per-Crop Sensor Request Validators (T3-003).

Acceptance Criteria:
1. Unknown field is rejected (extra='forbid').
2. null and 0 are treated differently (null raises MISSING_FIELD, 0 is numeric).
3. Error message names field, unit, and allowed range.

Re-run command:
    python scripts/verify_sensor_request_validators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.schemas.sensors import (
    LettuceSensorRequest,
    SpinachSensorRequest,
    TomatoSensorRequest,
    get_sensor_request_model,
    validate_sensor_request,
)


def verify_unknown_fields_rejected() -> tuple[bool, str]:
    """Criterion 1: Unknown field is rejected with extra_forbidden error."""
    test_cases = [
        (LettuceSensorRequest, {"ph": 6.0, "ec": 1.5, "water_temp": 20.0, "humidity": 60.0, "air_temp": 22.0, "soil_moisture": 45.0}),
        (SpinachSensorRequest, {"air_temperature_C": 20.0, "relative_humidity_percent": 60.0, "water_temperature_C": 18.0, "ph": 6.5, "ec_mS_cm": 2.0, "tds_ppm": 1400.0}),
        (TomatoSensorRequest, {"air_temperature_C": 22.0, "relative_humidity_percent": 65.0, "water_temperature_C": 20.0, "ph": 6.2, "ec_mS_cm": 2.5, "nitrogen_mg_L": 120.0, "phosphorus_mg_L": 45.0, "potassium_mg_L": 200.0, "n_rules_broken": 0}),
    ]

    for model_cls, payload in test_cases:
        try:
            model_cls.model_validate(payload)
            return False, f"Failed to reject unknown extra field in {model_cls.__name__}"
        except ValidationError as e:
            errors = e.errors()
            if not any(err["type"] == "extra_forbidden" for err in errors):
                return False, f"Expected extra_forbidden error type in {model_cls.__name__}, got: {errors}"

    return True, "Extra unknown fields rejected across all 3 crop models with extra='forbid'"


def verify_null_and_zero_treated_differently() -> tuple[bool, str]:
    """Criterion 2: null and 0 are treated differently."""
    # 1. Test null gives MISSING_FIELD
    try:
        LettuceSensorRequest.model_validate({"ph": None, "ec": 1.5, "water_temp": 20.0, "humidity": 60.0, "air_temp": 22.0})
        return False, "Failed to reject null value for 'ph'"
    except ValidationError as e:
        msg = e.errors()[0]["msg"]
        if "MISSING_FIELD" not in msg:
            return False, f"Expected MISSING_FIELD in error message for null, got: {msg}"

    # 2. Test 0.0 is treated as numeric float:
    # For ec (range [0.0, 6.0]), 0.0 is physically valid!
    lettuce_zero_ec = LettuceSensorRequest.model_validate({
        "ph": 6.0,
        "ec": 0.0,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    })
    if lettuce_zero_ec.ec != 0.0:
        return False, f"Expected ec to be numeric 0.0, got: {lettuce_zero_ec.ec}"

    # For ph (range [3.0, 10.0]), 0.0 is OUT_OF_BOUNDS (not MISSING_FIELD!)
    try:
        LettuceSensorRequest.model_validate({
            "ph": 0.0,
            "ec": 1.5,
            "water_temp": 20.0,
            "humidity": 60.0,
            "air_temp": 22.0,
        })
        return False, "Failed to reject out-of-bounds 0.0 for 'ph'"
    except ValidationError as e:
        msg = e.errors()[0]["msg"]
        if "OUT_OF_BOUNDS" not in msg or "MISSING_FIELD" in msg:
            return False, f"Expected OUT_OF_BOUNDS (not MISSING_FIELD) for 0.0 ph, got: {msg}"

    return True, "null and 0 are strictly distinguished (null triggers MISSING_FIELD, 0 evaluates numerically against bounds)"


def verify_error_messages_contain_field_unit_range() -> tuple[bool, str]:
    """Criterion 3: Error message names field, unit, and allowed range."""
    # Test out-of-bounds on lettuce water_temp (unit: degC, range: [0.0, 45.0])
    try:
        LettuceSensorRequest.model_validate({
            "ph": 6.0,
            "ec": 1.5,
            "water_temp": 85.0,  # Far above 45.0 degC
            "humidity": 60.0,
            "air_temp": 22.0,
        })
        return False, "Failed to reject out-of-bounds water_temp"
    except ValidationError as e:
        msg = e.errors()[0]["msg"]
        if "water_temp" not in msg:
            return False, f"Field name 'water_temp' missing from error: {msg}"
        if "degC" not in msg:
            return False, f"Unit 'degC' missing from error: {msg}"
        if "0.0" not in msg or "45.0" not in msg:
            return False, f"Allowed range '[0.0, 45.0]' missing from error: {msg}"

    # Test out-of-bounds on tomato potassium_mg_L (unit: mg/L, range: [0.0, 800.0])
    try:
        TomatoSensorRequest.model_validate({
            "air_temperature_C": 22.0,
            "relative_humidity_percent": 65.0,
            "water_temperature_C": 20.0,
            "ph": 6.2,
            "ec_mS_cm": 2.5,
            "nitrogen_mg_L": 120.0,
            "phosphorus_mg_L": 45.0,
            "potassium_mg_L": 1500.0,  # Out of range
        })
        return False, "Failed to reject out-of-bounds potassium_mg_L"
    except ValidationError as e:
        msg = e.errors()[0]["msg"]
        if "potassium_mg_L" not in msg:
            return False, f"Field name missing from error: {msg}"
        if "mg/L" not in msg:
            return False, f"Unit 'mg/L' missing from error: {msg}"
        if "800.0" not in msg:
            return False, f"Allowed range max '800.0' missing from error: {msg}"

    return True, "Field-level error messages explicitly name the field, unit, and allowed physical range"


def verify_helpers_and_convenience_functions() -> tuple[bool, str]:
    """Verify get_sensor_request_model and validate_sensor_request work as expected."""
    for crop, expected_cls in [
        ("lettuce", LettuceSensorRequest),
        ("spinach", SpinachSensorRequest),
        ("tomato", TomatoSensorRequest),
    ]:
        if get_sensor_request_model(crop) is not expected_cls:
            return False, f"Model mismatch for crop '{crop}'"

    # Test validate_sensor_request
    req = validate_sensor_request("lettuce", {
        "ph": 6.0,
        "ec": 1.5,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    })
    if not isinstance(req, LettuceSensorRequest) or req.ph != 6.0:
        return False, "validate_sensor_request failed for lettuce"

    return True, "Helper functions get_sensor_request_model and validate_sensor_request verified"


def run_all_checks() -> int:
    """Execute all verification checks."""
    print("=" * 70)
    print("Per-Crop Sensor Request Validators Verification (T3-003)")
    print("=" * 70)

    checks = [
        ("Criterion 1: Unknown field is rejected", verify_unknown_fields_rejected),
        ("Criterion 2: null and 0 are treated differently", verify_null_and_zero_treated_differently),
        ("Criterion 3: Error message names field, unit, and allowed range", verify_error_messages_contain_field_unit_range),
        ("Criterion 4: Helper functions and model factory", verify_helpers_and_convenience_functions),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {title}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED (T3-003)")
        return 0
    else:
        print("SOME VERIFICATION CHECKS FAILED (T3-003)")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
