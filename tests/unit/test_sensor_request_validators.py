"""Unit tests for per-crop sensor request validators (T3-003)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.sensors import (
    LettuceSensorRequest,
    SpinachSensorRequest,
    TomatoSensorRequest,
    get_sensor_request_model,
    validate_sensor_request,
)

# ==============================================================================
# Valid Payloads
# ==============================================================================


def test_lettuce_valid_payload() -> None:
    """Verify LettuceSensorRequest validates clean agronomic readings."""
    payload = {
        "ph": 6.2,
        "ec": 1.8,
        "water_temp": 21.5,
        "humidity": 65.0,
        "air_temp": 23.0,
    }
    req = LettuceSensorRequest.model_validate(payload)
    assert req.ph == 6.2
    assert req.ec == 1.8
    assert req.water_temp == 21.5
    assert req.humidity == 65.0
    assert req.air_temp == 23.0

    features = req.to_feature_dict()
    assert features == payload


def test_spinach_valid_payload() -> None:
    """Verify SpinachSensorRequest validates clean readings."""
    payload = {
        "air_temperature_C": 20.0,
        "relative_humidity_percent": 60.0,
        "water_temperature_C": 19.0,
        "ph": 6.5,
        "ec_mS_cm": 2.1,
    }
    req = SpinachSensorRequest.model_validate(payload)
    assert req.ph == 6.5
    assert req.ec_mS_cm == 2.1
    assert req.to_feature_dict() == payload


def test_tomato_valid_payload() -> None:
    """Verify TomatoSensorRequest validates clean 8-feature readings."""
    payload = {
        "air_temperature_C": 24.0,
        "relative_humidity_percent": 70.0,
        "water_temperature_C": 22.0,
        "ph": 6.3,
        "ec_mS_cm": 2.8,
        "nitrogen_mg_L": 150.0,
        "phosphorus_mg_L": 50.0,
        "potassium_mg_L": 250.0,
    }
    req = TomatoSensorRequest.model_validate(payload)
    assert req.nitrogen_mg_L == 150.0
    assert req.to_feature_dict() == payload


# ==============================================================================
# Criterion 1: Unknown Field Rejected (extra='forbid')
# ==============================================================================


def test_extra_field_rejected_lettuce() -> None:
    """Verify extra unknown field is rejected for lettuce."""
    payload = {
        "ph": 6.0,
        "ec": 1.8,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
        "co2": 450.0,  # Blacklisted/unknown in request
    }
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate(payload)
    errors = exc_info.value.errors()
    assert any(e["type"] == "extra_forbidden" and e["loc"] == ("co2",) for e in errors)


def test_extra_field_rejected_spinach() -> None:
    """Verify extra unknown field is rejected for spinach."""
    payload = {
        "air_temperature_C": 20.0,
        "relative_humidity_percent": 60.0,
        "water_temperature_C": 19.0,
        "ph": 6.5,
        "ec_mS_cm": 2.1,
        "tds_ppm": 1470,  # Blacklisted in request
    }
    with pytest.raises(ValidationError) as exc_info:
        SpinachSensorRequest.model_validate(payload)
    assert any(e["type"] == "extra_forbidden" for e in exc_info.value.errors())


def test_extra_field_rejected_tomato() -> None:
    """Verify extra unknown field is rejected for tomato."""
    payload = {
        "air_temperature_C": 24.0,
        "relative_humidity_percent": 70.0,
        "water_temperature_C": 22.0,
        "ph": 6.3,
        "ec_mS_cm": 2.8,
        "nitrogen_mg_L": 150.0,
        "phosphorus_mg_L": 50.0,
        "potassium_mg_L": 250.0,
        "sample_id": "S12345",  # Blacklisted in request
    }
    with pytest.raises(ValidationError) as exc_info:
        TomatoSensorRequest.model_validate(payload)
    assert any(e["type"] == "extra_forbidden" for e in exc_info.value.errors())


# ==============================================================================
# Criterion 2: null and 0 are Treated Differently
# ==============================================================================


def test_null_value_raises_missing_field() -> None:
    """Verify null is never coerced to 0 and raises MISSING_FIELD."""
    payload = {
        "ph": None,
        "ec": 1.8,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate(payload)
    msg = exc_info.value.errors()[0]["msg"]
    assert "MISSING_FIELD" in msg
    assert "Field 'ph'" in msg
    assert "cannot be null" in msg


def test_omitted_field_raises_missing_field() -> None:
    """Verify omitted field raises MISSING_FIELD."""
    payload = {
        "ph": 6.0,
        # ec is omitted
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate(payload)
    errors = exc_info.value.errors()
    assert any("MISSING_FIELD" in e["msg"] and e["loc"] == ("ec",) for e in errors)


def test_zero_value_accepted_when_within_bounds() -> None:
    """Verify 0.0 is treated as valid numeric value when within physical bounds."""
    # EC has hard bounds [0.0, 6.0] mS/cm
    payload = {
        "ph": 6.0,
        "ec": 0.0,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    }
    req = LettuceSensorRequest.model_validate(payload)
    assert req.ec == 0.0


def test_zero_value_rejected_when_outside_bounds() -> None:
    """Verify 0.0 raises OUT_OF_BOUNDS (not MISSING_FIELD) when outside physical bounds."""
    # pH has hard bounds [3.0, 10.0]
    payload = {
        "ph": 0.0,
        "ec": 1.8,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate(payload)
    msg = exc_info.value.errors()[0]["msg"]
    assert "OUT_OF_BOUNDS" in msg
    assert "MISSING_FIELD" not in msg
    assert "Field 'ph' (pH) value 0.0" in msg


# ==============================================================================
# Criterion 3: Error Message Names Field, Unit, and Allowed Range
# ==============================================================================


def test_error_message_names_field_unit_range() -> None:
    """Verify field-level error messages name field, unit, and allowed physical bounds."""
    # Lettuce water_temp: unit degC, range [0.0, 45.0]
    payload = {
        "ph": 6.0,
        "ec": 1.8,
        "water_temp": 52.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate(payload)
    msg = exc_info.value.errors()[0]["msg"]
    assert "water_temp" in msg
    assert "degC" in msg
    assert "[0.0, 45.0]" in msg


def test_error_message_tomato_bounds() -> None:
    """Verify tomato field-level error messages name field, unit, and allowed bounds."""
    # Potassium: unit mg/L, range [0.0, 800.0]
    payload = {
        "air_temperature_C": 24.0,
        "relative_humidity_percent": 70.0,
        "water_temperature_C": 22.0,
        "ph": 6.3,
        "ec_mS_cm": 2.8,
        "nitrogen_mg_L": 150.0,
        "phosphorus_mg_L": 50.0,
        "potassium_mg_L": 950.0,  # > 800.0
    }
    with pytest.raises(ValidationError) as exc_info:
        TomatoSensorRequest.model_validate(payload)
    msg = exc_info.value.errors()[0]["msg"]
    assert "potassium_mg_L" in msg
    assert "mg/L" in msg
    assert "[0.0, 800.0]" in msg


def test_non_numeric_and_boolean_types_rejected() -> None:
    """Verify non-numeric types and booleans are rejected with INVALID_TYPE."""
    # Boolean
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate({
            "ph": True,
            "ec": 1.8,
            "water_temp": 20.0,
            "humidity": 60.0,
            "air_temp": 22.0,
        })
    assert "INVALID_TYPE" in exc_info.value.errors()[0]["msg"]

    # String non-numeric
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate({
            "ph": "acidic",
            "ec": 1.8,
            "water_temp": 20.0,
            "humidity": 60.0,
            "air_temp": 22.0,
        })
    assert "INVALID_TYPE" in exc_info.value.errors()[0]["msg"]


def test_multiple_field_errors_reported_together() -> None:
    """Verify multiple invalid fields produce multiple distinct field-level errors."""
    payload = {
        "ph": None,      # MISSING_FIELD
        "ec": -1.0,      # OUT_OF_BOUNDS (< 0.0)
        "water_temp": 60.0, # OUT_OF_BOUNDS (> 45.0)
        "humidity": 60.0,
        "air_temp": 22.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        LettuceSensorRequest.model_validate(payload)
    errors = exc_info.value.errors()
    assert len(errors) == 3
    locs = {e["loc"][0] for e in errors}
    assert locs == {"ph", "ec", "water_temp"}


def test_factory_and_convenience_functions() -> None:
    """Verify get_sensor_request_model and validate_sensor_request."""
    assert get_sensor_request_model("lettuce") is LettuceSensorRequest
    assert get_sensor_request_model("spinach") is SpinachSensorRequest
    assert get_sensor_request_model("tomato") is TomatoSensorRequest

    with pytest.raises(ValueError, match="Unknown crop 'kale'"):
        get_sensor_request_model("kale")

    req = validate_sensor_request("lettuce", {
        "ph": 6.0,
        "ec": 1.8,
        "water_temp": 20.0,
        "humidity": 60.0,
        "air_temp": 22.0,
    })
    assert isinstance(req, LettuceSensorRequest)
    assert req.ph == 6.0
