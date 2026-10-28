"""Unit tests for SensorService.predict (T6-004).

Acceptance Criteria:
- Out-of-order JSON keys still map to the correct feature order.
- Missing field is rejected (MISSING_FIELD).
- Tomato result carries experimental=true.
"""

from __future__ import annotations

import pytest

from backend.app.core.errors import InvalidInputError, MissingFieldError
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.sensor_service import (
    SensorService,
    get_sensor_service,
    reset_sensor_service,
)


@pytest.fixture(autouse=True)
def clean_sensor_service():
    """Reset singleton before and after each test."""
    reset_sensor_service()
    yield
    reset_sensor_service()


def test_criterion_1_out_of_order_keys_map_correctly():
    """Acceptance Criterion 1: Out-of-order JSON keys still map to the correct feature order."""
    service = get_sensor_service()

    # Lettuce standard payload in canonical schema order
    payload_canonical = {
        "ph": 6.2,
        "ec": 1.6,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }

    # Same payload with reversed keys
    payload_reversed = {
        "air_temp": 22.0,
        "humidity": 65.0,
        "water_temp": 20.0,
        "ec": 1.6,
        "ph": 6.2,
    }

    # Same payload with shuffled keys
    payload_shuffled = {
        "water_temp": 20.0,
        "ph": 6.2,
        "air_temp": 22.0,
        "humidity": 65.0,
        "ec": 1.6,
    }

    resp_canon = service.predict("lettuce", payload_canonical)
    resp_rev = service.predict("lettuce", payload_reversed)
    resp_shuf = service.predict("lettuce", payload_shuffled)

    assert isinstance(resp_canon, PredictionResponse)
    assert resp_canon.predicted_class == resp_rev.predicted_class == resp_shuf.predicted_class
    assert resp_canon.display_label == resp_rev.display_label == resp_shuf.display_label
    assert resp_canon.confidence == resp_rev.confidence == resp_shuf.confidence
    assert resp_canon.uncertain == resp_rev.uncertain == resp_shuf.uncertain


def test_criterion_1_spinach_out_of_order_keys():
    """Verify spinach model also handles arbitrary key ordering identically."""
    service = get_sensor_service()

    payload_a = {
        "air_temperature_C": 20.0,
        "relative_humidity_percent": 60.0,
        "water_temperature_C": 19.0,
        "ph": 6.5,
        "ec_mS_cm": 1.8,
    }

    payload_b = {
        "ec_mS_cm": 1.8,
        "ph": 6.5,
        "water_temperature_C": 19.0,
        "relative_humidity_percent": 60.0,
        "air_temperature_C": 20.0,
    }

    resp_a = service.predict("spinach", payload_a)
    resp_b = service.predict("spinach", payload_b)

    assert resp_a.predicted_class == resp_b.predicted_class
    assert resp_a.confidence == resp_b.confidence


def test_criterion_2_missing_field_is_rejected():
    """Acceptance Criterion 2: Missing field is rejected with MISSING_FIELD."""
    service = get_sensor_service()

    # Omitted 'ec'
    payload_missing_ec = {
        "ph": 6.0,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }
    with pytest.raises(MissingFieldError) as exc_info:
        service.predict("lettuce", payload_missing_ec)
    assert exc_info.value.code == "MISSING_FIELD"
    assert "ec" in str(exc_info.value).lower()

    # Null 'air_temp' (cannot be coerced to 0)
    payload_null_air_temp = {
        "ph": 6.0,
        "ec": 1.5,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": None,
    }
    with pytest.raises(MissingFieldError) as exc_info_null:
        service.predict("lettuce", payload_null_air_temp)
    assert exc_info_null.value.code == "MISSING_FIELD"
    assert "air_temp" in str(exc_info_null.value).lower()

    # Missing nitrogen in tomato
    payload_tomato_missing = {
        "air_temperature_C": 24.0,
        "relative_humidity_percent": 65.0,
        "water_temperature_C": 21.0,
        "ph": 6.0,
        "ec_mS_cm": 2.2,
        # missing nitrogen, phosphorus, potassium
    }
    with pytest.raises(MissingFieldError) as exc_info_tom:
        service.predict("tomato", payload_tomato_missing)
    assert exc_info_tom.value.code == "MISSING_FIELD"


def test_criterion_3_tomato_result_carries_experimental_true():
    """Acceptance Criterion 3: Tomato result carries experimental=true."""
    service = get_sensor_service()

    tomato_payload = {
        "air_temperature_C": 24.0,
        "relative_humidity_percent": 65.0,
        "water_temperature_C": 21.0,
        "ph": 6.0,
        "ec_mS_cm": 2.2,
        "nitrogen_mg_L": 180.0,
        "phosphorus_mg_L": 45.0,
        "potassium_mg_L": 240.0,
    }
    resp_tomato = service.predict("tomato", tomato_payload)
    assert resp_tomato.crop == "tomato"
    assert resp_tomato.experimental is True

    # Lettuce and spinach must NOT be experimental
    lettuce_payload = {
        "ph": 6.0,
        "ec": 1.6,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }
    resp_lettuce = service.predict("lettuce", lettuce_payload)
    assert resp_lettuce.crop == "lettuce"
    assert resp_lettuce.experimental is False

    spinach_payload = {
        "air_temperature_C": 20.0,
        "relative_humidity_percent": 60.0,
        "water_temperature_C": 19.0,
        "ph": 6.5,
        "ec_mS_cm": 1.8,
    }
    resp_spinach = service.predict("spinach", spinach_payload)
    assert resp_spinach.crop == "spinach"
    assert resp_spinach.experimental is False


def test_unknown_field_rejected():
    """Verify unknown fields are rejected per extra='forbid'."""
    service = get_sensor_service()

    payload_extra = {
        "ph": 6.0,
        "ec": 1.6,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
        "bogus_sensor": 99.0,
    }
    with pytest.raises(InvalidInputError) as exc_info:
        service.predict("lettuce", payload_extra)
    assert exc_info.value.code == "INVALID_INPUT"
    assert "bogus_sensor" in str(exc_info.value)


def test_prediction_confidence_semantics_and_warnings():
    """Verify confidence semantics, threshold uncertainty, and agronomic warnings."""
    service = get_sensor_service()

    # Payload with high pH outside optimal band (normal: 5.5 - 6.5 for lettuce)
    payload_high_ph = {
        "ph": 8.0,
        "ec": 1.6,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }
    resp = service.predict("lettuce", payload_high_ph)

    assert resp.confidence is not None
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.confidence_semantics == "multinomial_probability"
    assert isinstance(resp.uncertain, bool)

    # Operational warning triggered for out-of-band pH
    assert any("ph" in w.lower() for w in resp.warnings)

    # Limitations must include rule-derived labels
    assert any("rule-derived labels" in lim for lim in resp.limitations)
