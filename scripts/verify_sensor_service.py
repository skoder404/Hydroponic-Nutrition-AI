"""Verification script for SensorService.predict (T6-004).

Acceptance Criteria:
1. Out-of-order JSON keys still map to the correct feature order.
2. Missing field is rejected (MISSING_FIELD).
3. Tomato result carries experimental=true.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.errors import MissingFieldError
from backend.app.services.sensor_service import get_sensor_service, reset_sensor_service


def verify_criterion_1_out_of_order_keys() -> None:
    """Verify that incoming JSON keys in arbitrary orders produce identical results."""
    print("[1/3] Testing Criterion 1: Out-of-order JSON keys map to correct feature order...")
    service = get_sensor_service()

    payload_canonical = {
        "ph": 6.2,
        "ec": 1.6,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }
    payload_reversed = {
        "air_temp": 22.0,
        "humidity": 65.0,
        "water_temp": 20.0,
        "ec": 1.6,
        "ph": 6.2,
    }
    payload_shuffled = {
        "water_temp": 20.0,
        "ph": 6.2,
        "air_temp": 22.0,
        "humidity": 65.0,
        "ec": 1.6,
    }

    resp1 = service.predict("lettuce", payload_canonical)
    resp2 = service.predict("lettuce", payload_reversed)
    resp3 = service.predict("lettuce", payload_shuffled)

    if not (resp1.predicted_class == resp2.predicted_class == resp3.predicted_class):
        raise RuntimeError(
            f"Key ordering resulted in different predictions: {resp1.predicted_class} vs {resp2.predicted_class} vs {resp3.predicted_class}"
        )
    if not (resp1.confidence == resp2.confidence == resp3.confidence):
        raise RuntimeError(
            f"Key ordering resulted in different confidence: {resp1.confidence} vs {resp2.confidence}"
        )

    print(
        f"  -> Prediction class='{resp1.predicted_class}', conf={resp1.confidence}. "
        "Canonical, reversed, and shuffled key orders produced identical results."
    )


def verify_criterion_2_missing_field_rejected() -> None:
    """Verify missing required fields raise MISSING_FIELD with status 422."""
    print("[2/3] Testing Criterion 2: Missing field is rejected (MISSING_FIELD)...")
    service = get_sensor_service()

    # Lettuce missing ec
    payload_missing = {
        "ph": 6.0,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }

    try:
        service.predict("lettuce", payload_missing)
        raise RuntimeError("Expected MissingFieldError when 'ec' is omitted, but predict succeeded.")
    except MissingFieldError as exc:
        if exc.code != "MISSING_FIELD":
            raise RuntimeError(f"Expected code MISSING_FIELD, got '{exc.code}'")
        if exc.status_code != 422:
            raise RuntimeError(f"Expected HTTP 422, got {exc.status_code}")
        print(f"  -> Correctly rejected missing field with {exc.code} (HTTP {exc.status_code}): {exc}")

    # Null value must not be coerced to 0
    payload_null = {
        "ph": 6.0,
        "ec": None,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }
    try:
        service.predict("lettuce", payload_null)
        raise RuntimeError("Expected MissingFieldError when 'ec' is null, but predict succeeded.")
    except MissingFieldError as exc:
        if exc.code != "MISSING_FIELD":
            raise RuntimeError(f"Expected code MISSING_FIELD for null, got '{exc.code}'")
        print(f"  -> Null field correctly distinguished from 0.0 and rejected: {exc}")


def verify_criterion_3_tomato_experimental() -> None:
    """Verify tomato model returns experimental=True, while lettuce/spinach return False."""
    print("[3/3] Testing Criterion 3: Tomato result carries experimental=True...")
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
    if resp_tomato.experimental is not True:
        raise RuntimeError(f"Expected tomato experimental=True, got {resp_tomato.experimental}")

    lettuce_payload = {
        "ph": 6.0,
        "ec": 1.6,
        "water_temp": 20.0,
        "humidity": 65.0,
        "air_temp": 22.0,
    }
    resp_lettuce = service.predict("lettuce", lettuce_payload)
    if resp_lettuce.experimental is not False:
        raise RuntimeError(f"Expected lettuce experimental=False, got {resp_lettuce.experimental}")

    print(
        f"  -> Tomato experimental={resp_tomato.experimental}, "
        f"Lettuce experimental={resp_lettuce.experimental}."
    )


def main() -> None:
    """Run all verification checks for T6-004."""
    print("================================================================================")
    print("Verifying T6-004: SensorService.predict")
    print("================================================================================")
    try:
        reset_sensor_service()
        verify_criterion_1_out_of_order_keys()
        verify_criterion_2_missing_field_rejected()
        verify_criterion_3_tomato_experimental()
        print("\nSUCCESS: All T6-004 acceptance criteria verified successfully.")
    except Exception as exc:
        print(f"\nFAILURE: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
