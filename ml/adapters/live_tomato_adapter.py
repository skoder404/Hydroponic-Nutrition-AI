"""Campus gateway to live-compatible tomato sensor feature adapter (T4-020).

Adheres to ADR 0008 (Campus Server Interface) and PRD D11:
- Maps live Modbus sensor telemetry from the campus gateway device to the
  5-feature schema required by the live-compatible tomato variant.
- Strict feature isolation: NEVER maps or includes nitrogen, phosphorus, or
  potassium (which are confirmed disabled/deleted in campus hardware).
- Provides the authoritative Feature Availability Table comparing server fields,
  engineering units, conversion formulas, and availability status.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ml.sensor_models.schemas import (
    TOMATO_LIVE_ORDERED_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Authoritative Feature Availability Table (Criterion 1)
# ---------------------------------------------------------------------------

FEATURE_AVAILABILITY_TABLE: Final[list[dict[str, str]]] = [
    {
        "model_feature": "air_temperature_C",
        "server_field": "temp_avg",
        "server_unit": "°C",
        "model_unit": "degC",
        "conversion": "Direct 1:1 float (fallback: temperature1/temperature2)",
        "status": "Available",
        "notes": "Verified live Modbus slave 5 (~41.5 °C).",
    },
    {
        "model_feature": "relative_humidity_percent",
        "server_field": "humidity_avg",
        "server_unit": "%",
        "model_unit": "%",
        "conversion": "Direct 1:1 float (fallback: humidity1/humidity2)",
        "status": "Available",
        "notes": "Verified live Modbus slave 5 (~34.7 %).",
    },
    {
        "model_feature": "water_temperature_C",
        "server_field": "None (unmetered)",
        "server_unit": "N/A",
        "model_unit": "degC",
        "conversion": "Thermal proxy: 8.0 + 0.55 * air_temp",
        "status": "Derived Proxy",
        "notes": "Physical campus tank lacks immersion probe; modeled from ambient thermal equilibrium.",
    },
    {
        "model_feature": "ph",
        "server_field": "ph",
        "server_unit": "pH",
        "model_unit": "pH",
        "conversion": "Direct 1:1 float",
        "status": "Available",
        "notes": "Verified live Modbus slave 1 (~7.66 pH).",
    },
    {
        "model_feature": "ec_mS_cm",
        "server_field": "ec",
        "server_unit": "µS/cm",
        "model_unit": "mS/cm",
        "conversion": "Scaled: ec / 1000.0 (if raw > 10.0)",
        "status": "Available",
        "notes": "Verified live Modbus slave 6 (~36.8 µS/cm -> 0.0368 mS/cm).",
    },
    {
        "model_feature": "nitrogen_mg_L",
        "server_field": "nitrogen",
        "server_unit": "mg/L",
        "model_unit": "mg/L",
        "conversion": "N/A — Excluded from live variant",
        "status": "Missing / Disabled",
        "notes": "Hardware sensor deleted in gateway config._deleted_keys (ADR 0008).",
    },
    {
        "model_feature": "phosphorus_mg_L",
        "server_field": "phosphorus",
        "server_unit": "mg/L",
        "model_unit": "mg/L",
        "conversion": "N/A — Excluded from live variant",
        "status": "Missing / Disabled",
        "notes": "Hardware sensor deleted in gateway config._deleted_keys (ADR 0008).",
    },
    {
        "model_feature": "potassium_mg_L",
        "server_field": "potassium",
        "server_unit": "mg/L",
        "model_unit": "mg/L",
        "conversion": "N/A — Excluded from live variant",
        "status": "Missing / Disabled",
        "notes": "Hardware sensor deleted in gateway config._deleted_keys (ADR 0008).",
    },
]


def format_feature_availability_markdown() -> str:
    """Render the feature availability table as GitHub-flavored Markdown."""
    headers = [
        "Model Feature",
        "Server Field",
        "Server Unit",
        "Model Unit",
        "Conversion / Mapping",
        "Status",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in FEATURE_AVAILABILITY_TABLE:
        lines.append(
            f"| `{row['model_feature']}` | `{row['server_field']}` | {row['server_unit']} | "
            f"{row['model_unit']} | {row['conversion']} | **{row['status']}** | {row['notes']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live Adapter Mapping (Criterion 3)
# ---------------------------------------------------------------------------


class LiveAdapterMappingError(ValueError):
    """Raised when campus telemetry payload cannot be mapped to the live feature schema."""


def map_campus_to_live_tomato_features(payload: Mapping[str, Any]) -> dict[str, float]:
    """Map raw campus gateway/server telemetry to the 5-feature live tomato schema.

    Strictly maps ONLY the features used by the live-compatible tomato variant:
    ['air_temperature_C', 'relative_humidity_percent', 'water_temperature_C', 'ph', 'ec_mS_cm'].
    Never maps or emits nitrogen, phosphorus, potassium, or prohibited metadata.

    Args:
        payload: Telemetry dictionary from campus gateway (e.g. GET /api/data or
                 cloud server GET /api/status latest object).

    Returns:
        Ordered dictionary containing strictly the 5 live-compatible features.

    Raises:
        LiveAdapterMappingError: If required telemetry measurements are completely missing.
    """
    # 1. Resolve air temperature
    air_temp: float | None = None
    for key in (
        "temp_avg",
        "temperature1",
        "temperature2",
        "air_temp",
        "air_temperature_C",
    ):
        if key in payload and payload[key] is not None:
            try:
                air_temp = float(payload[key])
                break
            except (ValueError, TypeError):
                continue
    if air_temp is None:
        raise LiveAdapterMappingError(
            "Missing air temperature telemetry ('temp_avg' not found)."
        )

    # 2. Resolve relative humidity
    humidity: float | None = None
    for key in (
        "humidity_avg",
        "humidity1",
        "humidity2",
        "humidity",
        "relative_humidity_percent",
    ):
        if key in payload and payload[key] is not None:
            try:
                humidity = float(payload[key])
                break
            except (ValueError, TypeError):
                continue
    if humidity is None:
        raise LiveAdapterMappingError(
            "Missing humidity telemetry ('humidity_avg' not found)."
        )

    # 3. Resolve pH
    ph_val: float | None = None
    if "ph" in payload and payload["ph"] is not None:
        try:
            ph_val = float(payload["ph"])
        except (ValueError, TypeError):
            pass
    if ph_val is None:
        raise LiveAdapterMappingError("Missing pH telemetry ('ph' not found).")

    # 4. Resolve EC (scale µS/cm to mS/cm if necessary per ADR 0008)
    ec_val: float | None = None
    for key in ("ec", "ec_mS_cm"):
        if key in payload and payload[key] is not None:
            try:
                raw_ec = float(payload[key])
                # Campus server Modbus sends EC in µS/cm (e.g. 36.8 µS/cm)
                # Model schema expects mS/cm: 1 mS/cm = 1000 µS/cm
                if key == "ec" and raw_ec > 10.0:
                    ec_val = raw_ec / 1000.0
                else:
                    ec_val = raw_ec
                break
            except (ValueError, TypeError):
                continue
    if ec_val is None:
        raise LiveAdapterMappingError("Missing EC telemetry ('ec' not found).")

    # 5. Resolve water temperature (immersion sensor or thermal proxy)
    water_temp: float | None = None
    for key in ("water_temp", "water_temperature_C"):
        if key in payload and payload[key] is not None:
            try:
                water_temp = float(payload[key])
                break
            except (ValueError, TypeError):
                continue
    if water_temp is None:
        # Calibrated thermal proxy from PRD D11 / Kolar pipeline:
        # reservoir water temperature closely follows ambient air temperature
        water_temp = round(8.0 + 0.55 * air_temp, 2)

    # 6. Construct feature dictionary strictly limited to variant features
    features: dict[str, float] = {
        "air_temperature_C": air_temp,
        "relative_humidity_percent": humidity,
        "water_temperature_C": water_temp,
        "ph": ph_val,
        "ec_mS_cm": ec_val,
    }

    # Verify that the keys strictly match the authorized variant feature list
    expected_names = list(TOMATO_LIVE_ORDERED_FEATURE_NAMES)
    assert list(features.keys()) == expected_names, (
        f"Live adapter output keys {list(features.keys())} do not match {expected_names}"
    )

    # Explicit check: Never include N/P/K in live adapter output
    prohibited = {
        "nitrogen_mg_L",
        "phosphorus_mg_L",
        "potassium_mg_L",
        "nitrogen",
        "phosphorus",
        "potassium",
    }
    assert not any(k in features for k in prohibited), (
        "Live adapter must never emit N/P/K features!"
    )

    return features
