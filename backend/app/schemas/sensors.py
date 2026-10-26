"""Per-crop sensor request schemas and validation (T3-003).

Built directly from ml/sensor_models/schemas.py (single source of truth):
- Strict field configuration (extra='forbid') rejecting any unknown fields.
- Explicit physical units and hard plausibility ranges.
- Optional field definitions paired with validate_default=True so 'missing' or 'null' is
  strictly distinguished from 0.0.
- Field-level validation returning explicit errors naming the field, unit, and allowed range.
- Missing required fields raise 'MISSING_FIELD' errors without numeric coercion of null to zero.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ml.sensor_models.schemas import get_sensor_schema


def validate_sensor_field(crop: str, field_name: str, value: Any) -> float:
    """Validate a sensor input value against crop schema physical bounds and non-null constraints.

    Args:
        crop: Target crop name ('lettuce', 'spinach', 'tomato').
        field_name: Sensor field name in schema.
        value: Input value to validate.

    Returns:
        Validated float value.

    Raises:
        ValueError: If value is null, non-numeric, or outside allowed physical bounds.
    """
    schema = get_sensor_schema(crop)
    unit = schema.units.get(field_name, "")
    bounds = schema.hard_bounds.get(field_name)

    if value is None:
        raise ValueError(
            f"MISSING_FIELD: Field '{field_name}' ({unit}) is required and cannot be null or omitted."
        )

    # Boolean is a subclass of int in Python, reject True/False explicitly
    if isinstance(value, bool):
        # Pydantic field validators require ValueError to construct field-level ValidationError
        raise ValueError(  # noqa: TRY004
            f"INVALID_TYPE: Field '{field_name}' ({unit}) must be a numeric float, got boolean."
        )

    try:
        val = float(value)
    except (ValueError, TypeError) as err:
        # Pydantic field validators require ValueError to construct field-level ValidationError
        raise ValueError(
            f"INVALID_TYPE: Field '{field_name}' ({unit}) must be a valid numeric float."
        ) from err

    if bounds is not None:
        lo, hi = bounds
        if val < lo or val > hi:
            raise ValueError(
                f"OUT_OF_BOUNDS: Field '{field_name}' ({unit}) value {val} is outside allowed "
                f"physical range [{lo}, {hi}] {unit}."
            )

    return val


# ==============================================================================
# Lettuce Sensor Request Validator
# ==============================================================================


class LettuceSensorRequest(BaseModel):
    """Lettuce hydroponic sensor input validation schema."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    ph: float | None = Field(
        default=None,
        description="Nutrient solution pH (unit: pH, physical range: [3.0, 10.0])",
    )
    ec: float | None = Field(
        default=None,
        description="Electrical conductivity (unit: mS/cm, physical range: [0.0, 6.0])",
    )
    water_temp: float | None = Field(
        default=None,
        description="Water temperature (unit: degC, physical range: [0.0, 45.0])",
    )
    humidity: float | None = Field(
        default=None,
        description="Relative humidity (unit: %, physical range: [0.0, 100.0])",
    )
    air_temp: float | None = Field(
        default=None,
        description="Air temperature (unit: degC, physical range: [0.0, 55.0])",
    )

    @field_validator("ph", "ec", "water_temp", "humidity", "air_temp", mode="before")
    @classmethod
    def _validate_fields(cls, v: Any, info: ValidationInfo) -> float:
        assert info.field_name is not None
        return validate_sensor_field("lettuce", info.field_name, v)

    def to_feature_dict(self) -> dict[str, float]:
        """Convert validated request to ordered feature dictionary for model inference."""
        return {
            "ph": float(self.ph),  # type: ignore[arg-type]
            "ec": float(self.ec),  # type: ignore[arg-type]
            "water_temp": float(self.water_temp),  # type: ignore[arg-type]
            "humidity": float(self.humidity),  # type: ignore[arg-type]
            "air_temp": float(self.air_temp),  # type: ignore[arg-type]
        }


# ==============================================================================
# Spinach Sensor Request Validator
# ==============================================================================


class SpinachSensorRequest(BaseModel):
    """Spinach hydroponic sensor input validation schema."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    air_temperature_C: float | None = Field(
        default=None,
        description="Greenhouse air temperature (unit: degC, physical range: [0.0, 55.0])",
    )
    relative_humidity_percent: float | None = Field(
        default=None,
        description="Greenhouse relative humidity (unit: %, physical range: [0.0, 100.0])",
    )
    water_temperature_C: float | None = Field(
        default=None,
        description="Nutrient solution water temperature (unit: degC, physical range: [0.0, 45.0])",
    )
    ph: float | None = Field(
        default=None,
        description="Nutrient solution pH (unit: pH, physical range: [3.0, 10.0])",
    )
    ec_mS_cm: float | None = Field(
        default=None,
        description="Electrical conductivity (unit: mS/cm, physical range: [0.0, 6.0])",
    )

    @field_validator(
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
        mode="before",
    )
    @classmethod
    def _validate_fields(cls, v: Any, info: ValidationInfo) -> float:
        assert info.field_name is not None
        return validate_sensor_field("spinach", info.field_name, v)

    def to_feature_dict(self) -> dict[str, float]:
        """Convert validated request to ordered feature dictionary for model inference."""
        return {
            "air_temperature_C": float(self.air_temperature_C),  # type: ignore[arg-type]
            "relative_humidity_percent": float(self.relative_humidity_percent),  # type: ignore[arg-type]
            "water_temperature_C": float(self.water_temperature_C),  # type: ignore[arg-type]
            "ph": float(self.ph),  # type: ignore[arg-type]
            "ec_mS_cm": float(self.ec_mS_cm),  # type: ignore[arg-type]
        }


# ==============================================================================
# Tomato Sensor Request Validator
# ==============================================================================


class TomatoSensorRequest(BaseModel):
    """Tomato hydroponic sensor input validation schema."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    air_temperature_C: float | None = Field(
        default=None,
        description="Greenhouse air temperature (unit: degC, physical range: [0.0, 55.0])",
    )
    relative_humidity_percent: float | None = Field(
        default=None,
        description="Greenhouse relative humidity (unit: %, physical range: [0.0, 100.0])",
    )
    water_temperature_C: float | None = Field(
        default=None,
        description="Water temperature (unit: degC, physical range: [0.0, 45.0])",
    )
    ph: float | None = Field(
        default=None,
        description="Nutrient solution pH (unit: pH, physical range: [3.0, 10.0])",
    )
    ec_mS_cm: float | None = Field(
        default=None,
        description="Electrical conductivity (unit: mS/cm, physical range: [0.0, 8.0])",
    )
    nitrogen_mg_L: float | None = Field(
        default=None,
        description="Dissolved nitrogen (unit: mg/L, physical range: [0.0, 500.0])",
    )
    phosphorus_mg_L: float | None = Field(
        default=None,
        description="Dissolved phosphorus (unit: mg/L, physical range: [0.0, 200.0])",
    )
    potassium_mg_L: float | None = Field(
        default=None,
        description="Dissolved potassium (unit: mg/L, physical range: [0.0, 800.0])",
    )

    @field_validator(
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
        "nitrogen_mg_L",
        "phosphorus_mg_L",
        "potassium_mg_L",
        mode="before",
    )
    @classmethod
    def _validate_fields(cls, v: Any, info: ValidationInfo) -> float:
        assert info.field_name is not None
        return validate_sensor_field("tomato", info.field_name, v)

    def to_feature_dict(self) -> dict[str, float]:
        """Convert validated request to ordered feature dictionary for model inference."""
        return {
            "air_temperature_C": float(self.air_temperature_C),  # type: ignore[arg-type]
            "relative_humidity_percent": float(self.relative_humidity_percent),  # type: ignore[arg-type]
            "water_temperature_C": float(self.water_temperature_C),  # type: ignore[arg-type]
            "ph": float(self.ph),  # type: ignore[arg-type]
            "ec_mS_cm": float(self.ec_mS_cm),  # type: ignore[arg-type]
            "nitrogen_mg_L": float(self.nitrogen_mg_L),  # type: ignore[arg-type]
            "phosphorus_mg_L": float(self.phosphorus_mg_L),  # type: ignore[arg-type]
            "potassium_mg_L": float(self.potassium_mg_L),  # type: ignore[arg-type]
        }


# ==============================================================================
# Model Factory and Utility Functions
# ==============================================================================

CropSensorRequest = LettuceSensorRequest | SpinachSensorRequest | TomatoSensorRequest

CROP_REQUEST_MODELS: dict[str, type[LettuceSensorRequest | SpinachSensorRequest | TomatoSensorRequest]] = {
    "lettuce": LettuceSensorRequest,
    "spinach": SpinachSensorRequest,
    "tomato": TomatoSensorRequest,
}


def get_sensor_request_model(
    crop: str,
) -> type[LettuceSensorRequest | SpinachSensorRequest | TomatoSensorRequest]:
    """Return the request validation model class for the given crop."""
    normalized_crop = crop.strip().lower()
    if normalized_crop not in CROP_REQUEST_MODELS:
        raise ValueError(
            f"Unknown crop '{crop}'. Valid crops are: {sorted(CROP_REQUEST_MODELS.keys())}"
        )
    return CROP_REQUEST_MODELS[normalized_crop]


def validate_sensor_request(
    crop: str,
    payload: dict[str, Any],
) -> LettuceSensorRequest | SpinachSensorRequest | TomatoSensorRequest:
    """Validate a raw sensor payload dictionary against the target crop schema."""
    model_cls = get_sensor_request_model(crop)
    return model_cls.model_validate(payload)
