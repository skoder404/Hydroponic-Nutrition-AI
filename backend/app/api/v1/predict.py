"""Inference endpoints for sensor telemetry and plant leaf imagery (T6-005).

Provides:
- POST /predict/sensor: Accepts crop name, sensor readings, and optional engineering units.
  Validates units and values, runs SensorService.predict, and returns PredictionResponse
  with display label, confidence, timestamp, model version, and agronomic limitations.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from backend.app.core.errors import InvalidInputError
from backend.app.schemas.prediction import (
    PredictionResponse,
    PredictSensorRequest,
)
from backend.app.services.sensor_service import (
    SensorService,
    get_sensor_service,
)
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


def get_sensor_service_dep() -> SensorService:
    """Dependency provider returning singleton SensorService."""
    return get_sensor_service()


@router.post(
    "/sensor",
    response_model=PredictionResponse,
    summary="Predict crop nutrient deficiencies or operational status from sensor readings",
    description=(
        "Accepts crop type and sensor telemetry with units. Validates physical plausibility "
        "and engineering units, executes promoted model inference, and returns calibrated "
        "prediction with display label, model metadata, and agronomic limitations."
    ),
)
def predict_sensor(
    request: Request,
    payload: PredictSensorRequest,
    sensor_service: Annotated[SensorService, Depends(get_sensor_service_dep)],
) -> PredictionResponse:
    """Execute sensor inference for the specified crop."""
    clean_crop = payload.crop.strip().lower()

    try:
        schema = get_sensor_schema(clean_crop)
    except ValueError as err:
        raise InvalidInputError(
            message=f"Unknown crop '{clean_crop}'. Supported crops: lettuce, spinach, tomato, tomato_live.",
            details={"crop": clean_crop, "code": "INVALID_INPUT"},
        ) from err

    # 1. Validate top-level units dictionary if provided
    # Acceptance Criterion 2: Invalid unit returns 422 INVALID_INPUT
    if payload.units:
        for feat, unit_val in payload.units.items():
            expected_unit = schema.units.get(feat)
            if expected_unit is not None and str(unit_val).strip() != expected_unit:
                raise InvalidInputError(
                    message=f"Invalid unit '{unit_val}' for feature '{feat}'. Expected unit: '{expected_unit}'.",
                    details={
                        "crop": clean_crop,
                        "feature": feat,
                        "provided_unit": unit_val,
                        "expected_unit": expected_unit,
                        "code": "INVALID_INPUT",
                    },
                )

    # 2. Extract values and validate per-reading units if supplied as {value: ..., unit: ...}
    unpacked_readings: dict[str, Any] = {}
    for feat, item in payload.readings.items():
        if isinstance(item, dict):
            if "unit" in item:
                unit_val = item["unit"]
                expected_unit = schema.units.get(feat)
                if expected_unit is not None and str(unit_val).strip() != expected_unit:
                    raise InvalidInputError(
                        message=f"Invalid unit '{unit_val}' for feature '{feat}'. Expected unit: '{expected_unit}'.",
                        details={
                            "crop": clean_crop,
                            "feature": feat,
                            "provided_unit": unit_val,
                            "expected_unit": expected_unit,
                            "code": "INVALID_INPUT",
                        },
                    )
            unpacked_readings[feat] = item.get("value")
        else:
            unpacked_readings[feat] = item

    # 3. Call SensorService to execute prediction
    prediction = sensor_service.predict(
        crop=clean_crop,
        readings=unpacked_readings,
        provenance=payload.provenance,
    )

    return prediction
