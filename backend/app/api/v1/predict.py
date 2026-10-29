"""Inference endpoints for sensor telemetry and plant leaf imagery (T6-005, T6-007).

Provides:
- POST /predict/sensor: Accepts crop name, sensor readings, and optional engineering units.
  Validates units and values, runs SensorService.predict, and returns PredictionResponse
  with display label, confidence, timestamp, model version, and agronomic limitations.
- POST /predict/image: Multipart route accepting crop, required input_mode ('leaf' or 'plant'),
  optional model_family override, and an uploaded plant image. Executes ImageService per-leaf
  segmentation and classification, returning per-leaf predictions, uncertainty flags, and summary.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from backend.app.core.errors import (
    InvalidInputError,
    MissingFieldError,
    ModelUnavailableError,
)
from backend.app.core.uploads import validate_upload_file
from backend.app.schemas.prediction import (
    PredictionResponse,
    PredictSensorRequest,
)
from backend.app.services.image_service import (
    ImageService,
    get_image_service,
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


def get_image_service_dep() -> ImageService:
    """Dependency provider returning singleton ImageService."""
    return get_image_service()


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


@router.post(
    "/image",
    response_model=PredictionResponse,
    summary="Predict foliar conditions and nutrient deficiencies from plant images",
    description=(
        "Accepts crop type, required input_mode ('leaf' or 'plant'), optional model_family override, "
        "and an image file (multipart/form-data). Executes per-leaf instance segmentation "
        "and classification, returning per-leaf predictions, uncertainty flags, and plant summary."
    ),
)
async def predict_image(
    request: Request,
    crop: Annotated[
        str, Form(description="Target crop cultivar ('lettuce', 'spinach', 'tomato')")
    ],
    input_mode: Annotated[
        str | None,
        Form(
            description="Processing mode: 'leaf' for single leaf, 'plant' for whole plant"
        ),
    ] = None,
    file: Annotated[
        UploadFile | None,
        File(
            description="Uploaded plant image (multipart field name 'file' or 'image')"
        ),
    ] = None,
    image: Annotated[
        UploadFile | None,
        File(
            description="Uploaded plant image (multipart field name 'image' or 'file')"
        ),
    ] = None,
    model_family: Annotated[
        str | None,
        Form(description="Optional model family architecture override"),
    ] = None,
    provenance: Annotated[
        Literal["live", "manual", "imported"],
        Form(description="Source provenance of the image upload"),
    ] = "manual",
    mask_threshold: Annotated[
        float,
        Form(description="SAM2 mask binarization threshold"),
    ] = 0.50,
    max_leaves: Annotated[
        int,
        Form(description="Maximum leaves to assess per plant in plant mode"),
    ] = 10,
    image_service: Annotated[ImageService, Depends(get_image_service_dep)] = None,  # type: ignore[assignment]
) -> PredictionResponse:
    """Execute per-leaf image inference for the specified crop."""
    # 1. Validate input_mode (AC 1: Missing input_mode is rejected with 422)
    if input_mode is None or not str(input_mode).strip():
        raise MissingFieldError(
            message="Field 'input_mode' is required ('leaf' or 'plant').",
            details={"field": "input_mode", "code": "MISSING_FIELD"},
        )

    clean_input_mode = str(input_mode).strip().lower()
    if clean_input_mode not in ("leaf", "plant"):
        raise InvalidInputError(
            message=f"Invalid input_mode '{input_mode}'. Must be 'leaf' or 'plant'.",
            details={
                "input_mode": input_mode,
                "allowed_modes": ["leaf", "plant"],
                "code": "INVALID_INPUT",
            },
        )

    # 2. Validate crop and check promoted model availability (AC 3: Unavailable family is reported clearly)
    clean_crop = crop.strip().lower()
    try:
        _, metadata = image_service.registry_service.get_model(
            clean_crop, modality="image"
        )
    except Exception as exc:
        raise ModelUnavailableError(
            message=f"No promoted image model available for crop '{clean_crop}': {exc}",
            details={
                "crop": clean_crop,
                "modality": "image",
                "code": "MODEL_UNAVAILABLE",
            },
        ) from exc

    if model_family is not None and str(model_family).strip():
        clean_req_family = str(model_family).strip().lower()
        promoted_family = metadata.family
        allowed_aliases = {
            promoted_family.lower(),
            "efficientnet",
            "efficientnet_b0",
            "efficientnetb0",
        }
        if clean_req_family not in allowed_aliases:
            raise ModelUnavailableError(
                message=(
                    f"Model family '{model_family}' is not available or not promoted for crop '{clean_crop}'. "
                    f"Available promoted image model family: '{promoted_family}'."
                ),
                details={
                    "crop": clean_crop,
                    "requested_family": model_family,
                    "available_families": [promoted_family],
                    "code": "MODEL_UNAVAILABLE",
                },
            )

    # 3. Validate image upload file
    upload_file = file if file is not None else image
    if upload_file is None:
        raise MissingFieldError(
            message="An image file is required in multipart form data (field 'file' or 'image').",
            details={"field": "file", "code": "MISSING_FIELD"},
        )

    validated_upload = await validate_upload_file(upload_file)

    # 4. Execute image pipeline
    prediction = image_service.predict(
        image=validated_upload.saved_path,
        crop=clean_crop,
        input_mode=clean_input_mode,  # type: ignore[arg-type]
        model_family=model_family,
        provenance=provenance,
        mask_threshold=mask_threshold,
        max_leaves=max_leaves,
    )

    return prediction
