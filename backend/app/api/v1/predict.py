"""Inference endpoints for sensor telemetry and plant leaf imagery (T6-005, T6-007, T6-014, T6-015).

Provides:
- POST /predict/sensor: Accepts crop name, sensor readings, and optional engineering units.
  Validates units and values, runs SensorService.predict, and returns PredictionResponse.
- POST /predict/image: Accepts multipart upload or campus camera frame flag, validates input_mode,
  verifies model availability, and executes non-blocking off-threadpool inference via ImageService.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from backend.app.core.errors import (
    DeviceUnavailableError,
    InvalidInputError,
    MissingFieldError,
    ModelUnavailableError,
)
from backend.app.core.inference import run_blocking_inference
from backend.app.core.rate_limit import RateLimiter, get_rate_limiter
from backend.app.core.uploads import validate_and_save_image, validate_upload_file
from backend.app.devices.adapter import DeviceAdapter, get_device_adapter
from backend.app.schemas.prediction import (
    PredictionResponse,
    PredictSensorRequest,
)
from backend.app.services.image_service import ImageService, get_image_service
from backend.app.services.sensor_service import SensorService, get_sensor_service
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])

SUPPORTED_IMAGE_CROPS: set[str] = {"lettuce", "spinach", "tomato"}


def get_sensor_service_dep() -> SensorService:
    """Dependency provider returning singleton SensorService."""
    return get_sensor_service()


def get_image_service_dep() -> ImageService:
    """Dependency provider returning singleton ImageService."""
    return get_image_service()


def get_device_adapter_dep() -> DeviceAdapter:
    """Dependency provider returning configured DeviceAdapter."""
    return get_device_adapter()


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
async def predict_sensor(
    request: Request,
    payload: PredictSensorRequest,
    sensor_service: Annotated[SensorService, Depends(get_sensor_service_dep)],
) -> PredictionResponse:
    """Execute classical machine learning inference on sensor telemetry readings."""
    clean_crop = payload.crop.strip().lower()
    try:
        schema = get_sensor_schema(clean_crop)
    except ValueError as err:
        raise InvalidInputError(
            message=f"Unknown crop '{clean_crop}'. Supported crops: lettuce, spinach, tomato, tomato_live.",
            details={"crop": clean_crop, "code": "INVALID_INPUT"},
        ) from err

    # 1. Validate top-level engineering units map if provided
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

    # 3. Call SensorService to execute prediction off threadpool with bounded timeout
    prediction = await run_blocking_inference(
        sensor_service.predict,
        crop=clean_crop,
        readings=unpacked_readings,
        provenance=payload.provenance,
    )

    return prediction


@router.post(
    "/image",
    response_model=PredictionResponse,
    summary="Predict foliar conditions and nutrient deficiencies from plant images (T6-007, T6-014, T6-015)",
    description=(
        "Accepts crop type, required input_mode ('leaf' or 'plant'), optional model_family override, "
        "and an image file (multipart field 'file' or 'image') or use_campus_frame=True. "
        "Executes non-blocking per-leaf instance segmentation and classification off the event loop."
    ),
)
async def predict_image(
    request: Request,
    crop: Annotated[str, Form(description="Target crop cultivar ('lettuce', 'spinach', 'tomato')")],
    input_mode: Annotated[
        str | None,
        Form(description="Processing mode: 'leaf' for single leaf, 'plant' for whole plant"),
    ] = None,
    file: Annotated[
        UploadFile | None,
        File(description="Uploaded plant image (multipart field name 'file' or 'image')"),
    ] = None,
    image: Annotated[
        UploadFile | None,
        File(description="Uploaded plant image (multipart field name 'image' or 'file')"),
    ] = None,
    model_family: Annotated[
        str | None,
        Form(description="Optional model family architecture override"),
    ] = None,
    use_campus_frame: Annotated[
        bool,
        Form(description="Fetch latest camera frame from campus server instead of upload"),
    ] = False,
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
    adapter: Annotated[DeviceAdapter, Depends(get_device_adapter_dep)] = None,  # type: ignore[assignment]
) -> PredictionResponse:
    """Execute per-leaf image inference for the specified crop."""
    # 1. Validate input_mode (T6-007 AC1: Missing input_mode is rejected with 422)
    if input_mode is None or not str(input_mode).strip():
        raise MissingFieldError(
            message="Field 'input_mode' is required ('leaf' or 'plant').",
            details={"field": "input_mode", "code": "MISSING_FIELD"},
        )

    clean_mode = str(input_mode).strip().lower()
    if clean_mode in ("single_leaf", "single-leaf", "leaf"):
        clean_mode = "leaf"
    elif clean_mode not in ("leaf", "plant"):
        raise InvalidInputError(
            message=f"Invalid input_mode '{input_mode}'. Must be 'leaf' or 'plant'.",
            details={
                "field": "input_mode",
                "provided": input_mode,
                "allowed_modes": ["leaf", "plant"],
                "code": "INVALID_INPUT",
            },
        )

    clean_crop = crop.strip().lower()
    if clean_crop not in SUPPORTED_IMAGE_CROPS:
        raise InvalidInputError(
            message=f"Unsupported crop '{crop}'. Registered image crops: {sorted(SUPPORTED_IMAGE_CROPS)}.",
            details={"crop": clean_crop, "supported_crops": sorted(SUPPORTED_IMAGE_CROPS)},
        )

    # 2. Validate crop and check promoted model availability (T6-007 AC3: Unavailable family is reported clearly)
    summaries = image_service.registry_service.get_model_summaries()
    crop_image_models = [s for s in summaries if s.crop == clean_crop and s.modality == "image"]
    if not crop_image_models:
        available_crops = sorted(list({s.crop for s in summaries if s.modality == "image"}))
        raise ModelUnavailableError(
            message=f"No promoted image model available for crop '{clean_crop}'. Available crops: {available_crops}.",
            details={"crop": clean_crop, "available_crops": available_crops, "code": "MODEL_UNAVAILABLE"},
        )

    if model_family is not None and str(model_family).strip():
        clean_req_family = str(model_family).strip().lower()
        matching = [
            s
            for s in crop_image_models
            if s.family.lower() == clean_req_family
            or (clean_req_family in ("efficientnet", "efficientnet_b0", "efficientnetb0") and "efficientnet" in s.family.lower())
        ]
        if not matching:
            available_families = sorted(list({s.family for s in crop_image_models}))
            raise ModelUnavailableError(
                message=(
                    f"Model family '{model_family}' is not available or not promoted for crop '{clean_crop}'. "
                    f"Available families for '{clean_crop}': {available_families}."
                ),
                details={
                    "crop": clean_crop,
                    "requested_family": model_family,
                    "available_families": available_families,
                    "code": "MODEL_UNAVAILABLE",
                },
            )

    # 3. Campus camera frame routing (T6-014) vs Multipart upload
    if use_campus_frame:
        status_resp = adapter.get_status()
        if status_resp.camera.status in ("disconnected", "stale"):
            raise DeviceUnavailableError(
                message=f"Campus camera frame is unavailable (camera status: {status_resp.camera.status}).",
                details={
                    "camera_status": status_resp.camera.status,
                    "age_seconds": status_resp.camera.age_seconds,
                },
            )
        if status_resp.camera.age_seconds is not None and status_resp.camera.age_seconds > adapter.max_age_s:
            raise DeviceUnavailableError(
                message=f"Campus camera frame is stale ({status_resp.camera.age_seconds:.1f}s > {adapter.max_age_s}s).",
                details={
                    "age_seconds": status_resp.camera.age_seconds,
                    "max_age_s": adapter.max_age_s,
                },
            )

        # Fetch frame directly from adapter (never client-supplied URL)
        raw_bytes, content_type = adapter.get_latest_camera_frame()
        validated_upload = validate_and_save_image(
            file_bytes=raw_bytes,
            original_filename="campus_camera_frame.png",
            content_type=content_type,
        )

        exec_mode = "plant"
        provenance_source = "live"
    else:
        upload_file = file if file is not None else image
        if upload_file is None:
            raise MissingFieldError(
                message="An image file is required in multipart form data (field 'file' or 'image').",
                details={"field": "file", "code": "MISSING_FIELD"},
            )
        validated_upload = await validate_upload_file(upload_file)
        exec_mode = clean_mode
        provenance_source = provenance

    # 4. Execute non-blocking inference off event loop (T6-015)
    prediction: PredictionResponse = await run_blocking_inference(
        image_service.predict,
        image=validated_upload.saved_path,
        crop=clean_crop,
        input_mode=exec_mode,  # type: ignore[arg-type]
        model_family=model_family,
        provenance=provenance_source,
        mask_threshold=mask_threshold,
        max_leaves=max_leaves,
    )

    # 5. Add training-data mismatch warning for live camera frames (T6-014)
    if use_campus_frame:
        camera_warning = (
            "Camera images may differ from training images in lighting, angle, and background."
        )
        if camera_warning not in prediction.warnings:
            prediction.warnings.append(camera_warning)

    return prediction
