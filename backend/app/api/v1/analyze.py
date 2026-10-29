"""Unified multi-modal analysis endpoint (T6-011, T6-014, T6-015).

Provides:
- POST /analyze: Accepts optional sensor telemetry and/or optional plant image (or live camera frame).
  Executes available modalities independently; partial success is fully supported (if one modality
  fails, the available modality still returns cleanly).
  Returns discrete results, not_assessed[] disclosures, and agronomic recommendations.
  Invariants (Overview §19):
  - No merged or averaged confidence score.
  - No combined severity score.
  - No contradiction alert.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from backend.app.core.config import get_settings
from backend.app.core.errors import (
    AppException,
    DeviceUnavailableError,
    InvalidInputError,
)
from backend.app.core.inference import run_blocking_inference
from backend.app.core.uploads import validate_and_save_image, validate_upload_file
from backend.app.devices.adapter import DeviceAdapter, get_device_adapter
from backend.app.schemas.prediction import (
    AnalyzeResponse,
    NotAssessedItem,
    PredictionResponse,
)
from backend.app.services.image_service import (
    ImageService,
    get_image_service,
)
from backend.app.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)
from backend.app.services.sensor_service import (
    SensorService,
    get_sensor_service,
)
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])

SUPPORTED_CROPS: frozenset[str] = frozenset({"lettuce", "spinach", "tomato"})


def get_sensor_service_dep() -> SensorService:
    return get_sensor_service()


def get_image_service_dep() -> ImageService:
    return get_image_service()


def get_recommendation_service_dep() -> RecommendationService:
    settings = get_settings()
    active_prov = (
        "gemini"
        if (settings.gemini_api_key and settings.gemini_api_key.get_secret_value().strip())
        else settings.recommendation_provider
    )
    return get_recommendation_service(provider=active_prov)


def get_device_adapter_dep() -> DeviceAdapter:
    return get_device_adapter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Multi-modal crop diagnosis across sensor and image evidence (T6-011, T6-014)",
    description=(
        "Executes whichever modalities were supplied (sensor telemetry, foliar imagery, or live campus frame). "
        "Supports partial success: failure in one modality does not block the other. "
        "Discloses omitted or failed modalities in not_assessed[]; never outputs a merged confidence score."
    ),
)
async def analyze(
    request: Request,
    crop: Annotated[
        str, Form(description="Target crop cultivar ('lettuce', 'spinach', 'tomato')")
    ],
    sensor_service: Annotated[SensorService, Depends(get_sensor_service_dep)],
    image_service: Annotated[ImageService, Depends(get_image_service_dep)],
    recommendation_service: Annotated[
        RecommendationService, Depends(get_recommendation_service_dep)
    ],
    adapter: Annotated[DeviceAdapter, Depends(get_device_adapter_dep)],
    input_mode: Annotated[
        str, Form(description="Image mode: 'leaf' or 'plant'")
    ] = "plant",
    model_family: Annotated[
        str | None, Form(description="Optional model family override")
    ] = None,
    use_campus_frame: Annotated[
        bool, Form(description="Fetch latest camera frame from campus server")
    ] = False,
    sensor_readings: Annotated[
        str | None, Form(description="JSON-encoded sensor telemetry dictionary")
    ] = None,
    units: Annotated[
        str | None, Form(description="Optional JSON-encoded sensor units dictionary")
    ] = None,
    file: UploadFile | None = File(None),
) -> AnalyzeResponse:
    """Execute multi-modal analysis with partial failure tolerance."""
    clean_crop = crop.strip().lower()
    if clean_crop not in SUPPORTED_CROPS:
        raise InvalidInputError(
            message=f"Unsupported crop '{crop}'. Supported crops: {sorted(SUPPORTED_CROPS)}.",
            details={"crop": clean_crop, "code": "INVALID_INPUT"},
        )

    clean_mode = input_mode.strip().lower() if input_mode else "plant"
    if clean_mode in ("single_leaf", "single-leaf", "leaf"):
        clean_mode = "leaf"
    elif clean_mode not in ("leaf", "plant"):
        raise InvalidInputError(
            message=f"Invalid input_mode '{input_mode}'. Must be either 'leaf' (or 'single_leaf') or 'plant'.",
            details={"field": "input_mode", "provided": input_mode},
        )

    sensor_prediction: PredictionResponse | None = None
    image_prediction: PredictionResponse | None = None
    not_assessed: list[NotAssessedItem] = []
    combined_warnings: list[str] = []

    # --------------------------------------------------------------------------
    # 1. Evaluate Sensor Modality
    # --------------------------------------------------------------------------
    if sensor_readings and sensor_readings.strip():
        try:
            try:
                raw_readings = json.loads(sensor_readings)
                if not isinstance(raw_readings, dict):
                    raise ValueError("sensor_readings must be a JSON object")
            except Exception as err:
                raise InvalidInputError(
                    f"Malformed sensor_readings JSON: {err}",
                    details={"field": "sensor_readings"},
                ) from err

            # Check units if supplied
            if units and units.strip():
                try:
                    units_dict = json.loads(units)
                    schema = get_sensor_schema(clean_crop)
                    for feat, u in units_dict.items():
                        expected = schema.units.get(feat)
                        if expected and str(u).strip() != expected:
                            raise InvalidInputError(
                                f"Invalid unit '{u}' for feature '{feat}'. Expected '{expected}'.",
                                details={"feature": feat, "unit": u},
                            )
                except AppException:
                    raise
                except Exception as err:
                    raise InvalidInputError(
                        f"Malformed units JSON: {err}", details={"field": "units"}
                    ) from err

            # Execute sensor prediction off threadpool
            sensor_prediction = await run_blocking_inference(
                sensor_service.predict,
                crop=clean_crop,
                readings=raw_readings,
                provenance="manual",
            )
            combined_warnings.extend(sensor_prediction.warnings)

        except AppException as exc:
            logger.warning("Sensor evaluation failed in /analyze: %s", exc.message)
            not_assessed.append(
                NotAssessedItem(
                    modality="sensor",
                    reason=f"Sensor evaluation failed: {exc.message}",
                )
            )
        except Exception as exc:
            logger.error("Unexpected error in sensor evaluation: %s", exc)
            not_assessed.append(
                NotAssessedItem(
                    modality="sensor",
                    reason=f"Sensor evaluation encountered an unexpected error: {exc}",
                )
            )
    else:
        not_assessed.append(
            NotAssessedItem(
                modality="sensor",
                reason="No sensor telemetry provided in request; root-zone water chemistry was not assessed.",
            )
        )

    # --------------------------------------------------------------------------
    # 2. Evaluate Image Modality
    # --------------------------------------------------------------------------
    has_image_input = (file is not None and file.filename) or use_campus_frame

    if has_image_input:
        try:
            if use_campus_frame:
                status_resp = adapter.get_status()
                if status_resp.camera.status in ("disconnected", "stale"):
                    raise DeviceUnavailableError(
                        f"Campus camera frame unavailable (camera status: {status_resp.camera.status}).",
                        details={"camera_status": status_resp.camera.status},
                    )
                if (
                    status_resp.camera.age_seconds is not None
                    and status_resp.camera.age_seconds > adapter.max_age_s
                ):
                    raise DeviceUnavailableError(
                        f"Campus camera frame is stale ({status_resp.camera.age_seconds:.1f}s > {adapter.max_age_s}s)."
                    )

                raw_bytes, content_type = adapter.get_latest_camera_frame()
                validated_upload = validate_and_save_image(
                    file_bytes=raw_bytes,
                    original_filename="campus_camera_frame.png",
                    content_type=content_type,
                )
                exec_mode = "plant"
                prov = "live"
            else:
                assert file is not None
                validated_upload = await validate_upload_file(file)
                exec_mode = clean_mode
                prov = "manual"

            # Execute image prediction off threadpool
            image_prediction = await run_blocking_inference(
                image_service.predict,
                image=validated_upload.saved_path,
                crop=clean_crop,
                input_mode=exec_mode,
                model_family=model_family,
                provenance=prov,
            )

            if use_campus_frame:
                camera_warning = "Camera images may differ from training images in lighting, angle, and background."
                if camera_warning not in image_prediction.warnings:
                    image_prediction.warnings.append(camera_warning)

            combined_warnings.extend(image_prediction.warnings)

        except AppException as exc:
            logger.warning("Image evaluation failed in /analyze: %s", exc.message)
            not_assessed.append(
                NotAssessedItem(
                    modality="image",
                    reason=f"Foliar leaf image evaluation failed: {exc.message}",
                )
            )
        except Exception as exc:
            logger.error("Unexpected error in image evaluation: %s", exc)
            not_assessed.append(
                NotAssessedItem(
                    modality="image",
                    reason=f"Foliar leaf image evaluation encountered an error: {exc}",
                )
            )
    else:
        not_assessed.append(
            NotAssessedItem(
                modality="image",
                reason="No foliar plant image provided in request; foliar leaf health was not assessed.",
            )
        )

    # --------------------------------------------------------------------------
    # 3. Partial Success Verification
    # --------------------------------------------------------------------------
    # Acceptance Criterion 1 (T6-011): Image failure does not block sensor result, and vice-versa
    if sensor_prediction is None and image_prediction is None:
        reasons = "; ".join(item.reason for item in not_assessed)
        raise InvalidInputError(
            message=f"Multi-modal analysis failed: neither modality could be evaluated. {reasons}",
            details={"not_assessed": [item.model_dump() for item in not_assessed]},
        )

    # --------------------------------------------------------------------------
    # 4. Generate Agronomic Recommendations
    # --------------------------------------------------------------------------
    recommendation = None
    try:
        def _is_healthy(cls_name: str | None) -> bool:
            if not cls_name:
                return True
            c = cls_name.lower().strip()
            if "unhealthy" in c:
                return False
            return (
                c in ("healthier", "spinach_healthy", "tomato_healthy", "healthy", "normal")
                or c.endswith("_healthy")
                or "healthy foliage" in c
            )

        image_has_issue = image_prediction is not None and not _is_healthy(image_prediction.predicted_class)
        sensor_has_issue = sensor_prediction is not None and not _is_healthy(sensor_prediction.predicted_class)

        if image_has_issue and image_prediction is not None:
            recommendation = recommendation_service.recommend(
                crop=clean_crop,
                predicted_class=image_prediction.predicted_class,
                modality="image",
                confidence=image_prediction.confidence,
                uncertain=image_prediction.uncertain,
                sensor_readings=sensor_readings,
            )
        elif sensor_has_issue and sensor_prediction is not None:
            recommendation = recommendation_service.recommend_from_prediction(
                sensor_prediction,
                sensor_readings=sensor_readings,
            )
        elif image_prediction is not None:
            recommendation = recommendation_service.recommend(
                crop=clean_crop,
                predicted_class=image_prediction.predicted_class,
                modality="image",
                confidence=image_prediction.confidence,
                uncertain=image_prediction.uncertain,
                sensor_readings=sensor_readings,
            )
        elif sensor_prediction is not None:
            recommendation = recommendation_service.recommend_from_prediction(
                sensor_prediction,
                sensor_readings=sensor_readings,
            )
    except Exception as exc:
        logger.warning("Failed to generate recommendation: %s", exc)
        combined_warnings.append(
            f"Recommendation generation encountered an issue: {exc}"
        )

    # Remove duplicate warnings
    unique_warnings = list(dict.fromkeys(combined_warnings))

    return AnalyzeResponse(
        crop=clean_crop,
        sensor=sensor_prediction,
        image=image_prediction,
        not_assessed=not_assessed,
        recommendation=recommendation,
        warnings=unique_warnings,
    )
