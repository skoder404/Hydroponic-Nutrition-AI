"""Device status, sensor telemetry, camera frame, and manual telemetry submission (T6-012, T6-013).

Endpoints:
- GET /device/status: Aggregate operational health and telemetry statuses.
- GET /device/sensors/latest: Fetch mapped sensor readings strictly conforming to crop model schema.
- GET /device/camera/frame: Fetch the latest camera snapshot frame.
- POST /device/manual: Submit manual sensor observations when DEVICE_ADAPTER=manual.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from backend.app.core.errors import DeviceUnavailableError, InvalidInputError
from backend.app.devices.adapter import DeviceAdapter, ManualAdapter, get_device_adapter
from backend.app.schemas.device import (
    DeviceStatusResponse,
    LatestSensorsResponse,
    ManualReadingSubmission,
)
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Device"])


@router.get(
    "/device/status",
    response_model=DeviceStatusResponse,
    summary="Get hardware device and telemetry status",
)
def get_device_status(
    adapter: DeviceAdapter = Depends(get_device_adapter),
) -> DeviceStatusResponse:
    """Retrieve operational status, freshness, and per-reading validation for sensors and camera."""
    return adapter.get_status()


@router.get(
    "/device/sensors/latest",
    response_model=LatestSensorsResponse,
    summary="Get latest mapped sensor telemetry for a target crop",
)
def get_latest_sensors(
    crop: Annotated[str, Query(description="Target crop cultivar ('lettuce', 'spinach', 'tomato', 'tomato_live')")] = "lettuce",
    adapter: DeviceAdapter = Depends(get_device_adapter),
) -> LatestSensorsResponse:
    """Fetch the latest live or simulated sensor readings mapped strictly to the target crop's feature schema.

    Raises 503 DEVICE_UNAVAILABLE if device is disconnected or telemetry is stale.
    """
    clean_crop = crop.strip().lower()
    # Validate crop validity
    try:
        get_sensor_schema(clean_crop)
    except ValueError as err:
        raise InvalidInputError(
            message=f"Unknown crop '{clean_crop}'. Supported crops: lettuce, spinach, tomato, tomato_live.",
            details={"crop": clean_crop, "code": "INVALID_INPUT"},
        ) from err

    readings = adapter.get_latest_sensor_readings(clean_crop)
    now = datetime.now(timezone.utc)

    # Warnings and experimental disclosures
    warnings: list[str] = []
    is_experimental = False
    if clean_crop in ("tomato", "tomato_live"):
        is_experimental = True
        warnings.append(
            "Tomato live sensor readings are experimental: N/P/K sensors are not available on the physical campus setup."
        )

    return LatestSensorsResponse(
        crop=clean_crop,
        source=adapter.source_label,
        timestamp=now,
        readings=readings,
        is_mock=adapter.is_mock,
        experimental=is_experimental,
        warnings=warnings,
    )


@router.get(
    "/device/camera/frame",
    summary="Get latest greenhouse canopy camera frame",
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}},
            "description": "Raw image frame bytes with matching Content-Type header.",
        }
    },
)
def get_latest_camera_frame(
    adapter: DeviceAdapter = Depends(get_device_adapter),
) -> Response:
    """Stream or return the latest camera snapshot image frame."""
    frame_bytes, content_type = adapter.get_latest_camera_frame()
    return Response(content=frame_bytes, media_type=content_type)


@router.post(
    "/device/manual",
    response_model=DeviceStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit manual operator sensor observations",
)
def submit_manual_reading(
    payload: ManualReadingSubmission,
    adapter: DeviceAdapter = Depends(get_device_adapter),
) -> DeviceStatusResponse:
    """Register operator sensor measurements when running in manual adapter mode."""
    if not isinstance(adapter, ManualAdapter):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Manual reading submission is only allowed when DEVICE_ADAPTER is set to 'manual'.",
        )

    adapter.submit_readings(
        crop=payload.crop,
        readings=payload.readings,
        timestamp=payload.timestamp,
    )
    return adapter.get_status()
