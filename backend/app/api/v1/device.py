"""Device status endpoint and manual telemetry submission (T6-012).

Endpoints:
- GET /device/status: Aggregate operational health and telemetry statuses.
- POST /device/manual: Submit manual sensor observations when DEVICE_ADAPTER=manual.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.errors import InvalidInputError
from backend.app.devices.adapter import DeviceAdapter, ManualAdapter, get_device_adapter
from backend.app.schemas.device import DeviceStatusResponse, ManualReadingSubmission

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
