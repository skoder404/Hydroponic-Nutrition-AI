"""Health check endpoint router."""

import logging

from fastapi import APIRouter

from backend.app.core.device import get_device_manager
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return application health, active device, and subsystem status."""
    device_mgr = get_device_manager()
    active_device = device_mgr.get_device_actually_used()

    logger.info("Health check requested", extra={"device": active_device})

    return HealthResponse(
        status="ok",
        device=active_device,
        db="unknown",
        registry="unknown",
    )
