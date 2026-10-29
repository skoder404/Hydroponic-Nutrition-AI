"""Health check endpoint router."""

import logging

from fastapi import APIRouter

from sqlalchemy import text

from backend.app.core.device import get_device_manager
from backend.app.database.session import get_engine
from backend.app.schemas.health import HealthResponse
from backend.app.services.model_service import get_model_registry_service

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return application health, active device, and subsystem status."""
    device_mgr = get_device_manager()
    active_device = device_mgr.get_device_actually_used()

    model_service = get_model_registry_service()
    registry_status = model_service.get_health_status()

    # Active database connectivity verification
    db_status = "connected"
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        db_status = "unavailable"

    # Overall service status is ok when registry is ok and db is connected, otherwise degraded
    overall_status = "ok" if registry_status == "ok" and db_status == "connected" else "degraded"

    logger.info(
        "Health check requested",
        extra={"device": active_device, "registry": registry_status, "db": db_status, "overall": overall_status},
    )

    return HealthResponse(
        status=overall_status,
        device=active_device,
        db=db_status,
        registry=registry_status,
    )
