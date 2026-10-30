"""Health check and metrics endpoints (T9-004)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, Query, Response
from sqlalchemy import text

from backend.app.core.device import get_device_manager
from backend.app.core.telemetry import get_fallback_tracker
from backend.app.database.session import get_engine
from backend.app.schemas.health import (
    FallbackCounters,
    GpuHealth,
    HealthDetailsResponse,
    HealthResponse,
)
from backend.app.services.model_service import get_model_registry_service

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)


def _check_db_status() -> str:
    """Check database connectivity without exposing connection details or secrets."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "connected"
    except Exception as exc:
        logger.warning(
            "Database connectivity check failed: %s",
            exc.__class__.__name__,
            extra={"db_check": "failed", "error_type": exc.__class__.__name__},
        )
        return "unavailable"


def _get_gpu_health() -> GpuHealth:
    """Inspect GPU availability and memory without exposing hardware host paths."""
    device_mgr = get_device_manager()
    is_available = device_mgr.is_cuda_available()
    dev_name = device_mgr.get_device_name() if is_available else None
    mem_info = device_mgr.get_memory_info() if is_available else None

    return GpuHealth(
        available=is_available,
        device_name=dev_name,
        memory=mem_info,
    )


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return application health, active device, subsystem statuses, and fallback counters."""
    device_mgr = get_device_manager()
    active_device = device_mgr.get_device_actually_used()

    model_service = get_model_registry_service()
    registry_status = model_service.get_health_status()

    # Active database connectivity verification
    db_status = _check_db_status()

    # GPU status inspection
    gpu_status = _get_gpu_health()

    # Telemetry fallback counters
    tracker = get_fallback_tracker()
    fallback_counters = FallbackCounters(**tracker.get_counters())

    # Overall service status is ok when registry is ok and db is connected, otherwise degraded
    overall_status = (
        "ok" if registry_status == "ok" and db_status == "connected" else "degraded"
    )

    logger.info(
        "Health check requested",
        extra={
            "device": active_device,
            "registry": registry_status,
            "db": db_status,
            "overall": overall_status,
            "gpu_available": gpu_status.available,
            "fallback_counters": fallback_counters.model_dump(),
        },
    )

    return HealthResponse(
        status=overall_status,
        device=active_device,
        db=db_status,
        registry=registry_status,
        gpu=gpu_status,
        fallback_counters=fallback_counters,
    )


@router.get("/health/details", response_model=HealthDetailsResponse)
async def get_health_details() -> HealthDetailsResponse:
    """Extended health diagnostics with model counts and device configuration."""
    device_mgr = get_device_manager()
    active_device = device_mgr.get_device_actually_used()

    model_service = get_model_registry_service()
    registry_status = model_service.get_health_status()
    models = model_service.get_model_summaries()

    db_status = _check_db_status()
    gpu_status = _get_gpu_health()

    tracker = get_fallback_tracker()
    fallback_counters = FallbackCounters(**tracker.get_counters())

    overall_status = (
        "ok" if registry_status == "ok" and db_status == "connected" else "degraded"
    )

    return HealthDetailsResponse(
        status=overall_status,
        device=active_device,
        db=db_status,
        registry=registry_status,
        gpu=gpu_status,
        fallback_counters=fallback_counters,
        active_models_count=len(models),
        target_device=device_mgr.get_target_device(),
        force_cpu=device_mgr.force_cpu,
    )


@router.get("/metrics")
async def get_metrics(
    format: str | None = Query(
        None, description="Output format: 'prometheus' or 'json'"
    ),
    accept: str | None = Header(None),
) -> Response:
    """Expose telemetry and resilience fallback counters in Prometheus or JSON format."""
    tracker = get_fallback_tracker()

    if format == "json" or (accept and "application/json" in accept):
        import json

        data: dict[str, Any] = {
            "fallback_counters": tracker.get_counters(),
            "status": "ok",
        }
        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
        )

    # Standard Prometheus text format
    content = tracker.format_prometheus_metrics()
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
