"""API v1 router assembly."""

from fastapi import APIRouter

from backend.app.api.v1.analyze import router as analyze_router
from backend.app.api.v1.device import router as device_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.history import router as history_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.predict import router as predict_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(history_router)
api_v1_router.include_router(device_router)
api_v1_router.include_router(predict_router)
api_v1_router.include_router(analyze_router)

__all__ = ["api_v1_router"]
