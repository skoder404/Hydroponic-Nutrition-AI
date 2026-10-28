"""API v1 router assembly."""
from fastapi import APIRouter

from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.models import router as models_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(models_router)

__all__ = ["api_v1_router"]
