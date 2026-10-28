"""Model discovery and registry endpoints router (T6-002).

Exposes GET /api/v1/models returning:
crop, task, family, version, status, validation macro-F1, experimental flag.
Missing or corrupt artifacts yield MODEL_UNAVAILABLE for the affected crop only.
Incompatible feature lists yield SCHEMA_MISMATCH for the affected crop only.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

from backend.app.schemas.models import ModelSummary
from backend.app.services.model_service import get_model_registry_service

router = APIRouter(tags=["Models"])
logger = logging.getLogger(__name__)


@router.get("/models", response_model=list[ModelSummary])
async def list_models(
    crop: Optional[str] = Query(default=None, description="Optional crop filter"),
    modality: Optional[str] = Query(default=None, description="Optional modality filter"),
) -> list[ModelSummary]:
    """Expose promoted and active machine learning models with validation macro-F1 and status.

    Missing or corrupt artifacts yield MODEL_UNAVAILABLE for the affected crop only;
    no silent substitution. Incompatible feature list is reported as SCHEMA_MISMATCH.
    """
    service = get_model_registry_service()
    summaries = service.get_model_summaries(crop=crop, modality=modality)

    logger.info(
        "Models list retrieved",
        extra={
            "models_count": len(summaries),
            "filter_crop": crop,
            "filter_modality": modality,
        },
    )
    return summaries
