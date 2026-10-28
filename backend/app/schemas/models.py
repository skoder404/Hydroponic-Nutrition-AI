"""Pydantic schemas for model discovery, status, and metadata inspection (T6-002).

Exposes model specifications per T6-002:
- crop, task, family, version, status, validation macro-F1, experimental flag.
- Detailed error reporting for MODEL_UNAVAILABLE and SCHEMA_MISMATCH.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelSummary(BaseModel):
    """Summary of a promoted or registered machine learning model."""

    model_config = ConfigDict(extra="ignore")

    crop: str = Field(
        ..., description="Crop family: lettuce, spinach, tomato, tomato_live."
    )
    task: str = Field(
        ..., description="ML task: classification, segmentation, feature_extraction."
    )
    modality: str = Field(..., description="Modality: sensor, image, feature.")
    family: str = Field(
        ...,
        description="Model family/architecture (e.g. DecisionTreeClassifier, RandomForestClassifier).",
    )
    version: str = Field(..., description="Model artifact semantic version (e.g. 1.0.0).")
    status: str = Field(
        ...,
        description="Model lifecycle status: promoted, candidate, MODEL_UNAVAILABLE, SCHEMA_MISMATCH.",
    )
    validation_macro_f1: float | None = Field(
        default=None, description="Macro-averaged F1 score achieved on validation partition."
    )
    experimental: bool = Field(
        default=False,
        description="Flag indicating whether model is experimental or production-approved.",
    )
    error_detail: str | None = Field(
        default=None,
        description="Failure rationale if status is MODEL_UNAVAILABLE or SCHEMA_MISMATCH.",
    )
    classes: list[str] | None = Field(
        default=None, description="Ordered target class taxonomy labels."
    )
    ordered_features: list[str] | None = Field(
        default=None, description="Ordered list of feature column names expected by model."
    )
    display_mapping: dict[str, str] | None = Field(
        default=None, description="Mapping from raw class label to user-facing display label."
    )
    split_protocol: str | None = Field(
        default=None, description="Protocol used for train/val/test partitioning."
    )
    metrics: dict[str, Any] | None = Field(
        default=None, description="Complete validation metrics dictionary."
    )
