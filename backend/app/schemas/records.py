"""Analysis record schemas conforming to TECH_RULES (T6-001).

Implements RecordCreate (with strict extra='forbid' validation) and RecordRead
for the core analysis_records table and historical queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.recommendation import RecommendationRead


class RecordCreate(BaseModel):
    """Schema for registering a new agronomic observation record.

    Strict configuration (extra='forbid') rejects unexpected fields.
    """

    model_config = ConfigDict(extra="forbid")

    crop: str = Field(
        ...,
        description="Target crop cultivar ('lettuce', 'spinach', 'tomato')",
        examples=["lettuce"],
    )
    mode: Literal["sensor", "image", "both"] = Field(
        ...,
        description="Analysis operational mode",
        examples=["sensor"],
    )
    source: Literal["manual", "live", "imported"] = Field(
        ...,
        description="Data ingestion origin",
        examples=["live"],
    )
    observed_at: datetime = Field(
        ...,
        description="Observation timestamp in UTC",
        examples=["2026-10-27T17:30:00Z"],
    )
    observed_tz: str = Field(
        ...,
        description="Original timezone string where telemetry or imagery was captured (e.g. 'UTC', 'Asia/Kolkata')",
        examples=["Asia/Kolkata"],
    )
    notes: str | None = Field(
        None,
        description="Optional operator notes or greenhouse bay identifiers",
        examples=["Bay 3 channel A testing row"],
    )
    inputs: dict[str, Any] | None = Field(
        None,
        description="Raw or normalized input payload reference",
        examples=[{"ph": 6.2, "ec_ms_cm": 1.4, "water_temp_c": 21.5}],
    )
    predictions: list[PredictionResponse] | None = Field(
        None,
        description="Optional list of prediction responses to associate atomically",
    )
    recommendation: RecommendationRead | None = Field(
        None,
        description="Optional agronomic recommendation to associate atomically",
    )


class RecordRead(BaseModel):
    """Full analysis record returned by GET /history and GET /history/{id}."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Primary UUID identifier for the analysis record",
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
    )
    crop: str = Field(..., description="Crop cultivar", examples=["lettuce"])
    mode: Literal["sensor", "image", "both"] = Field(
        ..., description="Analysis mode", examples=["both"]
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Server timestamp when record was persisted",
        examples=["2026-10-27T18:00:00Z"],
    )
    observed_at: datetime = Field(
        ...,
        description="Timestamp when observation was captured",
        examples=["2026-10-27T17:30:00Z"],
    )
    observed_tz: str = Field(
        ...,
        description="Original timezone string of the sensor capture",
        examples=["Asia/Kolkata"],
    )
    source: Literal["manual", "live", "imported"] = Field(
        ...,
        description="Data provenance source",
        examples=["live"],
    )
    notes: str | None = Field(
        None, description="Operator notes", examples=["Routine inspection"]
    )
    inputs: dict[str, Any] | None = Field(
        None, description="Input sensor readings or image references"
    )
    predictions: list[PredictionResponse] = Field(
        default_factory=list,
        description="Associated prediction responses for this record",
    )
    recommendation: RecommendationRead | None = Field(
        None,
        description="Associated agronomic recommendation if available",
    )
    historical: bool = Field(
        default=True,
        description="Flag indicating that this record is historical (persisted observation)",
        examples=[True],
    )
