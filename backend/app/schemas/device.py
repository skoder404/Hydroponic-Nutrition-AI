"""Device status schemas conforming to TECH_RULES (T6-012).

Defines typed schemas for hardware telemetry status, per-sensor reading status
(missing / disconnected / invalid / stale / ok), camera feed status, and device metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReadingStatus = Literal["missing", "disconnected", "invalid", "stale", "ok"]


class SensorReadingStatus(BaseModel):
    """Status and value of an individual sensor reading."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Sensor variable name (e.g. 'ph', 'ec')")
    value: float | None = Field(None, description="Current measured numeric value")
    unit: str = Field("", description="Engineering physical unit (e.g. 'pH', 'mS/cm', 'degC')")
    status: ReadingStatus = Field(
        ...,
        description="Freshness and validity status: missing / disconnected / invalid / stale / ok",
    )
    timestamp: datetime | None = Field(
        None, description="Observation timestamp reported by the hardware logger"
    )
    age_seconds: float | None = Field(
        None, description="Elapsed age in seconds since observation timestamp"
    )
    message: str | None = Field(
        None, description="Descriptive status narrative or reason for invalidity"
    )


class CameraDeviceStatus(BaseModel):
    """Status of the greenhouse canopy camera feed."""

    model_config = ConfigDict(extra="forbid")

    status: ReadingStatus = Field(
        ...,
        description="Camera feed status: missing / disconnected / invalid / stale / ok",
    )
    timestamp: datetime | None = Field(
        None, description="Timestamp of the latest captured image frame"
    )
    age_seconds: float | None = Field(
        None, description="Elapsed age in seconds since frame capture"
    )
    resolution: str | None = Field(
        None, description="Frame resolution (e.g. '1920x1080')", examples=["1920x1080"]
    )
    message: str | None = Field(
        None, description="Diagnostic narrative or error message"
    )


class DeviceStatusResponse(BaseModel):
    """Aggregate hardware status response for GET /api/v1/device/status."""

    model_config = ConfigDict(extra="forbid")

    adapter: Literal["mock", "campus", "manual"] = Field(
        ...,
        description="Active device adapter backend identifier",
    )
    source: Literal["mock", "manual", "live"] = Field(
        ...,
        description="Data provenance label (Mock data is strictly labelled 'mock', never 'live')",
    )
    status: ReadingStatus = Field(
        ...,
        description="Aggregate operational health across sensors and camera",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the status assessment",
    )
    max_age_s: int = Field(
        ...,
        description="Configured maximum freshness age threshold in seconds (CAMPUS_MAX_AGE_S)",
    )
    readings: dict[str, SensorReadingStatus] = Field(
        default_factory=dict,
        description="Per-feature sensor reading statuses",
    )
    camera: CameraDeviceStatus = Field(
        ...,
        description="Latest camera feed status",
    )
    is_mock: bool = Field(
        ...,
        description="True if telemetry or imagery originates from mock fixtures",
    )


class ManualReadingSubmission(BaseModel):
    """Payload for submitting operator manual sensor measurements."""

    model_config = ConfigDict(extra="forbid")

    crop: str = Field(
        "lettuce",
        description="Target crop cultivar ('lettuce', 'spinach', 'tomato')",
        examples=["lettuce"],
    )
    readings: dict[str, float] = Field(
        ...,
        description="Map of sensor names to physical float values",
        examples=[{"ph": 6.2, "ec": 1.8, "water_temp": 20.5, "humidity": 65.0, "air_temp": 22.0}],
    )
    timestamp: datetime | None = Field(
        None,
        description="Optional observation timestamp (defaults to current UTC if omitted)",
    )
