"""Pydantic schemas for health and system status endpoints (T9-004)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GpuHealth(BaseModel):
    """GPU hardware and memory availability diagnostics."""

    model_config = ConfigDict(extra="forbid")

    available: bool = Field(
        default=False, description="Whether CUDA GPU acceleration is available"
    )
    device_name: str | None = Field(
        default=None, description="GPU hardware name if detected"
    )
    memory: dict[str, Any] | None = Field(
        default=None,
        description="GPU memory statistics (allocated_mb, reserved_mb, total_mb, free_mb)",
    )


class FallbackCounters(BaseModel):
    """Operational resilience fallback counters."""

    model_config = ConfigDict(extra="forbid")

    gemini_to_rules: int = Field(
        default=0,
        description="Count of Gemini failures falling back to rule-based recommendations",
    )
    gpu_to_cpu: int = Field(
        default=0,
        description="Count of GPU OOM or runtime failures falling back to CPU",
    )
    segmentation_failure: int = Field(
        default=0,
        description="Count of leaf segmentation failure events",
    )


class HealthResponse(BaseModel):
    """Health check response schema with subsystem statuses and telemetry."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="ok", description="Overall service status ('ok', 'degraded')"
    )
    device: str = Field(
        description="Detected compute device from DeviceManager ('cuda:0' or 'cpu')"
    )
    db: str = Field(
        default="unknown",
        description="Database connectivity status ('connected', 'unavailable')",
    )
    registry: str = Field(
        default="unknown",
        description="Model registry status ('ok', 'degraded', 'unknown')",
    )
    gpu: GpuHealth = Field(
        default_factory=lambda: GpuHealth(
            available=False, device_name=None, memory=None
        ),
        description="GPU hardware and memory status",
    )
    fallback_counters: FallbackCounters = Field(
        default_factory=FallbackCounters,
        description="Counters tracking fallback occurrences across subsystems",
    )


class HealthDetailsResponse(HealthResponse):
    """Extended health diagnostics including models and runtime telemetry."""

    active_models_count: int = Field(
        default=0, description="Total active models loaded in registry"
    )
    target_device: str = Field(
        default="cpu", description="Configured target compute device"
    )
    force_cpu: bool = Field(
        default=False, description="Whether FORCE_CPU override is enabled"
    )
