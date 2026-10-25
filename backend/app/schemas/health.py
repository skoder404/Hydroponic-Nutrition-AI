"""Pydantic schemas for health and system status endpoints."""
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", description="Overall service status")
    device: str = Field(description="Detected compute device from DeviceManager ('cuda:0' or 'cpu')")
    db: str = Field(default="unknown", description="Database connectivity status")
    registry: str = Field(default="unknown", description="Model registry status")
