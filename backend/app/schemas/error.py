"""Single unified error envelope schema conforming to TECH_RULES (T6-001, T3-004)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error object embedded within the unified error envelope."""

    code: str = Field(
        ...,
        description="Standard machine-readable error code (e.g. SEGMENTATION_FAILED, INVALID_INPUT, MODEL_UNAVAILABLE)",
        examples=["SEGMENTATION_FAILED"],
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of why the operation failed",
        examples=["No valid leaf mask could be isolated from the supplied image."],
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured contextual error diagnostics",
        examples=[{"rejected_leaves": [{"leaf_id": "leaf_0", "reason": "empty mask"}]}],
    )
    retryable: bool = Field(
        False,
        description="Flag indicating whether a client may retry the operation with adjusted input/prompting",
        examples=[True],
    )


class ErrorResponse(BaseModel):
    """Single standard error envelope returned by all HTTP failure responses."""

    error: ErrorDetail
