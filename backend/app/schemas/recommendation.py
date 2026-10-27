"""Recommendation schemas conforming to TECH_RULES (T6-001).

Implements RecommendationRead for persisted agronomic recommendations
generated via GeminiProvider or rule-based fallback (RuleProvider).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RecommendationRead(BaseModel):
    """Agronomic recommendation record generated from predictions and sensor telemetry."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique UUID for the recommendation entry",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    record_id: UUID = Field(
        ...,
        description="Foreign key UUID linking to the parent analysis_records entry",
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
    )
    source: Literal["gemini", "rules"] = Field(
        ...,
        description="Provider that authored the recommendation ('gemini' LLM or deterministic 'rules')",
        examples=["rules"],
    )
    rules_version: str = Field(
        ...,
        description="Semantic version of the agronomic decision matrix/ruleset applied",
        examples=["v1.2.0"],
    )
    content: dict[str, Any] = Field(
        ...,
        description="Structured recommendation text and corrective dosing actions (e.g. pH down, Ca(NO3)2 adjustment)",
        examples=[
            {
                "summary": "Nitrogen deficiency detected; increase calcium nitrate dosage by 15%.",
                "actions": [
                    "Adjust reservoir pH to 5.8",
                    "Add 50mL Part A nutrient stock",
                ],
            }
        ],
    )
    evidence: dict[str, Any] = Field(
        ...,
        description="Scientific evidence and sensor observations justifying the recommendation",
        examples=[
            {
                "observed_ec_ms_cm": 1.1,
                "target_ec_ms_cm": 1.6,
                "visual_symptoms": ["Lower canopy chlorosis"],
            }
        ],
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the recommendation was formulated",
        examples=["2026-10-27T18:00:00Z"],
    )
