"""Prediction request and response schemas conforming to TECH_RULES (T6-001).

Key invariants:
- `confidence_semantics` is strictly mandatory whenever `confidence` is present.
- Per-leaf predictions in image models are never averaged into a single confidence score.
- Input requests strictly forbid unknown fields (`extra="forbid"`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LeafPrediction(BaseModel):
    """Prediction for an isolated single leaf instance in an image."""

    leaf_id: str = Field(
        ...,
        description="Unique identifier for the leaf instance within the image (e.g. leaf_001)",
        examples=["leaf_001"],
    )
    bbox: list[int | float] = Field(
        ...,
        description="Bounding box coordinates in [xmin, ymin, xmax, ymax] pixel or normalized format",
        examples=[[42, 58, 280, 310]],
    )
    contour_area: int | None = Field(
        None,
        description="Segment area in pixels",
        examples=[14200],
    )
    mask_ref: str | None = Field(
        None,
        description="Optional local artifact or storage URI reference to the binary mask",
        examples=["artifacts/masks/leaf_001.png"],
    )
    predicted_class: str = Field(
        ...,
        description="Target biological deficiency class or healthy status",
        examples=["NitrogenDeficiency"],
    )
    display_label: str = Field(
        ...,
        description="Human-friendly label for UI presentation",
        examples=["Nitrogen Deficiency"],
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence score in [0.0, 1.0]",
        examples=[0.892],
    )
    confidence_semantics: str | None = Field(
        None,
        description="Mandatory semantic interpretation of the confidence score whenever confidence is present",
        examples=["multinomial_probability"],
    )
    uncertain: bool = Field(
        ...,
        description="True if model confidence is below the empirical crop threshold or out-of-distribution",
        examples=[False],
    )

    @model_validator(mode="after")
    def validate_confidence_semantics(self) -> LeafPrediction:
        """Enforce: confidence_semantics is mandatory if and only if confidence is present."""
        if self.confidence is not None:
            if not self.confidence_semantics or not self.confidence_semantics.strip():
                raise ValueError(
                    "`confidence_semantics` is mandatory whenever `confidence` is present."
                )
        elif self.confidence_semantics is not None:
            raise ValueError(
                "`confidence_semantics` cannot be specified when `confidence` is null."
            )
        return self


class RejectedLeaf(BaseModel):
    """Metadata for a candidate leaf mask rejected by per-leaf validation quality gates."""

    leaf_id: str = Field(
        ..., description="Candidate leaf identifier", examples=["leaf_002"]
    )
    reason: str = Field(
        ...,
        description="Reason why the mask was rejected (e.g. empty, tiny dust speck, border contact)",
        examples=[
            "Mask is too small (0.0004 < min_area_ratio 0.0100); likely dust or noise"
        ],
    )
    verdict: Literal["reject", "retry"] = Field(
        "reject",
        description="Validator verdict category",
        examples=["reject"],
    )


class PredictionSummary(BaseModel):
    """Discrete counts summary across extracted leaf instances.

    NOTE: In accordance with TECH_RULES Rule 16 and PRD, per-leaf predictions
    are NEVER averaged into a single confidence score.
    """

    model_config = ConfigDict(extra="forbid")

    leaf_count: int = Field(
        ...,
        ge=0,
        description="Total count of successfully segmented and assessed leaves",
        examples=[4],
    )
    class_counts: dict[str, int] = Field(
        ...,
        description="Histogram of counts per predicted biological class",
        examples=[{"NitrogenDeficiency": 3, "Healthy": 1}],
    )


class PredictSensorRequest(BaseModel):
    """Request payload for POST /predict/sensor."""

    model_config = ConfigDict(extra="forbid")

    crop: str = Field(
        ...,
        description="Target crop cultivar ('lettuce', 'spinach', 'tomato', 'tomato_live')",
        examples=["lettuce"],
    )
    readings: dict[str, Any] = Field(
        ...,
        description="Sensor readings dictionary (feature name -> float or {value: float, unit: str})",
        examples=[
            {
                "ph": 6.2,
                "ec": 1.6,
                "water_temp": 20.0,
                "humidity": 65.0,
                "air_temp": 22.0,
            }
        ],
    )
    units: dict[str, str] | None = Field(
        None,
        description="Optional dictionary mapping sensor features to physical engineering units",
        examples=[
            {
                "ph": "pH",
                "ec": "mS/cm",
                "water_temp": "degC",
                "humidity": "%",
                "air_temp": "degC",
            }
        ],
    )
    provenance: Literal["live", "manual", "imported"] = Field(
        "manual",
        description="Origin of telemetry (manual input, live device, or imported fixture)",
        examples=["manual"],
    )


class PredictionResponse(BaseModel):
    """Comprehensive prediction response payload for sensor and image modalities.

    Directly implements TECH_RULES API Standards (v1):
    `crop, modality, predicted_class, display_label, confidence, confidence_semantics,
    uncertain, experimental, model_id, model_version, timestamp, provenance, warnings[], limitations[]`
    """

    crop: str = Field(
        ...,
        description="Crop cultivar ('lettuce', 'spinach', 'tomato')",
        examples=["lettuce"],
    )
    modality: Literal["sensor", "image", "image_features"] = Field(
        ...,
        description="Inference modality",
        examples=["sensor"],
    )
    predicted_class: str = Field(
        ...,
        description="Model output class",
        examples=["NitrogenDeficiency"],
    )
    display_label: str = Field(
        ...,
        description="Formatted label for UI display",
        examples=["Nitrogen Deficiency"],
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model output score or confidence",
        examples=[0.875],
    )
    confidence_semantics: str | None = Field(
        None,
        description="Mandatory semantic description of confidence score when confidence is present",
        examples=["multinomial_probability"],
    )
    uncertain: bool = Field(
        ...,
        description="True if confidence is below calibrated threshold",
        examples=[False],
    )
    experimental: bool = Field(
        False,
        description="True if model is unpromoted or running in experimental trial",
        examples=[False],
    )
    model_id: str = Field(
        ...,
        description="Unique identifier of the executing model",
        examples=["lettuce_sensor_rf"],
    )
    model_version: str = Field(
        ...,
        description="Semantic version of the model artifact",
        examples=["v1.0.0"],
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when prediction was generated",
        examples=["2026-10-27T18:00:00Z"],
    )
    provenance: Literal["live", "manual", "imported"] = Field(
        ...,
        description="Source provenance of the underlying inputs",
        examples=["live"],
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Operational warnings (e.g. out-of-range sensor drift, uncalibrated lighting)",
        examples=[[]],
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known scientific and hardware limitations applicable to this prediction",
        examples=[
            "Model trained exclusively on greenhouse hydro-channels; field crops unvalidated."
        ],
    )

    # Image-specific extension fields
    input_mode: Literal["single_leaf", "plant"] | None = Field(
        default=None,
        description="Image capture mode",
        examples=["plant"],
    )
    leaves: list[LeafPrediction] = Field(
        default_factory=list,
        description="Per-leaf instance predictions",
    )
    rejected_leaves: list[RejectedLeaf] = Field(
        default_factory=list,
        description="Candidate masks rejected during quality validation",
    )
    summary: PredictionSummary | None = Field(
        default=None,
        description="Summary counts across leaves (never averaged confidence)",
    )

    @model_validator(mode="after")
    def validate_confidence_semantics(self) -> PredictionResponse:
        """Enforce: confidence_semantics is mandatory if and only if confidence is present."""
        if self.confidence is not None:
            if not self.confidence_semantics or not self.confidence_semantics.strip():
                raise ValueError(
                    "`confidence_semantics` is mandatory whenever `confidence` is present."
                )
        elif self.confidence_semantics is not None:
            raise ValueError(
                "`confidence_semantics` cannot be specified when `confidence` is null."
            )
        return self


class NotAssessedItem(BaseModel):
    """Explains why a modality was omitted or failed (T6-011, Overview §19)."""

    modality: Literal["sensor", "image"] = Field(
        ...,
        description="The modality that was omitted, skipped, or failed",
        examples=["image"],
    )
    reason: str = Field(
        ...,
        description="Clear human-readable explanation of why this modality was not evaluated",
        examples=["No foliar photograph provided; leaf health is not assessed."],
    )


class AnalyzeResponse(BaseModel):
    """Comprehensive response for multi-modal POST /analyze (T6-011).

    Invariants (Overview §19):
    - Image and sensor predictions are returned separately.
    - No merged or averaged confidence score; no combined severity score.
    - No subjective or arbitrary 'contradiction' alerts.
    - If one modality fails, the available modality still returns cleanly.
    - not_assessed[] discloses skipped or failed modalities with explicit reasons.
    """

    crop: str = Field(
        ...,
        description="Target crop cultivar ('lettuce', 'spinach', 'tomato')",
        examples=["lettuce"],
    )
    sensor: PredictionResponse | None = Field(
        default=None,
        description="Prediction result from water chemistry sensor telemetry",
    )
    image: PredictionResponse | None = Field(
        default=None,
        description="Prediction result from foliar plant imagery",
    )
    not_assessed: list[NotAssessedItem] = Field(
        default_factory=list,
        description="List of omitted or failed modalities with reasons",
    )
    recommendation: Any | None = Field(
        default=None,
        description="Agronomic action recommendations based on validated available evidence",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Operational, lighting, or hardware warnings",
    )

