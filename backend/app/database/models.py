"""SQLAlchemy models for application historical records (T2-021).

Tables implemented per TECH_RULES and PROJECT_OVERVIEW §21.1:
1. analysis_records: Parent transaction record for an inference / analysis run.
   - UUID primary key
   - observed_at (timestamptz UTC) + observed_tz (original timezone string)
   - source constrained to: ('manual', 'live', 'imported')
2. predictions: Output prediction per modality / leaf.
   - UUID primary key, foreign key to analysis_records
   - modality constrained to: ('image', 'sensor', 'image_features')
   - warnings stored as JSONB / JSON
3. inputs: Recorded input data or references.
   - UUID primary key, foreign key to analysis_records
   - kind constrained to: ('sensor', 'image')
   - payload / units stored as JSONB / JSON
4. recommendations: Recommended actions and evidence.
   - UUID primary key, foreign key to analysis_records
   - source constrained to: ('gemini', 'rules')
   - content / evidence stored as JSONB / JSON
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.session import Base

# Universal JSON type using PostgreSQL JSONB when available, falling back to JSON
JSONType = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class AnalysisRecord(Base):
    """Parent transaction table for analysis runs (TECH_RULES §Database Architecture)."""

    __tablename__ = "analysis_records"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    crop: Mapped[str] = mapped_column(String(50), nullable=False)
    mode: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    observed_tz: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships (cascade delete dependent records on parent purge)
    predictions: Mapped[list[Prediction]] = relationship(
        "Prediction",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    inputs: Mapped[list[Input]] = relationship(
        "Input",
        back_populates="record",
        cascade="all, delete-orphan",
    )
    recommendations: Mapped[list[Recommendation]] = relationship(
        "Recommendation",
        back_populates="record",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "source IN ('manual', 'live', 'imported')",
            name="check_analysis_records_source",
        ),
        Index("ix_analysis_records_created_at_desc", created_at.desc()),
        Index("ix_analysis_records_crop_created_at", crop, created_at),
    )


class Prediction(Base):
    """Prediction results per modality or leaf (TECH_RULES §Database Architecture)."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    record_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    modality: Mapped[str] = mapped_column(String(30), nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(100), nullable=False)
    display_label: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_semantics: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="score",
    )
    uncertain: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    experimental: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(50), nullable=False)
    warnings: Mapped[list[Any]] = mapped_column(
        JSONType,
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    # Relationships
    record: Mapped[AnalysisRecord] = relationship(
        "AnalysisRecord",
        back_populates="predictions",
    )

    __table_args__ = (
        CheckConstraint(
            "modality IN ('image', 'sensor', 'image_features')",
            name="check_predictions_modality",
        ),
    )


class Input(Base):
    """Input payload or file references for an analysis run."""

    __tablename__ = "inputs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    record_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    file_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    units: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    # Relationships
    record: Mapped[AnalysisRecord] = relationship(
        "AnalysisRecord",
        back_populates="inputs",
    )

    __table_args__ = (
        CheckConstraint(
            "kind IN ('sensor', 'image')",
            name="check_inputs_kind",
        ),
    )


class Recommendation(Base):
    """Recommended nutrient actions and evidence (TECH_RULES §Database Architecture)."""

    __tablename__ = "recommendations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    record_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    rules_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    # Relationships
    record: Mapped[AnalysisRecord] = relationship(
        "AnalysisRecord",
        back_populates="recommendations",
    )

    __table_args__ = (
        CheckConstraint(
            "source IN ('gemini', 'rules')",
            name="check_recommendations_source",
        ),
    )
