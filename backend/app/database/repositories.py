"""Repository implementations and session management for historical records (T2-022).

Conforms to TECH_RULES and PROJECT_OVERVIEW §21.1:
- create_record: Persists an analysis run with child predictions, inputs, recommendations.
  - Distinguishes 'saved' from 'not saved' with an explicit SaveResult.
  - Surfaces database failures as typed PersistenceError (code: 'PERSISTENCE_FAILED').
  - Preserves original timestamp and timezone on round-trip.
- get_record: Retrieves an AnalysisRecord by UUID with eager loaded children.
- list_records: Retrieves paginated AnalysisRecords with filtering and created_at DESC order.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Any, Generic, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from backend.app.database.models import (
    AnalysisRecord,
    Input,
    Prediction,
    Recommendation,
)

T = TypeVar("T")


class RepositoryError(Exception):
    """Base exception for all database repository operations."""


class PersistenceError(RepositoryError):
    """Raised when persistence of an analysis record fails (TECH_RULES error code: PERSISTENCE_FAILED)."""

    def __init__(
        self,
        message: str,
        *,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = "PERSISTENCE_FAILED"
        self.original_error = original_error
        self.details = details or {}


@dataclass(frozen=True)
class SaveResult(Generic[T]):
    """Explicit result of a database persistence operation distinguishing saved from not saved.

    The API and calling services can inspect .saved and .error to ensure a failed save
    is never mistakenly acknowledged as successful.
    """

    saved: bool
    record: T | None = None
    error: PersistenceError | None = None

    def unwrap(self) -> T:
        """Return the saved record or raise the underlying PersistenceError."""
        if not self.saved or self.record is None:
            if self.error is not None:
                raise self.error
            raise PersistenceError("Record was not saved.")
        return self.record


def normalize_datetime_to_utc(dt: datetime, tz_name: str) -> datetime:
    """Ensure datetime is timezone-aware and converted to UTC for database storage."""
    if dt.tzinfo is None:
        tz: tzinfo
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            tz = timezone.utc
        dt = dt.replace(tzinfo=tz)

    return dt.astimezone(timezone.utc)


def reconstruct_observed_datetime(record: AnalysisRecord) -> datetime:
    """Reconstruct the original observed_at datetime with its original timezone."""
    dt = record.observed_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    tz_str = record.observed_tz
    try:
        return dt.astimezone(ZoneInfo(tz_str))
    except (ZoneInfoNotFoundError, ValueError):
        # Fall back to UTC if timezone name cannot be resolved
        return dt


get_original_observed_datetime = reconstruct_observed_datetime


def create_record(
    db: Session,
    *,
    crop: str,
    mode: str,
    observed_at: datetime,
    observed_tz: str,
    source: str,
    notes: str | None = None,
    predictions: Sequence[dict[str, Any] | Prediction] | None = None,
    inputs: Sequence[dict[str, Any] | Input] | None = None,
    recommendations: Sequence[dict[str, Any] | Recommendation] | None = None,
    record_id: uuid.UUID | None = None,
    raise_on_error: bool = True,
) -> SaveResult[AnalysisRecord]:
    """Persist an AnalysisRecord with child predictions, inputs, and recommendations.

    Args:
        db: SQLAlchemy database session.
        crop: Target crop identifier (e.g. 'lettuce', 'spinach', 'tomato').
        mode: Analysis mode ('single', 'batch', 'fusion', 'realtime').
        observed_at: Timestamp when measurement/photograph was taken.
        observed_tz: Original timezone string (e.g. 'UTC', 'Asia/Kolkata').
        source: Ingestion source constrained to ('manual', 'live', 'imported').
        notes: Optional narrative notes.
        predictions: Child predictions to insert atomically.
        inputs: Child input payload/references to insert atomically.
        recommendations: Child recommendations to insert atomically.
        record_id: Optional UUID (generates a new UUID if omitted).
        raise_on_error: If True, raises PersistenceError on failure; if False, returns SaveResult(saved=False).

    Returns:
        SaveResult[AnalysisRecord] indicating saved status, record, or error details.
    """
    rec_id = record_id or uuid.uuid4()
    utc_observed_at = normalize_datetime_to_utc(observed_at, observed_tz)

    record = AnalysisRecord(
        id=rec_id,
        crop=crop,
        mode=mode,
        observed_at=utc_observed_at,
        observed_tz=observed_tz,
        source=source,
        notes=notes,
    )

    # Attach predictions
    if predictions:
        for p in predictions:
            if isinstance(p, Prediction):
                pred_obj = p
                pred_obj.record_id = rec_id
            else:
                pred_obj = Prediction(
                    id=p.get("id") or uuid.uuid4(),
                    record_id=rec_id,
                    modality=p["modality"],
                    predicted_class=p["predicted_class"],
                    display_label=p["display_label"],
                    confidence=p.get("confidence"),
                    confidence_semantics=p.get("confidence_semantics", "score"),
                    uncertain=p.get("uncertain", False),
                    experimental=p.get("experimental", False),
                    model_id=p["model_id"],
                    model_version=p["model_version"],
                    preprocessing_version=p["preprocessing_version"],
                    warnings=p.get("warnings", []),
                )
            record.predictions.append(pred_obj)

    # Attach inputs
    if inputs:
        for inp in inputs:
            if isinstance(inp, Input):
                inp_obj = inp
                inp_obj.record_id = rec_id
            else:
                inp_obj = Input(
                    id=inp.get("id") or uuid.uuid4(),
                    record_id=rec_id,
                    kind=inp["kind"],
                    payload=inp.get("payload"),
                    file_ref=inp.get("file_ref"),
                    units=inp.get("units"),
                )
            record.inputs.append(inp_obj)

    # Attach recommendations
    if recommendations:
        for rec_item in recommendations:
            if isinstance(rec_item, Recommendation):
                rec_obj = rec_item
                rec_obj.record_id = rec_id
            else:
                rec_obj = Recommendation(
                    id=rec_item.get("id") or uuid.uuid4(),
                    record_id=rec_id,
                    source=rec_item["source"],
                    rules_version=rec_item.get("rules_version"),
                    content=rec_item["content"],
                    evidence=rec_item["evidence"],
                )
            record.recommendations.append(rec_obj)

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return SaveResult(saved=True, record=record, error=None)

    except SQLAlchemyError as exc:
        db.rollback()
        err = PersistenceError(
            f"Failed to persist analysis record '{rec_id}': {exc}",
            original_error=exc,
            details={"record_id": str(rec_id), "crop": crop, "source": source},
        )
        if raise_on_error:
            raise err from exc
        return SaveResult(saved=False, record=None, error=err)


def get_record(
    db: Session,
    record_id: uuid.UUID,
    *,
    eager_load: bool = True,
) -> AnalysisRecord | None:
    """Fetch an AnalysisRecord by its UUID.

    Args:
        db: SQLAlchemy session.
        record_id: Primary key UUID of the record.
        eager_load: If True, eagerly loads child predictions, inputs, and recommendations.

    Returns:
        AnalysisRecord if found, else None.
    """
    if eager_load:
        stmt = (
            sa.select(AnalysisRecord)
            .options(
                selectinload(AnalysisRecord.predictions),
                selectinload(AnalysisRecord.inputs),
                selectinload(AnalysisRecord.recommendations),
            )
            .where(AnalysisRecord.id == record_id)
        )
        return db.execute(stmt).scalar_one_or_none()

    return db.get(AnalysisRecord, record_id)


def list_records(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    crop: str | None = None,
    source: str | None = None,
    mode: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    eager_load: bool = False,
) -> tuple[list[AnalysisRecord], int]:
    """Retrieve paginated AnalysisRecords ordered by created_at DESC with optional filters.

    Args:
        db: SQLAlchemy session.
        limit: Max records to return (default 20, max 100 per TECH_RULES).
        offset: Number of records to skip.
        crop: Optional filter by crop name.
        source: Optional filter by source ('manual', 'live', 'imported').
        mode: Optional filter by analysis mode.
        start_time: Optional lower bound on observed_at.
        end_time: Optional upper bound on observed_at.
        eager_load: If True, eagerly loads child relationships.

    Returns:
        tuple (records: list[AnalysisRecord], total_count: int)
    """
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)

    # Base query for filtering
    conditions = []
    if crop:
        conditions.append(AnalysisRecord.crop == crop)
    if source:
        conditions.append(AnalysisRecord.source == source)
    if mode:
        conditions.append(AnalysisRecord.mode == mode)
    if start_time:
        conditions.append(AnalysisRecord.observed_at >= start_time)
    if end_time:
        conditions.append(AnalysisRecord.observed_at <= end_time)

    # Total count query
    count_stmt = sa.select(sa.func.count(AnalysisRecord.id))
    if conditions:
        count_stmt = count_stmt.where(*conditions)
    total_count = db.execute(count_stmt).scalar() or 0

    # Data query
    data_stmt = sa.select(AnalysisRecord)
    if conditions:
        data_stmt = data_stmt.where(*conditions)

    data_stmt = data_stmt.order_by(AnalysisRecord.created_at.desc())
    data_stmt = data_stmt.limit(safe_limit).offset(safe_offset)

    if eager_load:
        data_stmt = data_stmt.options(
            selectinload(AnalysisRecord.predictions),
            selectinload(AnalysisRecord.inputs),
            selectinload(AnalysisRecord.recommendations),
        )

    records = list(db.execute(data_stmt).scalars().all())
    return records, total_count
