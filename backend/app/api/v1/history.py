"""History API router conforming to TECH_RULES (T6-010).

Endpoints:
- POST /history: Create historical observation record with child predictions & recommendations.
- GET /history: Paginated list of historical records with filtering and X-Total-Count header.
- GET /history/{id}: Fetch single historical record by UUID.
"""

from __future__ import annotations

import csv
import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.errors import NotFoundError, PersistenceFailedError
from backend.app.database.models import AnalysisRecord
from backend.app.database.repositories import (
    PersistenceError,
    create_record,
    get_record,
    list_records,
)
from backend.app.database.session import get_db
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.recommendation import RecommendationRead
from backend.app.schemas.records import RecordCreate, RecordRead

logger = logging.getLogger(__name__)

router = APIRouter(tags=["History"])


def record_to_schema(record: AnalysisRecord) -> RecordRead:
    """Transform an AnalysisRecord ORM entity into a validated RecordRead response."""
    # Map predictions
    predictions_out: list[PredictionResponse] = []
    if record.predictions:
        for p in record.predictions:
            predictions_out.append(
                PredictionResponse(
                    crop=record.crop,
                    modality=p.modality,  # type: ignore[arg-type]
                    predicted_class=p.predicted_class,
                    display_label=p.display_label,
                    confidence=p.confidence,
                    confidence_semantics=p.confidence_semantics,
                    uncertain=p.uncertain,
                    experimental=p.experimental,
                    model_id=p.model_id,
                    model_version=p.model_version,
                    timestamp=p.created_at or record.created_at,
                    provenance=record.source,  # type: ignore[arg-type]
                    warnings=p.warnings or [],
                )
            )

    # Map recommendation
    rec_out: RecommendationRead | None = None
    if record.recommendations:
        first_rec = record.recommendations[0]
        rec_out = RecommendationRead(
            id=first_rec.id,
            record_id=first_rec.record_id,
            source=first_rec.source,  # type: ignore[arg-type]
            rules_version=first_rec.rules_version or "v1.0.0",
            content=first_rec.content,
            evidence=first_rec.evidence,
            created_at=first_rec.created_at,
        )

    # Extract inputs payload
    inputs_payload: dict[str, Any] | None = None
    if record.inputs:
        inputs_payload = record.inputs[0].payload

    # Reconstruct original observed_at in UTC
    dt = record.observed_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return RecordRead(
        id=record.id,
        crop=record.crop,
        mode=record.mode,  # type: ignore[arg-type]
        created_at=record.created_at,
        observed_at=dt,
        observed_tz=record.observed_tz,
        source=record.source,  # type: ignore[arg-type]
        notes=record.notes,
        inputs=inputs_payload,
        predictions=predictions_out,
        recommendation=rec_out,
        historical=True,
    )


@router.post(
    "/history",
    response_model=RecordRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create historical analysis record",
)
def create_history_record(
    payload: RecordCreate,
    db: Session = Depends(get_db),
) -> RecordRead:
    """Persist a new agronomic observation record with optional predictions and recommendations."""
    pred_dicts: list[dict[str, Any]] = []
    if payload.predictions:
        for p in payload.predictions:
            pred_dicts.append({
                "modality": p.modality,
                "predicted_class": p.predicted_class,
                "display_label": p.display_label,
                "confidence": p.confidence,
                "confidence_semantics": p.confidence_semantics,
                "uncertain": p.uncertain,
                "experimental": p.experimental,
                "model_id": p.model_id,
                "model_version": p.model_version,
                "preprocessing_version": "v1",
                "warnings": p.warnings,
            })

    inp_dicts: list[dict[str, Any]] = []
    if payload.inputs:
        inp_dicts.append({
            "kind": "sensor" if payload.mode == "sensor" else "image",
            "payload": payload.inputs,
        })

    rec_dicts: list[dict[str, Any]] = []
    if payload.recommendation:
        if isinstance(payload.recommendation, dict):
            rec_dicts.append({
                "source": payload.recommendation.get("source", "rules"),
                "rules_version": payload.recommendation.get("rules_version", "v1.0.0"),
                "content": payload.recommendation.get("content", {}),
                "evidence": payload.recommendation.get("evidence", {}),
            })
        else:
            rec_dicts.append({
                "source": payload.recommendation.source,
                "rules_version": payload.recommendation.rules_version,
                "content": payload.recommendation.content,
                "evidence": payload.recommendation.evidence,
            })

    try:
        save_result = create_record(
            db,
            crop=payload.crop,
            mode=payload.mode,
            observed_at=payload.observed_at,
            observed_tz=payload.observed_tz,
            source=payload.source,
            notes=payload.notes,
            predictions=pred_dicts or None,
            inputs=inp_dicts or None,
            recommendations=rec_dicts or None,
            raise_on_error=False,
        )

        if not save_result.saved or save_result.record is None:
            err_msg = (
                str(save_result.error)
                if save_result.error
                else "Database persistence failed. Result was not saved."
            )
            if "not saved" not in err_msg.lower():
                err_msg = f"{err_msg} Result was not saved."
            raise PersistenceFailedError(err_msg)

        return record_to_schema(save_result.record)

    except PersistenceFailedError:
        raise
    except (PersistenceError, SQLAlchemyError, Exception) as exc:
        logger.error(f"Persistence error creating history record: {exc}")
        raise PersistenceFailedError(
            f"Database persistence failed: {exc}. Result was not saved."
        ) from exc


@router.get(
    "/history",
    response_model=list[RecordRead],
    summary="List historical analysis records",
)
def get_history_records(
    response: Response,
    limit: int = Query(20, description="Max records to return (capped at 100)"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    crop: str | None = Query(None, description="Filter by crop cultivar"),
    modality: str | None = Query(None, description="Filter by modality/mode"),
    mode: str | None = Query(None, description="Alias filter for modality/mode"),
    source: str | None = Query(None, description="Filter by source provenance"),
    date: str | None = Query(None, description="Filter by date string (YYYY-MM-DD)"),
    start_date: str | None = Query(None, description="Filter start date ISO"),
    end_date: str | None = Query(None, description="Filter end date ISO"),
    db: Session = Depends(get_db),
) -> list[RecordRead]:
    """Retrieve paginated historical records with optional filtering."""
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    filter_mode = modality or mode

    start_dt: datetime | None = None
    end_dt: datetime | None = None

    if date:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_dt = parsed_date
            end_dt = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        except ValueError:
            pass

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    try:
        records, total_count = list_records(
            db,
            limit=safe_limit,
            offset=safe_offset,
            crop=crop,
            source=source,
            mode=filter_mode,
            start_time=start_dt,
            end_time=end_dt,
            eager_load=True,
        )
    except SQLAlchemyError as exc:
        logger.error(f"Failed to query history records: {exc}")
        raise PersistenceFailedError(
            f"Database query failed: {exc}. Records could not be retrieved; result was not saved."
        ) from exc

    response.headers["X-Total-Count"] = str(total_count)
    return [record_to_schema(rec) for rec in records]


def sanitize_csv_cell(value: Any) -> str:
    """Neutralise CSV injection formula trigger characters (=, +, -, @, \\t, \\r)."""
    if value is None:
        return ""
    val_str = str(value)
    if val_str and val_str[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{val_str}"
    return val_str


@router.get(
    "/history/export",
    summary="Export historical analysis records to CSV",
)
def export_history_records(
    crop: str | None = Query(None, description="Filter by crop cultivar"),
    modality: str | None = Query(None, description="Filter by modality/mode"),
    mode: str | None = Query(None, description="Alias filter for modality/mode"),
    source: str | None = Query(None, description="Filter by source provenance"),
    date: str | None = Query(None, description="Filter by date string (YYYY-MM-DD)"),
    start_date: str | None = Query(None, description="Filter start date ISO"),
    end_date: str | None = Query(None, description="Filter end date ISO"),
    db: Session = Depends(get_db),
) -> Response:
    """Export filtered history records as a secure CSV file with timezone-aware timestamps."""
    filter_mode = modality or mode

    start_dt: datetime | None = None
    end_dt: datetime | None = None

    if date:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_dt = parsed_date
            end_dt = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        except ValueError:
            pass

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    try:
        records, _ = list_records(
            db,
            limit=1000,
            offset=0,
            crop=crop,
            source=source,
            mode=filter_mode,
            start_time=start_dt,
            end_time=end_dt,
            eager_load=True,
        )
    except SQLAlchemyError as exc:
        logger.error(f"Failed to query history records for export: {exc}")
        raise PersistenceFailedError(
            f"Database query failed: {exc}. Export could not be generated."
        ) from exc

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    headers = [
        "id",
        "observed_at",
        "observed_tz",
        "crop",
        "mode",
        "source",
        "result",
        "predicted_class",
        "confidence",
        "uncertain",
        "model_id",
        "model_version",
        "notes",
    ]
    writer.writerow(headers)

    for rec in records:
        schema = record_to_schema(rec)
        main_pred = schema.predictions[0] if schema.predictions else None

        dt = schema.observed_at
        tz_name = schema.observed_tz or "UTC"
        ts_with_tz = f"{dt.isoformat()} ({tz_name})"

        row = [
            sanitize_csv_cell(schema.id),
            sanitize_csv_cell(ts_with_tz),
            sanitize_csv_cell(tz_name),
            sanitize_csv_cell(schema.crop),
            sanitize_csv_cell(schema.mode),
            sanitize_csv_cell(schema.source),
            sanitize_csv_cell(main_pred.display_label if main_pred else "Complete"),
            sanitize_csv_cell(main_pred.predicted_class if main_pred else "N/A"),
            sanitize_csv_cell(f"{main_pred.confidence:.4f}" if main_pred and main_pred.confidence is not None else ""),
            sanitize_csv_cell(str(main_pred.uncertain) if main_pred else "False"),
            sanitize_csv_cell(main_pred.model_id if main_pred else "N/A"),
            sanitize_csv_cell(main_pred.model_version if main_pred else "v1.0.0"),
            sanitize_csv_cell(schema.notes or ""),
        ]
        writer.writerow(row)

    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="history_export.csv"',
        },
    )


@router.get(
    "/history/{record_id}",
    response_model=RecordRead,
    summary="Get historical record by ID",
)
def get_history_record_by_id(
    record_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> RecordRead:
    """Retrieve a single historical analysis record by UUID."""
    try:
        record = get_record(db, record_id, eager_load=True)
    except SQLAlchemyError as exc:
        logger.error(f"Database error reading record '{record_id}': {exc}")
        raise PersistenceFailedError(
            f"Database query failed: {exc}. Result was not saved."
        ) from exc

    if record is None:
        raise NotFoundError(f"Analysis record '{record_id}' not found.")

    return record_to_schema(record)
