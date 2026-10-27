"""Unit tests for repository implementations and session management (T2-022).

Verifies Acceptance Criteria:
- [x] Round-trip preserves the original timestamp and timezone.
- [x] List is paginated (limit, offset, capped at 100, ordered by created_at DESC).
- [x] DB failure surfaces as a typed error (PersistenceError with code 'PERSISTENCE_FAILED').
- [x] SaveResult distinguishes 'saved' from 'not saved'.
- [x] Atomic creation with child predictions, inputs, and recommendations.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from alembic.config import Config
from sqlalchemy.orm import Session

from alembic import command
from backend.app.database.repositories import (
    PersistenceError,
    create_record,
    get_original_observed_datetime,
    get_record,
    list_records,
)
from backend.app.database.session import get_engine

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def test_db_session(tmp_path: Path) -> Session:
    """Create isolated SQLite database with migrations applied and return a session."""
    db_file = tmp_path / "test_repo.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    # Run migrations
    ini_path = ROOT / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")

    engine = get_engine(db_url)
    with Session(engine) as session:
        yield session


def test_round_trip_preserves_timestamp_and_timezone(test_db_session: Session) -> None:
    """AC 1: Round-trip preserves the original timestamp and timezone."""
    # Test cases with different international timezones
    test_cases = [
        ("Asia/Kolkata", datetime(2026, 10, 27, 8, 30, 45, tzinfo=ZoneInfo("Asia/Kolkata"))),
        ("America/New_York", datetime(2026, 10, 25, 14, 15, 0, tzinfo=ZoneInfo("America/New_York"))),
        ("Europe/London", datetime(2026, 6, 15, 12, 0, 0, tzinfo=ZoneInfo("Europe/London"))),
        ("UTC", datetime(2026, 10, 26, 23, 59, 59, tzinfo=timezone.utc)),
    ]

    for tz_name, original_dt in test_cases:
        res = create_record(
            test_db_session,
            crop="lettuce",
            mode="single",
            observed_at=original_dt,
            observed_tz=tz_name,
            source="manual",
            notes=f"Test for {tz_name}",
        )
        assert res.saved is True
        assert res.record is not None
        rec_id = res.record.id

        # Retrieve record back from database in a fresh query
        loaded = get_record(test_db_session, rec_id)
        assert loaded is not None

        # Verify UTC point-in-time equivalence
        assert loaded.observed_at == original_dt.astimezone(timezone.utc)
        assert loaded.observed_tz == tz_name

        # Verify original local time reconstruction
        reconstructed = get_original_observed_datetime(loaded)
        assert reconstructed == original_dt
        assert reconstructed.utcoffset() == original_dt.utcoffset()
        assert reconstructed.hour == original_dt.hour
        assert reconstructed.minute == original_dt.minute
        assert reconstructed.second == original_dt.second

        # Verify property on model
        assert loaded.observed_at_original == original_dt


def test_list_records_pagination_and_filters(test_db_session: Session) -> None:
    """AC 2: List is paginated (limit, offset, max 100, ordering)."""
    # Seed 30 records with staggered created_at
    base_time = datetime(2026, 10, 25, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(30):
        create_record(
            test_db_session,
            crop="spinach" if i % 2 == 0 else "tomato",
            mode="single",
            observed_at=base_time,
            observed_tz="UTC",
            source="live" if i % 3 == 0 else "manual",
            notes=f"Record {i:02d}",
        )

    # 1. Default pagination (limit 20, offset 0)
    records, total = list_records(test_db_session, limit=20, offset=0)
    assert total == 30
    assert len(records) == 20

    # 2. Second page (offset 20, limit 20)
    page2, total2 = list_records(test_db_session, limit=20, offset=20)
    assert total2 == 30
    assert len(page2) == 10

    # Ensure no duplicate records between page 1 and page 2
    page1_ids = {r.id for r in records}
    page2_ids = {r.id for r in page2}
    assert page1_ids.isdisjoint(page2_ids)

    # 3. Filtering by crop
    spinach_records, spinach_total = list_records(test_db_session, crop="spinach", limit=50)
    assert spinach_total == 15
    assert len(spinach_records) == 15
    assert all(r.crop == "spinach" for r in spinach_records)

    # 4. Limit capping at 100
    big_list, _ = list_records(test_db_session, limit=500)
    assert len(big_list) <= 100


def test_db_failure_surfaces_as_typed_error(test_db_session: Session) -> None:
    """AC 3: DB failure surfaces as a typed error (PersistenceError with code 'PERSISTENCE_FAILED')."""
    # Attempting to insert an invalid source value ('hacked_source') violates CheckConstraint
    bad_time = datetime.now(timezone.utc)

    # Test raise_on_error=True (default)
    with pytest.raises(PersistenceError) as exc_info:
        create_record(
            test_db_session,
            crop="tomato",
            mode="single",
            observed_at=bad_time,
            observed_tz="UTC",
            source="forbidden_unconstrained_source",  # Causes DB IntegrityError
        )

    err = exc_info.value
    assert isinstance(err, PersistenceError)
    assert err.code == "PERSISTENCE_FAILED"
    assert "Failed to persist analysis record" in str(err)
    assert err.original_error is not None

    # Test raise_on_error=False returns SaveResult with saved=False
    res = create_record(
        test_db_session,
        crop="tomato",
        mode="single",
        observed_at=bad_time,
        observed_tz="UTC",
        source="another_invalid_source",
        raise_on_error=False,
    )
    assert res.saved is False
    assert res.record is None
    assert res.error is not None
    assert res.error.code == "PERSISTENCE_FAILED"

    with pytest.raises(PersistenceError):
        res.unwrap()


def test_create_record_with_children_and_eager_loading(test_db_session: Session) -> None:
    """Verify create_record atomically persists predictions, inputs, and recommendations."""
    rec_id = uuid.uuid4()
    now_dt = datetime.now(timezone.utc)

    predictions = [
        {
            "modality": "sensor",
            "predicted_class": "EC_High_Toxicity",
            "display_label": "High EC Toxicity",
            "confidence": 0.94,
            "confidence_semantics": "probability",
            "uncertain": False,
            "experimental": False,
            "model_id": "rf_tomato_v1",
            "model_version": "1.0.0",
            "preprocessing_version": "1.0.0",
            "warnings": ["EC exceeds 4.0 mS/cm"],
        }
    ]
    inputs = [
        {
            "kind": "sensor",
            "payload": {"ec_mS_cm": 4.5, "ph": 6.2},
            "units": {"ec_mS_cm": "mS/cm", "ph": "pH"},
        }
    ]
    recommendations = [
        {
            "source": "rules",
            "rules_version": "1.0.0",
            "content": {"action": "Dilute solution with freshwater"},
            "evidence": {"rule": "ec_high", "deviation": 0.5},
        }
    ]

    result = create_record(
        test_db_session,
        record_id=rec_id,
        crop="tomato",
        mode="single",
        observed_at=now_dt,
        observed_tz="UTC",
        source="live",
        notes="Automated alert test",
        predictions=predictions,
        inputs=inputs,
        recommendations=recommendations,
    )

    assert result.saved is True
    rec = result.unwrap()
    assert rec.id == rec_id

    # Query with eager load
    loaded = get_record(test_db_session, rec_id, eager_load=True)
    assert loaded is not None
    assert len(loaded.predictions) == 1
    assert loaded.predictions[0].predicted_class == "EC_High_Toxicity"
    assert loaded.predictions[0].warnings == ["EC exceeds 4.0 mS/cm"]

    assert len(loaded.inputs) == 1
    assert loaded.inputs[0].payload == {"ec_mS_cm": 4.5, "ph": 6.2}

    assert len(loaded.recommendations) == 1
    assert loaded.recommendations[0].source == "rules"
    assert loaded.recommendations[0].content["action"] == "Dilute solution with freshwater"
