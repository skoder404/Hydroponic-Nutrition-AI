"""Unit tests for SQLAlchemy database models and Alembic migrations (T2-021).

Acceptance Criteria Verified:
- [x] alembic upgrade head creates all tables
- [x] Downgrade works
- [x] source rejects values outside the allowed set ('manual', 'live', 'imported')
- [x] Modality, kind, and recommendation constraints enforced
- [x] Parent cascade deletion purges child records
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from sqlalchemy.orm import Session

from alembic import command
from backend.app.database.models import (
    AnalysisRecord,
    Input,
    Prediction,
    Recommendation,
)
from backend.app.database.session import get_engine

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def test_db_url(tmp_path: Path) -> str:
    """Return an isolated SQLite database URL for tests."""
    db_file = tmp_path / "test_migration.db"
    return f"sqlite:///{db_file.as_posix()}"


@pytest.fixture
def alembic_config(test_db_url: str) -> Config:
    """Return Alembic config configured with test database URL."""
    ini_path = ROOT / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", test_db_url)
    cfg.attributes["configure_logger"] = False
    return cfg


def test_alembic_upgrade_and_downgrade(alembic_config: Config, test_db_url: str) -> None:
    """AC 1 & 2: alembic upgrade head creates all tables; downgrade works."""
    # 1. Upgrade to head
    command.upgrade(alembic_config, "head")

    engine = get_engine(test_db_url)
    inspector = sa.inspect(engine)
    tables = set(inspector.get_table_names())

    expected_tables = {"analysis_records", "predictions", "inputs", "recommendations", "alembic_version"}
    assert expected_tables.issubset(tables), f"Missing tables after upgrade: {expected_tables - tables}"

    # Verify column presence in analysis_records
    cols = {col["name"] for col in inspector.get_columns("analysis_records")}
    assert {"id", "crop", "mode", "created_at", "observed_at", "observed_tz", "source", "notes"}.issubset(cols)

    # 2. Downgrade to base
    command.downgrade(alembic_config, "base")

    # Re-inspect tables after downgrade
    inspector = sa.inspect(engine)
    remaining_tables = set(inspector.get_table_names())
    app_tables = {"analysis_records", "predictions", "inputs", "recommendations"}
    assert not app_tables.intersection(remaining_tables), f"Tables remained after downgrade: {app_tables.intersection(remaining_tables)}"

    # 3. Upgrade back to head for subsequent tests
    command.upgrade(alembic_config, "head")


def test_analysis_records_source_constraint(alembic_config: Config, test_db_url: str) -> None:
    """AC 3: source rejects values outside ('manual', 'live', 'imported')."""
    command.upgrade(alembic_config, "head")
    engine = get_engine(test_db_url)

    # Allowed sources succeed
    for valid_source in ("manual", "live", "imported"):
        with Session(engine) as session:
            rec = AnalysisRecord(
                id=uuid.uuid4(),
                crop="lettuce",
                mode="single",
                observed_at=datetime.now(timezone.utc),
                observed_tz="UTC",
                source=valid_source,
            )
            session.add(rec)
            session.commit()

    # Disallowed source rejected by CheckConstraint
    with Session(engine) as session:
        bad_rec = AnalysisRecord(
            id=uuid.uuid4(),
            crop="lettuce",
            mode="single",
            observed_at=datetime.now(timezone.utc),
            observed_tz="UTC",
            source="invalid_source_value",
        )
        session.add(bad_rec)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_child_table_constraints(alembic_config: Config, test_db_url: str) -> None:
    """Verify modality, kind, and recommendation source CheckConstraints."""
    command.upgrade(alembic_config, "head")
    engine = get_engine(test_db_url)

    rec_id = uuid.uuid4()
    with Session(engine) as session:
        rec = AnalysisRecord(
            id=rec_id,
            crop="spinach",
            mode="fusion",
            observed_at=datetime.now(timezone.utc),
            observed_tz="America/New_York",
            source="live",
        )
        session.add(rec)
        session.commit()

    # Invalid prediction modality
    with Session(engine) as session:
        bad_pred = Prediction(
            id=uuid.uuid4(),
            record_id=rec_id,
            modality="radar_synthetic",  # Disallowed
            predicted_class="Normal",
            display_label="Normal",
            confidence=0.88,
            model_id="test_model",
            model_version="1.0",
            preprocessing_version="1.0",
            warnings=[],
        )
        session.add(bad_pred)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()

    # Invalid input kind
    with Session(engine) as session:
        bad_inp = Input(
            id=uuid.uuid4(),
            record_id=rec_id,
            kind="audio_stream",  # Disallowed
            payload={"frequency": 440},
        )
        session.add(bad_inp)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()

    # Invalid recommendation source
    with Session(engine) as session:
        bad_rec_item = Recommendation(
            id=uuid.uuid4(),
            record_id=rec_id,
            source="external_bot",  # Disallowed
            content={"msg": "hello"},
            evidence={},
        )
        session.add(bad_rec_item)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_cascade_delete(alembic_config: Config, test_db_url: str) -> None:
    """Verify deleting an analysis_record cascades to predictions, inputs, and recommendations."""
    command.upgrade(alembic_config, "head")
    engine = get_engine(test_db_url)

    rec_id = uuid.uuid4()
    with Session(engine) as session:
        rec = AnalysisRecord(
            id=rec_id,
            crop="tomato",
            mode="single",
            observed_at=datetime.now(timezone.utc),
            observed_tz="Asia/Kolkata",
            source="manual",
        )
        pred = Prediction(
            id=uuid.uuid4(),
            record_id=rec_id,
            modality="sensor",
            predicted_class="Normal",
            display_label="Normal",
            confidence=0.92,
            model_id="rf_tomato_v1",
            model_version="1.0",
            preprocessing_version="1.0",
            warnings=[],
        )
        inp = Input(
            id=uuid.uuid4(),
            record_id=rec_id,
            kind="sensor",
            payload={"ec": 2.2, "ph": 6.3},
        )
        rec_item = Recommendation(
            id=uuid.uuid4(),
            record_id=rec_id,
            source="rules",
            content={"advice": "Maintain EC within optimal band"},
            evidence={"rule": "EC_normal"},
        )
        session.add_all([rec, pred, inp, rec_item])
        session.commit()

    # Confirm created
    with Session(engine) as session:
        assert session.get(AnalysisRecord, rec_id) is not None
        assert session.query(Prediction).filter_by(record_id=rec_id).count() == 1
        assert session.query(Input).filter_by(record_id=rec_id).count() == 1
        assert session.query(Recommendation).filter_by(record_id=rec_id).count() == 1

        # Delete parent
        parent = session.get(AnalysisRecord, rec_id)
        session.delete(parent)
        session.commit()

    # Confirm cascade deleted children
    with Session(engine) as session:
        assert session.get(AnalysisRecord, rec_id) is None
        assert session.query(Prediction).filter_by(record_id=rec_id).count() == 0
        assert session.query(Input).filter_by(record_id=rec_id).count() == 0
        assert session.query(Recommendation).filter_by(record_id=rec_id).count() == 0
