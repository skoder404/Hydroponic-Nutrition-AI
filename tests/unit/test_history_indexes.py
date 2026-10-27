"""Unit tests for history indexes migration (T2-023).

Acceptance Criteria:
- [x] Migration applies and reverts (upgrade to head, downgrade -1, upgrade back)
- [x] EXPLAIN on the list query uses an index
- [x] Migration is committed
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
from backend.app.database.models import AnalysisRecord, Prediction
from backend.app.database.repositories import list_records
from backend.app.database.session import get_engine

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Return isolated SQLite database file path."""
    return tmp_path / "test_history_indexes.db"


@pytest.fixture
def alembic_cfg(test_db_path: Path) -> Config:
    """Return Alembic config configured with test database URL."""
    ini_path = ROOT / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{test_db_path.as_posix()}")
    cfg.attributes["configure_logger"] = False
    return cfg


def test_migration_applies_and_reverts(alembic_cfg: Config, test_db_path: Path) -> None:
    """AC 1: Migration applies and reverts cleanly."""
    db_url = f"sqlite:///{test_db_path.as_posix()}"

    # 1. Upgrade to initial migration 73af9478eca4 (tables exist, no history indexes)
    command.upgrade(alembic_cfg, "73af9478eca4")
    engine = get_engine(db_url)
    inspector = sa.inspect(engine)

    analysis_indexes_init = {
        idx["name"] for idx in inspector.get_indexes("analysis_records")
    }
    pred_indexes_init = {idx["name"] for idx in inspector.get_indexes("predictions")}

    assert "ix_analysis_records_created_at_desc" not in analysis_indexes_init
    assert "ix_analysis_records_crop_created_at" not in analysis_indexes_init
    assert "ix_predictions_record_id" not in pred_indexes_init
    engine.dispose()

    # 2. Upgrade to head (adds history indexes: c4b1e892d3f1)
    command.upgrade(alembic_cfg, "head")
    engine = get_engine(db_url)
    inspector = sa.inspect(engine)

    analysis_indexes_head = {
        idx["name"] for idx in inspector.get_indexes("analysis_records")
    }
    pred_indexes_head = {idx["name"] for idx in inspector.get_indexes("predictions")}

    assert "ix_analysis_records_created_at_desc" in analysis_indexes_head
    assert "ix_analysis_records_crop_created_at" in analysis_indexes_head
    assert "ix_predictions_record_id" in pred_indexes_head
    engine.dispose()

    # 3. Downgrade by 1 revision (reverts c4b1e892d3f1 back to 73af9478eca4)
    command.downgrade(alembic_cfg, "-1")
    engine = get_engine(db_url)
    inspector = sa.inspect(engine)

    analysis_indexes_reverted = {
        idx["name"] for idx in inspector.get_indexes("analysis_records")
    }
    pred_indexes_reverted = {
        idx["name"] for idx in inspector.get_indexes("predictions")
    }

    assert "ix_analysis_records_created_at_desc" not in analysis_indexes_reverted
    assert "ix_analysis_records_crop_created_at" not in analysis_indexes_reverted
    assert "ix_predictions_record_id" not in pred_indexes_reverted
    engine.dispose()

    # 4. Upgrade back to head (re-applies cleanly)
    command.upgrade(alembic_cfg, "head")
    engine = get_engine(db_url)
    inspector = sa.inspect(engine)

    analysis_indexes_final = {
        idx["name"] for idx in inspector.get_indexes("analysis_records")
    }
    assert "ix_analysis_records_created_at_desc" in analysis_indexes_final
    assert "ix_analysis_records_crop_created_at" in analysis_indexes_final
    engine.dispose()


def test_explain_on_list_query_uses_index(
    alembic_cfg: Config, test_db_path: Path
) -> None:
    """AC 2: EXPLAIN on the list query uses an index."""
    db_url = f"sqlite:///{test_db_path.as_posix()}"

    # 1. Upgrade to head
    command.upgrade(alembic_cfg, "head")
    engine = get_engine(db_url)

    with engine.connect() as conn:
        # A. Filtered list query: crop filter + created_at DESC
        stmt_crop = sa.text(
            "EXPLAIN QUERY PLAN SELECT * FROM analysis_records "
            "WHERE crop = :crop ORDER BY created_at DESC LIMIT 20"
        )
        plan_crop_rows = conn.execute(stmt_crop, {"crop": "tomato"}).fetchall()
        plan_crop_text = " ".join(str(r[-1]) for r in plan_crop_rows)

        assert "USING INDEX ix_analysis_records_crop_created_at" in plan_crop_text, (
            f"Expected index ix_analysis_records_crop_created_at to be used, got:\n{plan_crop_text}"
        )

        # B. Unfiltered list query: created_at DESC
        stmt_recent = sa.text(
            "EXPLAIN QUERY PLAN SELECT * FROM analysis_records "
            "ORDER BY created_at DESC LIMIT 20"
        )
        plan_recent_rows = conn.execute(stmt_recent).fetchall()
        plan_recent_text = " ".join(str(r[-1]) for r in plan_recent_rows)

        assert "USING INDEX ix_analysis_records_created_at_desc" in plan_recent_text, (
            f"Expected index ix_analysis_records_created_at_desc to be used, got:\n{plan_recent_text}"
        )

        # C. Foreign key lookup on predictions: record_id
        stmt_pred = sa.text(
            "EXPLAIN QUERY PLAN SELECT * FROM predictions WHERE record_id = :rec_id"
        )
        plan_pred_rows = conn.execute(
            stmt_pred, {"rec_id": str(uuid.uuid4())}
        ).fetchall()
        plan_pred_text = " ".join(str(r[-1]) for r in plan_pred_rows)

        assert "USING INDEX ix_predictions_record_id" in plan_pred_text, (
            f"Expected index ix_predictions_record_id to be used, got:\n{plan_pred_text}"
        )

    engine.dispose()

    # Contrast check: revert migration and verify that queries fail to use these indexes
    command.downgrade(alembic_cfg, "-1")
    engine = get_engine(db_url)

    with engine.connect() as conn:
        plan_no_index = conn.execute(stmt_crop, {"crop": "tomato"}).fetchall()
        no_index_text = " ".join(str(r[-1]) for r in plan_no_index)
        assert "ix_analysis_records" not in no_index_text
        assert "USE TEMP B-TREE FOR ORDER BY" in no_index_text

    engine.dispose()


def test_repository_list_records_execution(
    alembic_cfg: Config, test_db_path: Path
) -> None:
    """Verify list_records repository function runs correctly with the indexes in place."""
    db_url = f"sqlite:///{test_db_path.as_posix()}"
    command.upgrade(alembic_cfg, "head")
    engine = get_engine(db_url)

    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        for i in range(5):
            rec = AnalysisRecord(
                id=uuid.uuid4(),
                crop="tomato" if i % 2 == 0 else "lettuce",
                mode="single",
                created_at=now,
                observed_at=now,
                observed_tz="UTC",
                source="manual",
            )
            pred = Prediction(
                id=uuid.uuid4(),
                record=rec,
                modality="sensor",
                predicted_class="Normal",
                display_label="Normal",
                confidence=0.95,
                confidence_semantics="score",
                uncertain=False,
                experimental=False,
                model_id="dt_v1",
                model_version="1.0",
                preprocessing_version="1.0",
                warnings=[],
            )
            session.add_all([rec, pred])
        session.commit()

        # Query filtered by crop
        records_tomato, total_tomato = list_records(session, crop="tomato", limit=10)
        assert total_tomato == 3
        assert len(records_tomato) == 3
        assert all(r.crop == "tomato" for r in records_tomato)

        # Query all records
        all_records, total = list_records(session, limit=10)
        assert total == 5
        assert len(all_records) == 5

    engine.dispose()
