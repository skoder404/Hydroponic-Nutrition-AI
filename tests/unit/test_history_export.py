"""Unit tests for History CSV Export (T10-005).

Acceptance Criteria:
- [ ] Cells starting with = + - @ are neutralised
- [ ] Export respects filters
- [ ] Timestamps include timezone
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.api.v1.history import sanitize_csv_cell
from backend.app.database.session import get_db, get_engine
from backend.app.main import create_app


@pytest.fixture
def test_db_session(tmp_path: Path):
    """Create isolated SQLite database with migrations applied."""
    db_file = tmp_path / "test_history_export.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    ini_path = ROOT / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")

    engine = get_engine(db_url)
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(test_db_session: Session):
    """Create FastAPI test client with database dependency override."""
    app = create_app(configure_logging=False)

    def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_sanitize_csv_cell_neutralises_injection():
    """AC 1: Cells starting with = + - @ are neutralised."""
    assert sanitize_csv_cell("=cmd|'/C calc'!A0") == "'=cmd|'/C calc'!A0"
    assert sanitize_csv_cell("+12345") == "'+12345"
    assert sanitize_csv_cell("-SOMETHING") == "'-SOMETHING"
    assert sanitize_csv_cell("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert sanitize_csv_cell("\tTABBED") == "'\tTABBED"
    assert sanitize_csv_cell("\rRETURN") == "'\rRETURN"

    # Benign text is unaffected
    assert sanitize_csv_cell("lettuce") == "lettuce"
    assert sanitize_csv_cell("Healthy foliage") == "Healthy foliage"
    assert sanitize_csv_cell("2026-10-27") == "2026-10-27"
    assert sanitize_csv_cell(None) == ""


def test_export_respects_filters_and_includes_timezone(client: TestClient):
    """AC 2 & AC 3: Export respects filters and timestamps include timezone."""
    # Seed 3 records across different crops and sources
    # 1. Lettuce manual
    client.post(
        "/history",
        json={
            "crop": "lettuce",
            "mode": "sensor",
            "source": "manual",
            "observed_at": "2026-10-27T10:00:00Z",
            "observed_tz": "America/New_York",
            "notes": "=1+1 formula note",
        },
    )

    # 2. Spinach live
    client.post(
        "/history",
        json={
            "crop": "spinach",
            "mode": "image",
            "source": "live",
            "observed_at": "2026-10-27T14:30:00Z",
            "observed_tz": "Asia/Kolkata",
            "notes": "-danger negative note",
        },
    )

    # 3. Tomato manual
    client.post(
        "/history",
        json={
            "crop": "tomato",
            "mode": "both",
            "source": "manual",
            "observed_at": "2026-10-28T09:15:00Z",
            "observed_tz": "Europe/London",
            "notes": "@mention note",
        },
    )

    # 1. Unfiltered Export
    resp = client.get("/history/export")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert 'attachment; filename="history_export.csv"' in resp.headers["content-disposition"]

    reader = list(csv.reader(io.StringIO(resp.text)))
    assert len(reader) == 4  # 1 header + 3 rows
    header = reader[0]
    assert "observed_at" in header
    assert "observed_tz" in header
    assert "crop" in header
    assert "source" in header

    # Verify formula neutralisation in exported CSV
    notes_idx = header.index("notes")
    all_notes = [row[notes_idx] for row in reader[1:]]
    assert "'=1+1 formula note" in all_notes
    assert "'-danger negative note" in all_notes
    assert "'@mention note" in all_notes

    # Verify timestamps include timezone
    obs_idx = header.index("observed_at")
    tz_idx = header.index("observed_tz")
    for row in reader[1:]:
        assert row[tz_idx] in ("America/New_York", "Asia/Kolkata", "Europe/London")
        # Timestamp cell must include timezone string or offset
        assert row[tz_idx] in row[obs_idx] or "+" in row[obs_idx] or "Z" in row[obs_idx]

    # 2. Filtered Export by crop=spinach (AC 2)
    resp_spinach = client.get("/history/export?crop=spinach")
    assert resp_spinach.status_code == 200
    reader_spinach = list(csv.reader(io.StringIO(resp_spinach.text)))
    assert len(reader_spinach) == 2  # header + 1 row
    assert reader_spinach[1][header.index("crop")] == "spinach"
    assert reader_spinach[1][header.index("source")] == "live"

    # 3. Filtered Export by source=manual
    resp_manual = client.get("/history/export?source=manual")
    assert resp_manual.status_code == 200
    reader_manual = list(csv.reader(io.StringIO(resp_manual.text)))
    assert len(reader_manual) == 3  # header + 2 rows
    for row in reader_manual[1:]:
        assert row[header.index("source")] == "manual"
