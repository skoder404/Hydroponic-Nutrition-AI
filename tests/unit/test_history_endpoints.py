"""Unit tests for History API endpoints (T6-010).

Acceptance Criteria:
- [x] Round trip preserves observed time and source
- [x] Limit is capped at 100
- [x] Simulated DB outage returns 'not saved'
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.database.session import get_db, get_engine
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def test_db_session(tmp_path: Path):
    """Create isolated SQLite database with migrations applied and return a sessionmaker / session."""
    db_file = tmp_path / "test_history.db"
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


def test_round_trip_preserves_observed_time_and_source(client: TestClient) -> None:
    """AC 1: Round trip preserves observed time and source."""
    observed_time_str = "2026-10-27T14:30:00Z"
    tz_name = "Asia/Kolkata"
    source = "live"

    payload = {
        "crop": "lettuce",
        "mode": "sensor",
        "source": source,
        "observed_at": observed_time_str,
        "observed_tz": tz_name,
        "notes": "Greenhouse Bay 1 baseline reading",
        "inputs": {"ph": 6.2, "ec_ms_cm": 1.5, "water_temp_c": 21.0},
    }

    # 1. Create record via POST /history
    resp = client.post("/history", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()

    assert data["crop"] == "lettuce"
    assert data["source"] == source
    assert data["observed_tz"] == tz_name
    assert data["historical"] is True
    assert "id" in data
    rec_id = data["id"]

    # Verify observed_at timestamp matches ISO representation
    created_dt = datetime.fromisoformat(data["observed_at"].replace("Z", "+00:00"))
    expected_dt = datetime(2026, 10, 27, 14, 30, 0, tzinfo=timezone.utc)
    assert created_dt == expected_dt

    # 2. Retrieve record via GET /history/{id}
    get_resp = client.get(f"/history/{rec_id}")
    assert get_resp.status_code == 200, get_resp.text
    get_data = get_resp.json()

    assert get_data["id"] == rec_id
    assert get_data["source"] == source
    assert get_data["observed_tz"] == tz_name
    assert get_data["historical"] is True
    fetched_dt = datetime.fromisoformat(get_data["observed_at"].replace("Z", "+00:00"))
    assert fetched_dt == expected_dt
    assert get_data["inputs"] == {"ph": 6.2, "ec_ms_cm": 1.5, "water_temp_c": 21.0}


def test_limit_is_capped_at_100(client: TestClient, test_db_session: Session) -> None:
    """AC 2: Limit is capped at 100."""
    from backend.app.database.repositories import create_record

    # Seed 105 records directly via repository
    base_time = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(105):
        create_record(
            test_db_session,
            crop="spinach" if i % 2 == 0 else "lettuce",
            mode="sensor",
            source="manual",
            observed_at=base_time,
            observed_tz="UTC",
            notes=f"Batch record {i}",
        )

    # Request limit=150 (exceeding max 100)
    list_resp = client.get("/history?limit=150")
    assert list_resp.status_code == 200
    records = list_resp.json()

    # Must be capped at exactly 100
    assert len(records) == 100

    # Header X-Total-Count must reflect total records count (105)
    assert "X-Total-Count" in list_resp.headers
    assert int(list_resp.headers["X-Total-Count"]) >= 105



def test_simulated_db_outage_returns_not_saved() -> None:
    """AC 3: Simulated DB outage returns 'not saved'."""
    app = create_app(configure_logging=False)

    # Mock DB that raises an OperationalError simulating disconnection / DB down
    mock_session = MagicMock(spec=Session)
    mock_session.add.side_effect = OperationalError("connection failure", {}, None)
    mock_session.commit.side_effect = OperationalError("connection failure", {}, None)

    def override_broken_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_broken_db
    with TestClient(app) as client:
        payload = {
            "crop": "lettuce",
            "mode": "sensor",
            "source": "live",
            "observed_at": "2026-10-27T12:00:00Z",
            "observed_tz": "UTC",
        }
        resp = client.post("/history", json=payload)

        # Must return 503 Service Unavailable (PERSISTENCE_FAILED)
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "PERSISTENCE_FAILED"

        # Explicitly verify 'not saved' appears in error message
        assert "not saved" in data["error"]["message"].lower()

    app.dependency_overrides.clear()


def test_get_history_not_found(client: TestClient) -> None:
    """Non-existent UUID returns 404 NOT_FOUND."""
    random_id = uuid.uuid4()
    resp = client.get(f"/history/{random_id}")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == "NOT_FOUND"
