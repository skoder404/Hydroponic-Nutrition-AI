"""Integration tests for history round trip and pagination (T8-007).

Validates:
1. Timezone survives round trip (preserves observed time, timezone string, and source provenance).
2. Limit above 100 is capped (pagination limits, offset navigation, and X-Total-Count header).
3. Old record is not marked live ('historical' flag is True, and stale telemetry cannot be treated as live).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.database.repositories import create_record
from backend.app.database.session import get_db, get_engine
from backend.app.devices.adapter import MockAdapter
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def test_db_session(tmp_path: Path):
    """Create isolated SQLite database with Alembic migrations applied."""
    db_file = tmp_path / "test_history_integration.db"
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
    engine.dispose()


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


# ==============================================================================
# AC 1: Timezone survives round trip
# ==============================================================================


@pytest.mark.parametrize(
    "tz_name,observed_iso,expected_local_hour",
    [
        ("Asia/Kolkata", "2026-10-27T10:00:00Z", 15),  # UTC 10:00 -> IST 15:30
        ("America/New_York", "2026-10-27T18:00:00Z", 14),  # UTC 18:00 -> EDT 14:00
        ("Europe/London", "2026-07-15T12:00:00Z", 13),  # UTC 12:00 -> BST 13:00 (DST summer)
        ("Asia/Tokyo", "2026-10-27T01:00:00Z", 10),  # UTC 01:00 -> JST 10:00
        ("UTC", "2026-10-27T08:00:00Z", 8),  # UTC 08:00 -> UTC 08:00
    ],
)
def test_timezone_survives_round_trip(
    client: TestClient,
    tz_name: str,
    observed_iso: str,
    expected_local_hour: int,
) -> None:
    """AC 1: Timezone survives round trip across multiple global timezones."""
    payload = {
        "crop": "lettuce",
        "mode": "sensor",
        "source": "manual",
        "observed_at": observed_iso,
        "observed_tz": tz_name,
        "notes": f"Observation taken in timezone {tz_name}",
        "inputs": {"ph": 6.1, "ec_ms_cm": 1.45, "water_temp_c": 21.2},
    }

    # 1. POST /history
    create_resp = client.post("/history", json=payload)
    assert create_resp.status_code == 201, create_resp.text
    created = create_resp.json()

    assert created["observed_tz"] == tz_name
    assert created["source"] == "manual"
    assert created["historical"] is True
    record_id = created["id"]

    # 2. GET /history/{id}
    detail_resp = client.get(f"/history/{record_id}")
    assert detail_resp.status_code == 200, detail_resp.text
    detail = detail_resp.json()

    assert detail["id"] == record_id
    assert detail["observed_tz"] == tz_name
    assert detail["source"] == "manual"
    assert detail["historical"] is True

    # Validate timestamp instant and timezone localization
    retrieved_utc = datetime.fromisoformat(detail["observed_at"].replace("Z", "+00:00"))
    expected_utc = datetime.fromisoformat(observed_iso.replace("Z", "+00:00"))
    assert retrieved_utc == expected_utc

    # Convert to original site timezone and verify local hour matches
    local_dt = retrieved_utc.astimezone(ZoneInfo(tz_name))
    assert local_dt.hour == expected_local_hour

    # 3. Verify in paginated list GET /history
    list_resp = client.get(f"/history?crop=lettuce&limit=10")
    assert list_resp.status_code == 200
    records = list_resp.json()
    matching = [r for r in records if r["id"] == record_id]
    assert len(matching) == 1
    assert matching[0]["observed_tz"] == tz_name
    assert matching[0]["historical"] is True


# ==============================================================================
# AC 2: Limit above 100 is capped and pagination limits enforced
# ==============================================================================


def test_limit_above_100_is_capped_and_paginated(
    client: TestClient,
    test_db_session: Session,
) -> None:
    """AC 2: Limit above 100 is capped, offset paginates cleanly, and X-Total-Count is correct."""
    total_to_seed = 110
    base_time = datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc)

    # Seed 110 records with predictable order
    for i in range(total_to_seed):
        create_record(
            test_db_session,
            crop="spinach" if i % 2 == 0 else "tomato",
            mode="sensor" if i % 3 == 0 else "both",
            source="manual" if i % 2 == 0 else "live",
            observed_at=base_time + timedelta(minutes=i),
            observed_tz="UTC",
            notes=f"Pagination test record #{i:03d}",
        )

    # 1. Request limit=150 (exceeding 100 max cap)
    over_limit_resp = client.get("/history?limit=150")
    assert over_limit_resp.status_code == 200
    records = over_limit_resp.json()

    assert len(records) == 100, f"Expected 100 capped records, got {len(records)}"
    assert "X-Total-Count" in over_limit_resp.headers
    assert int(over_limit_resp.headers["X-Total-Count"]) == total_to_seed

    # 2. Request limit=250 (well above cap)
    way_over_resp = client.get("/history?limit=250")
    assert len(way_over_resp.json()) == 100

    # 3. Paging navigation: Page 1 (0..50), Page 2 (50..100), Page 3 (100..110)
    page_1 = client.get("/history?offset=0&limit=50").json()
    page_2 = client.get("/history?offset=50&limit=50").json()
    page_3 = client.get("/history?offset=100&limit=50").json()

    assert len(page_1) == 50
    assert len(page_2) == 50
    assert len(page_3) == 10

    ids_1 = {r["id"] for r in page_1}
    ids_2 = {r["id"] for r in page_2}
    ids_3 = {r["id"] for r in page_3}

    # Verify disjointness between pages
    assert ids_1.isdisjoint(ids_2)
    assert ids_2.isdisjoint(ids_3)
    assert ids_1.isdisjoint(ids_3)
    assert len(ids_1 | ids_2 | ids_3) == total_to_seed

    # 4. Filtered pagination with X-Total-Count
    filtered_resp = client.get("/history?crop=spinach&limit=150")
    assert filtered_resp.status_code == 200
    spinach_records = filtered_resp.json()
    spinach_total = int(filtered_resp.headers["X-Total-Count"])

    assert spinach_total == 55  # 110 / 2
    assert len(spinach_records) == 55
    assert all(r["crop"] == "spinach" for r in spinach_records)


# ==============================================================================
# AC 3: Old record is not marked live
# ==============================================================================


def test_old_record_is_not_marked_live(
    client: TestClient,
    test_db_session: Session,
) -> None:
    """AC 3: Old record has historical=True and is never treated as live telemetry."""
    # Create an old record from 2 days ago that originally had source='live'
    two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
    old_id = uuid.uuid4()

    create_record(
        test_db_session,
        record_id=old_id,
        crop="lettuce",
        mode="sensor",
        source="live",  # Originally ingested from a live device stream 2 days ago
        observed_at=two_days_ago,
        observed_tz="UTC",
        notes="Historical record from 2 days ago",
    )

    # 1. Fetch via GET /history/{id}
    detail_resp = client.get(f"/history/{old_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()

    # Invariant: Persisted historical records ALWAYS have historical=True
    assert detail["historical"] is True
    # The source provenance indicates how it was captured, but historical flag denotes it is NOT live now
    assert detail["source"] == "live"

    # 2. Fetch via GET /history
    list_resp = client.get(f"/history?crop=lettuce&limit=5")
    assert list_resp.status_code == 200
    records = list_resp.json()
    matching = [r for r in records if r["id"] == str(old_id)]
    assert len(matching) == 1
    assert matching[0]["historical"] is True

    # 3. Verify against DeviceAdapter freshness gate
    # Telemetry older than max_age_s MUST evaluate to 'stale' and can NEVER be treated as live
    adapter = MockAdapter(max_age_s=120)
    retrieved_time = datetime.fromisoformat(detail["observed_at"].replace("Z", "+00:00"))
    freshness_status, age = adapter.evaluate_reading_freshness(retrieved_time)

    assert freshness_status == "stale"
    assert age is not None
    assert age > 170000.0  # ~2 days in seconds


def test_newly_persisted_record_flagged_historical(client: TestClient) -> None:
    """AC 3: Newly created record via POST /history is immediately flagged historical=True."""
    now_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "crop": "tomato",
        "mode": "sensor",
        "source": "live",
        "observed_at": now_utc,
        "observed_tz": "America/New_York",
        "inputs": {"ph": 6.0, "ec_ms_cm": 2.2, "water_temp_c": 20.0},
    }

    resp = client.post("/history", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    # The persisted record is stored in history, so historical must be True
    assert data["historical"] is True
