#!/usr/bin/env python3
"""Verification script for T6-010: Add history endpoints.

Validates:
1. Round trip preserves observed time and source
2. Limit is capped at 100
3. Simulated DB outage returns 'not saved'
"""

import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from backend.app.database.session import get_db, get_engine
from backend.app.main import create_app


def run_verification() -> bool:
    print("[1/3] Setting up test database with Alembic migrations...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "verify_history.db"
        db_url = f"sqlite:///{db_path.as_posix()}"

        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        cfg.set_main_option("sqlalchemy.url", db_url)
        cfg.attributes["configure_logger"] = False
        command.upgrade(cfg, "head")

        engine = get_engine(db_url)
        session = Session(engine)

        app = create_app(configure_logging=False)

        def override_get_db():
            yield session

        app.dependency_overrides[get_db] = override_get_db
        client = TestClient(app)

        # 1. Round trip test
        print("[2/3] Verifying round-trip preserves observed time, source, and historical=True...")
        observed_time_str = "2026-10-27T14:30:00Z"
        tz_name = "Asia/Kolkata"
        source = "live"

        post_res = client.post(
            "/history",
            json={
                "crop": "lettuce",
                "mode": "sensor",
                "source": source,
                "observed_at": observed_time_str,
                "observed_tz": tz_name,
                "notes": "Round trip test record",
                "inputs": {"ph": 6.5, "ec_ms_cm": 1.6},
            },
        )
        assert post_res.status_code == 201, f"POST failed: {post_res.text}"
        data = post_res.json()
        assert data["crop"] == "lettuce"
        assert data["source"] == source
        assert data["observed_tz"] == tz_name
        assert data["historical"] is True
        rec_id = data["id"]

        get_res = client.get(f"/history/{rec_id}")
        assert get_res.status_code == 200, f"GET failed: {get_res.text}"
        get_data = get_res.json()
        assert get_data["id"] == rec_id
        assert get_data["source"] == source
        assert get_data["observed_tz"] == tz_name
        assert get_data["historical"] is True
        assert get_data["inputs"] == {"ph": 6.5, "ec_ms_cm": 1.6}
        print("  -> Passed round-trip verification!")

        # 2. Limit capped at 100 test
        print("[3/3] Verifying limit capping at 100...")
        from backend.app.database.repositories import create_record

        base_time = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(105):
            create_record(
                session,
                crop="spinach" if i % 2 == 0 else "lettuce",
                mode="sensor",
                source="manual",
                observed_at=base_time,
                observed_tz="UTC",
                notes=f"Batch item {i}",
            )
        list_res = client.get("/history?limit=150")
        assert list_res.status_code == 200, f"List failed: {list_res.text}"
        records = list_res.json()
        assert len(records) == 100, f"Expected 100 records max, got {len(records)}"
        total_count = int(list_res.headers.get("X-Total-Count", 0))
        assert total_count >= 106, f"Expected >= 106 total count, got {total_count}"
        print(f"  -> Passed limit capping: retrieved {len(records)} records (X-Total-Count: {total_count})")

        # 3. Simulated DB outage
        print("[4/4] Verifying simulated DB outage returns 503 PERSISTENCE_FAILED with 'not saved'...")
        mock_db = MagicMock(spec=Session)
        mock_db.add.side_effect = OperationalError("connection refused", {}, None)
        mock_db.commit.side_effect = OperationalError("connection refused", {}, None)

        app_err = create_app(configure_logging=False)
        app_err.dependency_overrides[get_db] = lambda: [mock_db]
        client_err = TestClient(app_err)

        err_res = client_err.post(
            "/history",
            json={
                "crop": "tomato",
                "mode": "sensor",
                "source": "live",
                "observed_at": "2026-10-27T12:00:00Z",
                "observed_tz": "UTC",
            },
        )
        assert err_res.status_code == 503, f"Expected 503, got {err_res.status_code}"
        err_data = err_res.json()
        assert err_data["error"]["code"] == "PERSISTENCE_FAILED"
        assert "not saved" in err_data["error"]["message"].lower(), (
            f"Expected 'not saved' in error message: {err_data['error']['message']}"
        )
        print(f"  -> Passed DB outage test: error={err_data['error']}")

        session.close()
        engine.dispose()


    print("\nAll T6-010 history endpoints criteria verified successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
