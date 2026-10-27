#!/usr/bin/env python3
"""Verification script for T2-023: Add history indexes.

Validates the acceptance criteria:
1. Migration applies and reverts cleanly (alembic upgrade / downgrade).
2. EXPLAIN on the list query uses an index (both crop+created_at and created_at DESC).
3. Migration file is registered and committed in alembic/versions/.

Usage:
    python scripts/verify_history_indexes.py
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlalchemy as sa  # noqa: E402
from alembic.config import Config  # noqa: E402

from alembic import command  # noqa: E402
from backend.app.database.session import get_engine  # noqa: E402


def verify_migration_file_exists() -> tuple[bool, str]:
    """Check Criterion 3: Migration file exists in alembic/versions/."""
    versions_dir = PROJECT_ROOT / "alembic" / "versions"
    matching = list(versions_dir.glob("*_add_history_indexes.py"))
    if not matching:
        return (
            False,
            "FAIL: Migration file *_add_history_indexes.py not found in alembic/versions/.",
        )
    return True, f"PASS: Migration file found: {matching[0].name}"


def verify_migration_applies_and_reverts() -> tuple[bool, str]:
    """Check Criterion 1: Migration applies and reverts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "verify_indexes.db"
        db_url = f"sqlite:///{db_path.as_posix()}"

        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        cfg.set_main_option("sqlalchemy.url", db_url)
        cfg.attributes["configure_logger"] = False

        # 1. Upgrade to base table migration
        command.upgrade(cfg, "73af9478eca4")
        engine = get_engine(db_url)
        insp = sa.inspect(engine)
        init_indexes = {i["name"] for i in insp.get_indexes("analysis_records")}
        if "ix_analysis_records_crop_created_at" in init_indexes:
            return False, "FAIL: History index already present in initial migration!"
        engine.dispose()

        # 2. Upgrade to head
        command.upgrade(cfg, "head")
        engine = get_engine(db_url)
        insp = sa.inspect(engine)
        head_indexes = {i["name"] for i in insp.get_indexes("analysis_records")}
        pred_indexes = {i["name"] for i in insp.get_indexes("predictions")}

        if "ix_analysis_records_created_at_desc" not in head_indexes:
            return (
                False,
                "FAIL: ix_analysis_records_created_at_desc missing after upgrade.",
            )
        if "ix_analysis_records_crop_created_at" not in head_indexes:
            return (
                False,
                "FAIL: ix_analysis_records_crop_created_at missing after upgrade.",
            )
        if "ix_predictions_record_id" not in pred_indexes:
            return False, "FAIL: ix_predictions_record_id missing after upgrade."
        engine.dispose()

        # 3. Downgrade by 1
        command.downgrade(cfg, "-1")
        engine = get_engine(db_url)
        insp = sa.inspect(engine)
        reverted_indexes = {i["name"] for i in insp.get_indexes("analysis_records")}
        if "ix_analysis_records_crop_created_at" in reverted_indexes:
            return False, "FAIL: Index not dropped on downgrade."
        engine.dispose()

        # 4. Re-upgrade to head
        command.upgrade(cfg, "head")
        engine = get_engine(db_url)
        insp = sa.inspect(engine)
        re_head_indexes = {i["name"] for i in insp.get_indexes("analysis_records")}
        if "ix_analysis_records_crop_created_at" not in re_head_indexes:
            return False, "FAIL: Index not re-created on second upgrade."
        engine.dispose()

    return True, "PASS: Migration applies and reverts successfully."


def verify_explain_on_list_query_uses_index() -> tuple[bool, str]:
    """Check Criterion 2: EXPLAIN on the list query uses an index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "verify_explain.db"
        db_url = f"sqlite:///{db_path.as_posix()}"

        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
        cfg.set_main_option("sqlalchemy.url", db_url)
        cfg.attributes["configure_logger"] = False

        command.upgrade(cfg, "head")
        engine = get_engine(db_url)

        with engine.connect() as conn:
            # 1. Filtered query
            stmt_crop = sa.text(
                "EXPLAIN QUERY PLAN SELECT * FROM analysis_records "
                "WHERE crop = :crop ORDER BY created_at DESC LIMIT 20"
            )
            rows_crop = conn.execute(stmt_crop, {"crop": "lettuce"}).fetchall()
            text_crop = " ".join(str(r[-1]) for r in rows_crop)

            if "ix_analysis_records_crop_created_at" not in text_crop:
                return (
                    False,
                    f"FAIL: Crop list query did not use index:\n{text_crop}",
                )

            # 2. Recent records query
            stmt_recent = sa.text(
                "EXPLAIN QUERY PLAN SELECT * FROM analysis_records "
                "ORDER BY created_at DESC LIMIT 20"
            )
            rows_recent = conn.execute(stmt_recent).fetchall()
            text_recent = " ".join(str(r[-1]) for r in rows_recent)

            if "ix_analysis_records_created_at_desc" not in text_recent:
                return (
                    False,
                    f"FAIL: Recent records query did not use index:\n{text_recent}",
                )

            # 3. Foreign key query
            stmt_pred = sa.text(
                "EXPLAIN QUERY PLAN SELECT * FROM predictions WHERE record_id = :id"
            )
            rows_pred = conn.execute(stmt_pred, {"id": str(uuid.uuid4())}).fetchall()
            text_pred = " ".join(str(r[-1]) for r in rows_pred)

            if "ix_predictions_record_id" not in text_pred:
                return (
                    False,
                    f"FAIL: Predictions foreign key query did not use index:\n{text_pred}",
                )

        engine.dispose()

    return (
        True,
        "PASS: EXPLAIN on list queries and foreign key lookups uses the indexes.",
    )


def main() -> int:
    print("=" * 70)
    print("T2-023 Verification: Add History Indexes")
    print("=" * 70)

    checks = [
        (
            "Criterion 1: Migration applies and reverts",
            verify_migration_applies_and_reverts,
        ),
        (
            "Criterion 2: EXPLAIN on list queries uses indexes",
            verify_explain_on_list_query_uses_index,
        ),
        ("Criterion 3: Migration file committed", verify_migration_file_exists),
    ]

    all_passed = True
    for name, check_fn in checks:
        ok, msg = check_fn()
        status = "[OK]  " if ok else "[FAIL]"
        print(f"{status} {name}")
        print(f"       {msg}")
        if not ok:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL T2-023 VERIFICATION CHECKS PASSED!")
        return 0
    else:
        print("SOME CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
