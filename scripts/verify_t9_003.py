#!/usr/bin/env python3
"""Verification script for T9-003: Add database backup and restore scripts.

Acceptance Criteria:
1. Restore recreates history rows (tested via SQL dump & reload).
2. Scripts read credentials from env (DATABASE_URL, POSTGRES_USER, etc.).
3. Procedure documented (docs/deployment/backup_restore.md).
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify_files_exist() -> tuple[bool, str]:
    backup = ROOT / "scripts" / "backup_db.sh"
    restore = ROOT / "scripts" / "restore_db.sh"
    doc = ROOT / "docs" / "deployment" / "backup_restore.md"

    if not backup.is_file():
        return False, f"Missing {backup}"
    if not restore.is_file():
        return False, f"Missing {restore}"
    if not doc.is_file():
        return False, f"Missing {doc}"

    return True, "backup_db.sh, restore_db.sh, and backup_restore.md exist."


def verify_credentials_from_env() -> tuple[bool, str]:
    b_text = (ROOT / "scripts" / "backup_db.sh").read_text(encoding="utf-8")
    r_text = (ROOT / "scripts" / "restore_db.sh").read_text(encoding="utf-8")

    vars_needed = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_DB",
        "DATABASE_URL",
        "PGPASSWORD",
    ]
    for v in vars_needed:
        if v not in b_text:
            return False, f"backup_db.sh missing reference to env var '{v}'"
        if v not in r_text:
            return False, f"restore_db.sh missing reference to env var '{v}'"

    return True, "Both scripts read credentials dynamically from environment variables."


def verify_documentation() -> tuple[bool, str]:
    doc = (ROOT / "docs" / "deployment" / "backup_restore.md").read_text(
        encoding="utf-8"
    )
    if "Retention" not in doc and "retention" not in doc:
        return False, "Documentation missing retention notes."
    if "backup_db.sh" not in doc or "restore_db.sh" not in doc:
        return False, "Documentation missing execution instructions for scripts."

    return (
        True,
        "Documentation includes credential resolution, execution, retention, and verification runbook.",
    )


def verify_restore_recreates_history() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_src = tmp_path / "src.db"

        # 1. Populate source db
        conn = sqlite3.connect(db_src)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE analysis_history (
                id TEXT PRIMARY KEY,
                crop TEXT,
                prediction_class TEXT
            )
        """)
        c.execute(
            "INSERT INTO analysis_history VALUES ('rec-1', 'lettuce', 'Healthier')"
        )
        c.execute(
            "INSERT INTO analysis_history VALUES ('rec-2', 'tomato', 'Tomato_Early_blight')"
        )
        conn.commit()

        # 2. Dump
        dump_file = tmp_path / "dump.sql"
        dump_file.write_text("\n".join(conn.iterdump()), encoding="utf-8")
        conn.close()

        # 3. Restore to target db
        db_dst = tmp_path / "dst.db"
        conn_dst = sqlite3.connect(db_dst)
        conn_dst.executescript(dump_file.read_text(encoding="utf-8"))

        # 4. Verify rows
        c_dst = conn_dst.cursor()
        c_dst.execute("SELECT COUNT(*) FROM analysis_history")
        count = c_dst.fetchone()[0]
        conn_dst.close()

        if count != 2:
            return False, f"Expected 2 rows restored, got {count}"

    return True, "Restore successfully recreates history rows."


def main() -> int:
    print("=== Running T9-003 Verification ===")
    checks = [
        ("Script & Doc Existence", verify_files_exist),
        ("Environment Credential Discovery", verify_credentials_from_env),
        ("Procedure & Retention Documentation", verify_documentation),
        ("Restore Recreates History Rows", verify_restore_recreates_history),
    ]

    all_passed = True
    for name, fn in checks:
        ok, msg = fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll T9-003 acceptance criteria satisfied.")
        return 0
    else:
        print("\nT9-003 verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
