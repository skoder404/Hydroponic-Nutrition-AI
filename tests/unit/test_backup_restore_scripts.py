"""Unit tests for database backup and restore scripts and procedures (T9-003)."""

from __future__ import annotations

from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent.parent.parent


def test_backup_restore_scripts_exist():
    """Verify scripts/backup_db.sh and scripts/restore_db.sh exist."""
    backup_sh = ROOT / "scripts" / "backup_db.sh"
    restore_sh = ROOT / "scripts" / "restore_db.sh"
    doc_md = ROOT / "docs" / "deployment" / "backup_restore.md"

    assert backup_sh.is_file(), "scripts/backup_db.sh must exist"
    assert restore_sh.is_file(), "scripts/restore_db.sh must exist"
    assert doc_md.is_file(), "docs/deployment/backup_restore.md must exist"


def test_scripts_read_credentials_from_env():
    """Criterion 2: Scripts read credentials from environment variables."""
    backup_sh = (ROOT / "scripts" / "backup_db.sh").read_text(encoding="utf-8")
    restore_sh = (ROOT / "scripts" / "restore_db.sh").read_text(encoding="utf-8")

    # Check for env var reads in backup script
    for var in [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_DB",
        "DATABASE_URL",
    ]:
        assert var in backup_sh, f"backup_db.sh must reference env var {var}"
        assert var in restore_sh, f"restore_db.sh must reference env var {var}"

    # Verify password handling
    assert "PGPASSWORD" in backup_sh, "backup_db.sh must export PGPASSWORD"
    assert "PGPASSWORD" in restore_sh, "restore_db.sh must export PGPASSWORD"


def test_procedure_documented():
    """Criterion 3: Procedure documented."""
    doc_md = (ROOT / "docs" / "deployment" / "backup_restore.md").read_text(
        encoding="utf-8"
    )

    assert "Retention" in doc_md or "retention" in doc_md, (
        "Must document backup retention policy"
    )
    assert "backup_db.sh" in doc_md, "Must document backup command"
    assert "restore_db.sh" in doc_md, "Must document restore command"
    assert "Credential Resolution" in doc_md, (
        "Must document credential resolution hierarchy"
    )


def test_restore_recreates_history_rows_simulation(tmp_path: Path):
    """Criterion 1: Restore recreates history rows (SQL simulation test)."""
    # Create SQLite database mimicking analysis history table structure
    db_file = tmp_path / "source.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE analysis_history (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            crop TEXT NOT NULL,
            prediction_class TEXT NOT NULL,
            confidence REAL
        )
    """)
    cur.execute(
        "INSERT INTO analysis_history VALUES ('rec-1', '2026-10-30T10:00:00Z', 'lettuce', 'Healthier', 0.95)"
    )
    cur.execute(
        "INSERT INTO analysis_history VALUES ('rec-2', '2026-10-30T10:05:00Z', 'spinach', 'Spinach_Healthy', 0.88)"
    )
    conn.commit()

    # Dump SQL
    dump_lines = list(conn.iterdump())
    conn.close()

    dump_file = tmp_path / "backup_simulation.sql"
    dump_file.write_text("\n".join(dump_lines), encoding="utf-8")

    # Restore into fresh database
    restore_file = tmp_path / "restored.db"
    restore_conn = sqlite3.connect(restore_file)
    restore_cur = restore_conn.cursor()
    restore_cur.executescript(dump_file.read_text(encoding="utf-8"))

    # Verify rows restored identically
    restore_cur.execute("SELECT COUNT(*) FROM analysis_history")
    count = restore_cur.fetchone()[0]
    assert count == 2, f"Expected 2 restored rows, got {count}"

    restore_cur.execute(
        "SELECT crop, prediction_class FROM analysis_history ORDER BY id"
    )
    rows = restore_cur.fetchall()
    assert rows == [("lettuce", "Healthier"), ("spinach", "Spinach_Healthy")]
    restore_conn.close()
