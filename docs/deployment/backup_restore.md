# Database Backup & Restore Runbook (T9-003)

This runbook defines the operational procedures for backing up and restoring the Hydroponic Nutrition AI PostgreSQL database, including environment credentials discovery, retention automation, and restore validation.

---

## 1. Credential Resolution & Environment Hierarchy

The backup and restore scripts ([`scripts/backup_db.sh`](../../scripts/backup_db.sh) and [`scripts/restore_db.sh`](../../scripts/restore_db.sh)) read database connection parameters dynamically from environment variables, avoiding hard-coded credentials:

| Variable | Description | Default Value |
|---|---|---|
| `DATABASE_URL` | Complete connection string (e.g., `postgresql://user:pass@host:5432/hydroponics`) | Extracted if present |
| `POSTGRES_HOST` / `DB_HOST` | Database server hostname | `localhost` |
| `POSTGRES_PORT` / `DB_PORT` | Database server port | `5432` |
| `POSTGRES_USER` / `DB_USER` | PostgreSQL user | `postgres` |
| `POSTGRES_PASSWORD` / `DB_PASSWORD` | PostgreSQL password | None (passed via `PGPASSWORD`) |
| `POSTGRES_DB` / `DB_NAME` | Target database name | `hydroponics` |
| `BACKUP_DIR` | Output destination directory for dumps | `./backups` |
| `RETENTION_DAYS` | Number of days to retain backup files | `7` |

---

## 2. Backup Procedure

### 2.1 Manual Execution
To create a timestamped database dump:

```bash
# Using environment variables from .env
export $(grep -v '^#' .env | xargs)
./scripts/backup_db.sh ./backups
```

The script produces a clean SQL dump named:
`backups/hydroponics_backup_YYYYMMDD_HHMMSSZ.sql`

Features:
- Includes `--clean` and `--if-exists` flags so the restore script can overwrite existing schema objects cleanly.
- Omits ownership flags (`--no-owner`, `--no-privileges`) ensuring portability across different database user accounts.
- Automatically purges backups older than `RETENTION_DAYS` (default: 7 days).

### 2.2 Automated Scheduled Backups (Cron)
On production and staging Linux hosts, schedule daily backups at 02:00 AM UTC:

```cron
0 2 * * * cd /opt/hydroponics && ./scripts/backup_db.sh /var/backups/hydroponics >> /var/log/hydroponics_backup.log 2>&1
```

---

## 3. Restore Procedure

### 3.1 Restoring from Backup
To restore a database from an existing SQL dump:

```bash
# Pass the path to the backup file
./scripts/restore_db.sh ./backups/hydroponics_backup_20261030_120000Z.sql
```

The restore script:
1. Connects to PostgreSQL using `psql` (or falls back to the running Docker container `hydroponics_db`).
2. Drops and recreates the necessary tables and indexes.
3. Restores all table data, including analysis history rows, sensor readings, and model registry pointer records.

### 3.2 Post-Restore Integrity Verification
After restoring, verify that history rows and records are accessible:

```bash
python -c "
from backend.app.database.session import get_db_session_factory
from backend.app.database.models import AnalysisRecord
with get_db_session_factory()() as db:
    count = db.query(AnalysisRecord).count()
    print(f'Restored history records count: {count}')
"
```
Or execute the automated test suite:
```bash
pytest tests/unit/test_backup_restore_scripts.py
```
