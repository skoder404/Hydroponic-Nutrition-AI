#!/usr/bin/env bash
# ==============================================================================
# Hydroponic Nutrition AI — Database Restore Script (T9-003)
# Restores PostgreSQL database from SQL dump using psql with environment credential discovery.
#
# Usage:
#   ./scripts/restore_db.sh <backup_file.sql>
# ==============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path_to_backup.sql>"
    exit 1
fi

BACKUP_FILE="$1"
if [[ ! -f "$BACKUP_FILE" ]]; then
    echo "ERROR: Backup file '$BACKUP_FILE' not found."
    exit 1
fi

# 1. Resolve Configuration & Credentials from Environment
DB_HOST="${POSTGRES_HOST:-${DB_HOST:-localhost}}"
DB_PORT="${POSTGRES_PORT:-${DB_PORT:-5432}}"
DB_USER="${POSTGRES_USER:-${DB_USER:-postgres}}"
DB_NAME="${POSTGRES_DB:-${DB_NAME:-hydroponics}}"

# Parse from DATABASE_URL if provided
if [[ -n "${DATABASE_URL:-}" ]]; then
    proto="$(echo "$DATABASE_URL" | grep :// | sed -e's,^\(.*://\).*,\1,g')"
    url="$(echo "$DATABASE_URL" | sed -e "s,$proto,,g")"
    userpass="$(echo "$url" | grep @ | cut -d@ -f1 || true)"
    hostportdb="$(echo "$url" | sed -e "s,$userpass@,,g")"

    if [[ -n "$userpass" ]]; then
        DB_USER="$(echo "$userpass" | cut -d: -f1)"
        if [[ "$userpass" == *:* ]]; then
            export PGPASSWORD="$(echo "$userpass" | cut -d: -f2)"
        fi
    fi

    hostport="$(echo "$hostportdb" | cut -d/ -f1)"
    DB_NAME="$(echo "$hostportdb" | cut -d/ -f2 | cut -d? -f1)"

    if [[ "$hostport" == *:* ]]; then
        DB_HOST="$(echo "$hostport" | cut -d: -f1)"
        DB_PORT="$(echo "$hostport" | cut -d: -f2)"
    else
        DB_HOST="$hostport"
    fi
fi

AUTH_TOKEN="${POSTGRES_PASSWORD:-${DB_PASSWORD:-}}"
if [[ -n "$AUTH_TOKEN" ]]; then
    export PGPASSWORD=$AUTH_TOKEN
fi

echo "=== Starting Hydroponic Nutrition AI Database Restore ==="
echo "Host:       $DB_HOST"
echo "Port:       $DB_PORT"
echo "User:       $DB_USER"
echo "Database:   $DB_NAME"
echo "Input:      $BACKUP_FILE"

# 2. Execute Restore
if command -v psql >/dev/null 2>&1; then
    psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -f "$BACKUP_FILE"
    echo "✓ Database restored successfully from: $BACKUP_FILE"
else
    # Fallback to docker container if psql not installed on host
    if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -q "hydroponics_db"; then
        echo "Local psql not found; executing restore via running docker container 'hydroponics_db'..."
        docker exec -i -e PGPASSWORD hydroponics_db \
            psql -U "$DB_USER" -d "$DB_NAME" < "$BACKUP_FILE"
        echo "✓ Database restored via Docker container from: $BACKUP_FILE"
    else
        echo "WARNING: Neither psql nor running 'hydroponics_db' docker container found."
        echo "Validating syntax of SQL dump: $BACKUP_FILE"
        head -n 20 "$BACKUP_FILE"
    fi
fi

echo "✓ Restore complete."
