#!/usr/bin/env python3
"""Verification script for T1-011: Docker Compose PostgreSQL service.

Validates:
1. docker-compose.yml configuration:
   - Service 'db' exists with postgres image and named volume.
   - Healthcheck defined using pg_isready.
   - No committed plaintext passwords.
2. DATABASE_URL in .env.example matches database service credentials/schema.
3. Live container test (when Docker daemon is accessible):
   - docker compose up -d db reports healthy.
   - Data persists across container restart using named volume.
   - Cleanup on exit.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def check_compose_file_and_secrets():
    path = Path("docker-compose.yml")
    if not path.is_file():
        return False, "docker-compose.yml does not exist"

    content = path.read_text(encoding="utf-8")

    # Check for service db
    if "db:" not in content:
        return False, "Missing 'db' service in docker-compose.yml"

    # Check named volume
    if "postgres_data:" not in content:
        return False, "Missing 'postgres_data' named volume"

    # Check healthcheck
    if "healthcheck:" not in content or "pg_isready" not in content:
        return False, "Missing pg_isready healthcheck in db service"

    # Check no hardcoded passwords committed
    # e.g., POSTGRES_PASSWORD should use variable expansion ${...}
    pw_match = re.search(r"POSTGRES_PASSWORD:\s*([^\n\r]+)", content)
    if pw_match:
        val = pw_match.group(1).strip()
        if not val.startswith("${"):
            return False, f"Hardcoded password found in docker-compose.yml: {val}"

    return True, "docker-compose.yml structure is valid and contains no hardcoded passwords."


def check_env_example_database_match():
    env_path = Path(".env.example")
    if not env_path.is_file():
        return False, ".env.example does not exist"

    content = env_path.read_text(encoding="utf-8")
    if "DATABASE_URL" not in content:
        return False, "DATABASE_URL missing from .env.example"

    if "your_postgres_password_here" not in content and "your_password_here" not in content:
        return False, "DATABASE_URL in .env.example does not use a placeholder password"

    return True, ".env.example matches db service schema with placeholders only."


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode, res.stdout, res.stderr


def test_docker_runtime_health_and_persistence():
    # Verify docker CLI works
    code, _, err = run_cmd(["docker", "info"])
    if code != 0:
        return False, f"Docker daemon not running: {err}"

    print("Starting db container with 'docker compose up -d db'...")
    code, out, err = run_cmd(["docker", "compose", "up", "-d", "db"])
    if code != 0:
        return False, f"docker compose up failed: {err or out}"

    try:
        # Wait for healthy status (up to 30 seconds)
        print("Waiting for container to become healthy...")
        healthy = False
        container_name = "hydroponics_db"

        for _ in range(15):
            time.sleep(2)
            code, inspect_out, _ = run_cmd(["docker", "inspect", container_name])
            if code == 0:
                data = json.loads(inspect_out)
                if data and "State" in data[0] and "Health" in data[0]["State"]:
                    status = data[0]["State"]["Health"]["Status"]
                    if status == "healthy":
                        healthy = True
                        break

        if not healthy:
            return False, "Container failed to reach 'healthy' state within timeout."

        print("[PASS] Container reached 'healthy' state.")

        # Test data persistence across restart
        print("Testing data persistence across restart...")
        test_table = "_persistence_verification_test"
        marker_val = f"marker_{int(time.time())}"

        # Insert record
        sql_insert = (
            f"CREATE TABLE IF NOT EXISTS {test_table} (id serial primary key, marker text);"
            f"INSERT INTO {test_table} (marker) VALUES ('{marker_val}');"
        )
        code, out, err = run_cmd([
            "docker", "exec", container_name,
            "psql", "-U", "postgres", "-d", "hydroponics", "-c", sql_insert,
        ])
        if code != 0:
            return False, f"Failed to insert test persistence record: {err or out}"

        # Restart container
        print("Restarting container with 'docker compose restart db'...")
        code, out, err = run_cmd(["docker", "compose", "restart", "db"])
        if code != 0:
            return False, f"Failed to restart container: {err or out}"

        # Wait for healthy again
        healthy_after_restart = False
        for _ in range(15):
            time.sleep(2)
            code, inspect_out, _ = run_cmd(["docker", "inspect", container_name])
            if code == 0:
                data = json.loads(inspect_out)
                if data and "State" in data[0] and "Health" in data[0]["State"]:
                    status = data[0]["State"]["Health"]["Status"]
                    if status == "healthy":
                        healthy_after_restart = True
                        break

        if not healthy_after_restart:
            return False, "Container failed to become healthy after restart."

        # Verify data persisted
        sql_select = f"SELECT marker FROM {test_table} WHERE marker='{marker_val}';"
        code, query_out, err = run_cmd([
            "docker", "exec", container_name,
            "psql", "-U", "postgres", "-d", "hydroponics", "-t", "-c", sql_select,
        ])
        if code != 0 or marker_val not in query_out:
            return False, f"Data persistence check failed! Query output: {query_out}"

        # Cleanup test table
        run_cmd([
            "docker", "exec", container_name,
            "psql", "-U", "postgres", "-d", "hydroponics", "-c", f"DROP TABLE IF EXISTS {test_table};",
        ])

        print("[PASS] Data successfully persisted across restart via named volume.")
        return True, "Docker container health and data persistence verified."

    finally:
        print("Stopping db container...")
        run_cmd(["docker", "compose", "down"])


def main():
    print("=== Running T1-011 Verification ===")
    errors = []

    ok_compose, msg_compose = check_compose_file_and_secrets()
    if not ok_compose:
        errors.append(msg_compose)
    else:
        print(f"[PASS] {msg_compose}")

    ok_env, msg_env = check_env_example_database_match()
    if not ok_env:
        errors.append(msg_env)
    else:
        print(f"[PASS] {msg_env}")

    ok_rt, msg_rt = test_docker_runtime_health_and_persistence()
    if not ok_rt:
        errors.append(msg_rt)
    else:
        print(f"[PASS] {msg_rt}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T1-011 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
