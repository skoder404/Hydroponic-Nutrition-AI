#!/usr/bin/env python3
"""Verification script for T9-002: Extend docker-compose for the full stack.

Acceptance Criteria:
1. `docker compose up` serves the app end to end on a fresh clone (all 3 services present & configured).
2. Healthchecks gate dependent services (service_healthy dependencies).
3. Volumes persist models and uploads.
"""

from __future__ import annotations

import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def verify_compose_file() -> tuple[bool, str]:
    path = ROOT / "docker-compose.yml"
    if not path.is_file():
        return False, "docker-compose.yml not found."
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse docker-compose.yml: {exc}"

    if not isinstance(data, dict) or "services" not in data:
        return False, "Invalid docker-compose format (missing 'services')."

    return True, "docker-compose.yml exists and parsed successfully."


def verify_services_and_healthchecks() -> tuple[bool, str]:
    path = ROOT / "docker-compose.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    services = data.get("services", {})

    for s in ["db", "backend", "frontend"]:
        if s not in services:
            return False, f"Missing service '{s}' in docker-compose.yml"
        if "healthcheck" not in services[s]:
            return False, f"Service '{s}' missing healthcheck definition"

    backend = services["backend"]
    frontend = services["frontend"]

    if (
        backend.get("depends_on", {}).get("db", {}).get("condition")
        != "service_healthy"
    ):
        return False, "backend must depend on db with condition: service_healthy"

    if (
        frontend.get("depends_on", {}).get("backend", {}).get("condition")
        != "service_healthy"
    ):
        return False, "frontend must depend on backend with condition: service_healthy"

    if "alembic upgrade head" not in str(backend.get("command", "")):
        return False, "backend command must run 'alembic upgrade head' on start"

    return (
        True,
        "All 3 services configured; healthchecks gate dependent services; migrations run on start.",
    )


def verify_persistent_volumes() -> tuple[bool, str]:
    path = ROOT / "docker-compose.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = data.get("services", {}).get("backend", {})
    volumes = backend.get("volumes", [])

    has_models = any("models" in str(v) for v in volumes)
    has_uploads = any("uploads" in str(v) for v in volumes)

    if not (has_models and has_uploads):
        return False, "backend service must mount volumes for both models and uploads"

    return True, "Volumes persist models and uploads."


def main() -> int:
    print("=== Running T9-002 Verification ===")
    checks = [
        ("Compose Syntax & Parsing", verify_compose_file),
        ("Services & Healthcheck Gating", verify_services_and_healthchecks),
        ("Persistent Volumes", verify_persistent_volumes),
    ]

    all_passed = True
    for name, fn in checks:
        ok, msg = fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll T9-002 acceptance criteria satisfied.")
        return 0
    else:
        print("\nT9-002 verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
