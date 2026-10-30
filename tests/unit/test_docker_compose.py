"""Unit tests for full-stack docker-compose configuration (T9-002)."""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def test_docker_compose_file_exists_and_parses():
    """Verify docker-compose.yml exists and parses as valid YAML."""
    compose_path = ROOT / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml must exist at repository root"

    content = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "docker-compose.yml must parse into a mapping"
    assert "services" in data, "docker-compose.yml must define 'services'"


def test_services_configured():
    """Verify db, backend, and frontend services are configured."""
    compose_path = ROOT / "docker-compose.yml"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = data.get("services", {})

    assert "db" in services, "Must configure 'db' service"
    assert "backend" in services, "Must configure 'backend' service"
    assert "frontend" in services, "Must configure 'frontend' service"


def test_healthchecks_gate_dependencies():
    """Acceptance Criteria 2: Healthchecks gate dependent services."""
    compose_path = ROOT / "docker-compose.yml"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = data.get("services", {})

    db = services["db"]
    backend = services["backend"]
    frontend = services["frontend"]

    # All services must have healthcheck definitions
    assert "healthcheck" in db, "db service must define a healthcheck"
    assert "healthcheck" in backend, "backend service must define a healthcheck"
    assert "healthcheck" in frontend, "frontend service must define a healthcheck"

    # Backend must wait for db to be healthy
    assert "db" in backend.get("depends_on", {}), "backend must depend on db"
    assert backend["depends_on"]["db"].get("condition") == "service_healthy", (
        "backend must wait for db to be service_healthy"
    )

    # Frontend must wait for backend to be healthy
    assert "backend" in frontend.get("depends_on", {}), (
        "frontend must depend on backend"
    )
    assert frontend["depends_on"]["backend"].get("condition") == "service_healthy", (
        "frontend must wait for backend to be service_healthy"
    )


def test_volumes_persist_models_and_uploads():
    """Acceptance Criteria 3: Volumes persist models and uploads."""
    compose_path = ROOT / "docker-compose.yml"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    backend = data.get("services", {}).get("backend", {})
    volumes = backend.get("volumes", [])

    has_models = any("models" in str(v) for v in volumes)
    has_uploads = any("uploads" in str(v) for v in volumes)

    assert has_models, "backend must mount models volume"
    assert has_uploads, "backend must mount uploads volume"


def test_migrations_run_on_start():
    """Verify backend executes database migrations on start."""
    compose_path = ROOT / "docker-compose.yml"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    backend = data.get("services", {}).get("backend", {})
    command = backend.get("command", "")

    assert "alembic upgrade head" in str(command), (
        "backend command must run 'alembic upgrade head' on start"
    )
