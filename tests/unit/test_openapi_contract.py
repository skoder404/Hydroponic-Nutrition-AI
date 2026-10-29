"""Unit and snapshot contract tests for OpenAPI schema (T6-016).

Acceptance Criteria:
- T6-016 AC1: Snapshot is committed
- T6-016 AC2: TECH_RULES route list matches the implementation
- T6-016 AC3: Breaking changes fail the contract test later
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.main import create_app

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_OPENAPI = ROOT / "docs" / "api" / "openapi.json"
FIXTURE_SNAPSHOT = ROOT / "tests" / "fixtures" / "openapi_snapshot.json"
TECH_RULES_PATH = ROOT / "docs" / "TECH_RULES.md"


def test_openapi_files_committed_and_valid_json() -> None:
    """T6-016 AC1: Snapshot is committed and contains valid OpenAPI 3.x schema."""
    assert DOCS_OPENAPI.is_file(), f"Missing docs openapi file at {DOCS_OPENAPI}"
    assert FIXTURE_SNAPSHOT.is_file(), f"Missing fixture openapi file at {FIXTURE_SNAPSHOT}"

    docs_json = json.loads(DOCS_OPENAPI.read_text(encoding="utf-8"))
    fixture_json = json.loads(FIXTURE_SNAPSHOT.read_text(encoding="utf-8"))

    assert docs_json.get("openapi", "").startswith("3.")
    assert fixture_json.get("openapi", "").startswith("3.")
    assert len(docs_json.get("paths", {})) >= 10
    assert len(fixture_json.get("paths", {})) >= 10


def test_tech_rules_route_list_matches_implementation() -> None:
    """T6-016 AC2: TECH_RULES route list matches the implemented routes."""
    assert TECH_RULES_PATH.is_file()
    tech_rules_text = TECH_RULES_PATH.read_text(encoding="utf-8")

    app = create_app(configure_logging=False)
    schema = app.openapi()
    implemented_paths = set(schema.get("paths", {}).keys())

    # All primary canonical endpoints must be documented in TECH_RULES
    canonical_endpoints = [
        "/health",
        "/models",
        "/predict/sensor",
        "/predict/image",
        "/analyze",
        "/history",
        "/device/status",
        "/device/sensors/latest",
        "/device/camera/frame",
    ]

    for ep in canonical_endpoints:
        assert ep in tech_rules_text, f"Endpoint {ep} missing from docs/TECH_RULES.md"
        assert (
            ep in implemented_paths or f"/api/v1{ep}" in implemented_paths
        ), f"Canonical endpoint {ep} not implemented in OpenAPI routes"


def test_contract_paths_and_operations_match_snapshot() -> None:
    """T6-016 AC3: Breaking changes (missing routes, altered operations) fail the test."""
    app = create_app(configure_logging=False)
    current_schema = app.openapi()
    snapshot = json.loads(FIXTURE_SNAPSHOT.read_text(encoding="utf-8"))

    current_paths = set(current_schema.get("paths", {}).keys())
    snapshot_paths = set(snapshot.get("paths", {}).keys())

    # Ensure no routes were dropped
    missing_paths = snapshot_paths - current_paths
    assert len(missing_paths) == 0, f"Breaking change detected! Dropped routes: {missing_paths}"

    # Ensure operations per path match
    for path, path_obj in snapshot.get("paths", {}).items():
        current_path_obj = current_schema.get("paths", {}).get(path, {})
        for method in path_obj.keys():
            if method.startswith("x-"):
                continue
            assert method in current_path_obj, (
                f"Breaking change! Method {method.upper()} for path {path} was removed."
            )

