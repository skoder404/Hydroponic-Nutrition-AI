"""Unit and snapshot contract tests for OpenAPI schema (T6-016).

Acceptance Criteria:
- T6-016 AC1: Snapshot is committed
- T6-016 AC2: TECH_RULES route list matches the implementation
- T6-016 AC3: Breaking changes fail the contract test later
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.main import create_app

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_OPENAPI = ROOT / "docs" / "api" / "openapi.json"
FIXTURE_SNAPSHOT = ROOT / "tests" / "fixtures" / "openapi_snapshot.json"
TECH_RULES_PATH = ROOT / "docs" / "TECH_RULES.md"


def test_openapi_files_committed_and_valid_json() -> None:
    """T6-016 AC1: Snapshot is committed and contains valid OpenAPI 3.x schema."""
    assert DOCS_OPENAPI.is_file(), f"Missing docs openapi file at {DOCS_OPENAPI}"
    assert FIXTURE_SNAPSHOT.is_file(), (
        f"Missing fixture openapi file at {FIXTURE_SNAPSHOT}"
    )

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
        assert ep in implemented_paths or f"/api/v1{ep}" in implemented_paths, (
            f"Canonical endpoint {ep} not implemented in OpenAPI routes"
        )


def test_contract_paths_and_operations_match_snapshot() -> None:
    """T6-016 AC3: Breaking changes (missing routes, altered operations) fail the test."""
    app = create_app(configure_logging=False)
    current_schema = app.openapi()
    snapshot = json.loads(FIXTURE_SNAPSHOT.read_text(encoding="utf-8"))

    current_paths = set(current_schema.get("paths", {}).keys())
    snapshot_paths = set(snapshot.get("paths", {}).keys())

    # Ensure no routes were dropped
    missing_paths = snapshot_paths - current_paths
    assert len(missing_paths) == 0, (
        f"Breaking change detected! Dropped routes: {missing_paths}"
    )

    # Ensure operations per path match
    for path, path_obj in snapshot.get("paths", {}).items():
        current_path_obj = current_schema.get("paths", {}).get(path, {})
        for method in path_obj:
            if method.startswith("x-"):
                continue
            assert method in current_path_obj, (
                f"Breaking change! Method {method.upper()} for path {path} was removed."
            )


# -----------------------------------------------------------------------------
# T8-008: Contract tests against the OpenAPI snapshot
# -----------------------------------------------------------------------------


def check_schema_compatibility(current_schema: dict, snapshot: dict) -> list[str]:
    """Inspect schema models, properties, and required fields for breaking changes."""
    breaking_changes: list[str] = []
    current_components = current_schema.get("components", {}).get("schemas", {})
    snapshot_components = snapshot.get("components", {}).get("schemas", {})

    for model_name, snap_model in snapshot_components.items():
        if model_name not in current_components:
            breaking_changes.append(
                f"Model '{model_name}' removed from components/schemas."
            )
            continue

        curr_model = current_components[model_name]

        # Check required fields
        snap_required = set(snap_model.get("required", []))
        curr_required = set(curr_model.get("required", []))
        removed_required = snap_required - curr_required
        if removed_required:
            breaking_changes.append(
                f"Model '{model_name}' required fields removed: {sorted(removed_required)}"
            )

        # Check properties: no property present in snapshot may be removed
        snap_props = snap_model.get("properties", {})
        curr_props = curr_model.get("properties", {})
        missing_props = set(snap_props.keys()) - set(curr_props.keys())
        if missing_props:
            breaking_changes.append(
                f"Model '{model_name}' properties dropped: {sorted(missing_props)}"
            )

    return breaking_changes


def test_contract_schemas_and_properties_match_snapshot() -> None:
    """T8-008: Ensure response models, properties, and required fields match snapshot."""
    app = create_app(configure_logging=False)
    current_schema = app.openapi()
    snapshot = json.loads(FIXTURE_SNAPSHOT.read_text(encoding="utf-8"))

    breaking_changes = check_schema_compatibility(current_schema, snapshot)
    assert len(breaking_changes) == 0, (
        "OpenAPI breaking schema changes detected! Run 'python scripts/export_openapi.py' "
        "and commit as an explicit snapshot update.\n" + "\n".join(breaking_changes)
    )


def test_intentional_field_removal_fails_contract_test() -> None:
    """T8-008 AC 1: Intentional field removal fails the test."""
    import copy

    app = create_app(configure_logging=False)
    current_schema = app.openapi()
    snapshot = json.loads(FIXTURE_SNAPSHOT.read_text(encoding="utf-8"))

    # Create mutated current schema missing a required property from PredictionResponse
    mutated_current = copy.deepcopy(current_schema)
    target_model = mutated_current["components"]["schemas"].get(
        "PredictionResponse", {}
    )
    assert "properties" in target_model
    assert "display_label" in target_model["properties"]

    # Intentionally remove 'display_label'
    del target_model["properties"]["display_label"]
    if "required" in target_model and "display_label" in target_model["required"]:
        target_model["required"].remove("display_label")

    # Verify that compatibility check detects the missing field and reports failure
    breaking = check_schema_compatibility(mutated_current, snapshot)
    assert len(breaking) > 0, (
        "Expected compatibility check to fail when field is removed"
    )
    assert any("display_label" in b for b in breaking), (
        f"Expected 'display_label' to be cited in breaking changes, got: {breaking}"
    )


def test_snapshot_update_requires_explicit_commit_workflow() -> None:
    """T8-008 AC 2: Snapshot update is an explicit commit."""
    export_script = ROOT / "scripts" / "export_openapi.py"
    assert export_script.is_file(), f"Missing export script at {export_script}"
    content = export_script.read_text(encoding="utf-8")
    assert "docs/api/openapi.json" in content
    assert "tests/fixtures/openapi_snapshot.json" in content


def test_ci_workflow_runs_contract_tests() -> None:
    """T8-008 AC 3: Runs in CI (unit test suite in CI runs test_openapi_contract)."""
    ci_workflow = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_workflow.is_file(), f"Missing CI workflow at {ci_workflow}"
    content = ci_workflow.read_text(encoding="utf-8")
    assert "pytest tests/unit/" in content, "CI workflow must run tests/unit/"
