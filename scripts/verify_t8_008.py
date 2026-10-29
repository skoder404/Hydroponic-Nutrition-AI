"""Verification script for T8-008: Contract tests against the OpenAPI snapshot.

Acceptance Criteria:
- Intentional field removal fails the test
- Snapshot update is an explicit commit
- Runs in CI
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import create_app  # noqa: E402
from tests.unit.test_openapi_contract import (  # noqa: E402
    FIXTURE_SNAPSHOT,
    check_schema_compatibility,
)


def main() -> int:
    print("Verifying T8-008: OpenAPI contract tests against snapshot...")

    app = create_app(configure_logging=False)
    current_schema = app.openapi()
    snapshot = json.loads(FIXTURE_SNAPSHOT.read_text(encoding="utf-8"))

    # 1. Base validation: current matches snapshot
    breaking = check_schema_compatibility(current_schema, snapshot)
    assert len(breaking) == 0, f"Unexpected breaking changes: {breaking}"
    print("  [PASS] Current schema is 100% compatible with committed snapshot")

    # 2. AC 1: Intentional field removal fails the test
    mutated = copy.deepcopy(current_schema)
    pred_response = mutated["components"]["schemas"]["PredictionResponse"]
    del pred_response["properties"]["display_label"]
    if "display_label" in pred_response.get("required", []):
        pred_response["required"].remove("display_label")

    simulated_breaking = check_schema_compatibility(mutated, snapshot)
    assert len(simulated_breaking) > 0, (
        "Expected intentional removal to produce breaking change"
    )
    assert any("display_label" in b for b in simulated_breaking)
    print("  [PASS] AC 1: Intentional field removal fails contract test")

    # 3. AC 2: Snapshot update is an explicit commit
    export_script = ROOT / "scripts" / "export_openapi.py"
    assert export_script.is_file(), "scripts/export_openapi.py must exist"
    export_content = export_script.read_text(encoding="utf-8")
    assert "docs/api/openapi.json" in export_content
    assert "tests/fixtures/openapi_snapshot.json" in export_content
    print(
        "  [PASS] AC 2: Snapshot update is an explicit commit using scripts/export_openapi.py"
    )

    # 4. AC 3: Runs in CI
    ci_file = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_file.is_file(), ".github/workflows/ci.yml must exist"
    ci_content = ci_file.read_text(encoding="utf-8")
    assert "pytest tests/unit/" in ci_content, "CI must execute unit test suite"
    print("  [PASS] AC 3: Contract tests run in CI workflow")

    print("ALL ACs for T8-008 PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
