#!/usr/bin/env python3
"""Export and snapshot the FastAPI OpenAPI schema contract (T6-016).

Writes:
- docs/api/openapi.json (canonical reference contract)
- tests/fixtures/openapi_snapshot.json (test fixture snapshot)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import create_app


def export_openapi() -> None:
    print("Generating FastAPI OpenAPI contract...")
    app = create_app(configure_logging=False)
    openapi_schema = app.openapi()

    docs_target = PROJECT_ROOT / "docs" / "api" / "openapi.json"
    docs_target.parent.mkdir(parents=True, exist_ok=True)
    docs_target.write_text(json.dumps(openapi_schema, indent=2), encoding="utf-8")
    print(f"Exported canonical contract to: {docs_target}")

    fixture_target = PROJECT_ROOT / "tests" / "fixtures" / "openapi_snapshot.json"
    fixture_target.parent.mkdir(parents=True, exist_ok=True)
    fixture_target.write_text(json.dumps(openapi_schema, indent=2), encoding="utf-8")
    print(f"Exported snapshot test fixture to: {fixture_target}")

    paths = list(openapi_schema.get("paths", {}).keys())
    print(f"Total API paths documented ({len(paths)}):")
    for p in sorted(paths):
        print(f"  {p}")


if __name__ == "__main__":
    export_openapi()

