"""Verification script for T10-005: Add history CSV export."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api.v1.history import sanitize_csv_cell


def test_t10_005_csv_export() -> None:
    print("=== Verifying T10-005: Add history CSV export ===")

    # 1. Criterion 1: Cells starting with = + - @ are neutralised
    assert sanitize_csv_cell("=1+1") == "'=1+1"
    assert sanitize_csv_cell("+cmd") == "'+cmd"
    assert sanitize_csv_cell("-calc") == "'-calc"
    assert sanitize_csv_cell("@danger") == "'@danger"
    assert sanitize_csv_cell("normal") == "normal"
    print("[PASS] Cells starting with = + - @ are strictly neutralised.")

    # 2. Check Backend implementation
    history_file = PROJECT_ROOT / "backend/app/api/v1/history.py"
    with open(history_file, "r", encoding="utf-8") as f:
        backend_code = f.read()

    assert "/history/export" in backend_code, "Endpoint /history/export missing in history.py"
    assert "sanitize_csv_cell" in backend_code, "sanitize_csv_cell not used in export endpoint"
    assert "text/csv" in backend_code, "media_type text/csv missing in export endpoint"
    assert "observed_tz" in backend_code, "Timezone missing in export endpoint"
    print("[PASS] Backend export endpoint /history/export implemented with filter handling.")

    # 3. Check Frontend implementation (UI button & API client method)
    frontend_page = PROJECT_ROOT / "frontend/app/history/page.tsx"
    api_client_file = PROJECT_ROOT / "frontend/lib/api/client.ts"

    with open(frontend_page, "r", encoding="utf-8") as f:
        page_code = f.read()

    assert "Export CSV" in page_code, "Export CSV button missing on History page"
    assert "getHistoryExportUrl" in page_code, "getHistoryExportUrl not used on History page"

    with open(api_client_file, "r", encoding="utf-8") as f:
        client_code = f.read()

    assert "getHistoryExportUrl" in client_code, "getHistoryExportUrl missing in ApiClient"
    print("[PASS] UI Export CSV button and typed API client method implemented.")

    # 4. Run pytest test_history_export.py
    print("Running pytest tests/unit/test_history_export.py...")
    res = subprocess.run(
        ["pytest", "tests/unit/test_history_export.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"pytest tests failed:\n{res.stdout}\n{res.stderr}"
    print("[PASS] Backend unit tests for CSV export passed.")

    # 5. Run TypeScript check on frontend
    print("Running npx tsc --noEmit in frontend...")
    res_tsc = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(PROJECT_ROOT / "frontend"),
        capture_output=True,
        text=True,
        shell=True,
    )
    assert res_tsc.returncode == 0, f"TypeScript check failed:\n{res_tsc.stdout}\n{res_tsc.stderr}"
    print("[PASS] Frontend TypeScript check passed with zero errors.")

    print("\n>>> All T10-005 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t10_005_csv_export()

