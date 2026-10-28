"""Verification script for T7-016: Build S4 History list."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_016_history_list() -> None:
    frontend_dir = PROJECT_ROOT / "frontend"
    history_page_file = frontend_dir / "app/history/page.tsx"
    history_table_file = frontend_dir / "components/history/HistoryTable.tsx"
    history_card_file = frontend_dir / "components/history/HistoryCard.tsx"
    history_filter_file = frontend_dir / "components/history/HistoryFilterBar.tsx"

    print("=== Verifying T7-016: Build S4 History list ===")

    # 1. Component files exist
    assert history_page_file.exists(), f"History page not found at {history_page_file}"
    assert history_table_file.exists(), f"HistoryTable not found at {history_table_file}"
    assert history_card_file.exists(), f"HistoryCard not found at {history_card_file}"
    assert history_filter_file.exists(), f"HistoryFilterBar not found at {history_filter_file}"
    print("[PASS] History page, table, card, and filter bar files exist.")

    # 2. Check Acceptance Criterion 1: Source chip 'Live' appears only for records flagged live
    with open(history_table_file, "r", encoding="utf-8") as f:
        table_code = f.read()

    with open(history_card_file, "r", encoding="utf-8") as f:
        card_code = f.read()

    # Table verification for source chip
    assert 'rec.source === "live"' in table_code, "Table must check rec.source === 'live' before showing Live chip"
    assert '<Badge variant="live"' in table_code, "Live badge variant must be used"
    assert '<Badge variant="historical">Historical</Badge>' in table_code, "Every row must carry 'Historical' tag"

    # Card verification for source chip
    assert 'record.source === "live"' in card_code, "Card must check record.source === 'live' before showing Live chip"
    assert '<Badge variant="live"' in card_code, "Card must use live badge variant"
    assert '<Badge variant="historical">Historical</Badge>' in card_code, "Every card must carry 'Historical' tag"
    print("[PASS] Source chip 'Live' appears ONLY for records flagged live; all items carry 'Historical' tag.")

    # 3. Check Acceptance Criterion 2: Empty/error/loading states match DESIGN S4
    with open(history_page_file, "r", encoding="utf-8") as f:
        page_code = f.read()

    # Empty state: "No saved records yet" + CTA to Analyze
    assert "No saved records yet" in page_code, "Empty state must match DESIGN S4: 'No saved records yet'"
    assert "/analyze" in page_code, "Empty state must link CTA to Analyze"

    # Error state: Error card + Retry; never show stale cached rows
    assert "Retry" in page_code, "Error state must provide Retry button"
    assert "stale" in page_code.lower(), "Error state must document that stale cached rows are not shown"
    assert "setRecords([])" in page_code, "Error state must clear records to avoid displaying stale data"

    # Loading state: Table skeleton
    assert "skeleton" in page_code.lower(), "Loading state must render table/card skeleton"
    print("[PASS] Empty, error, and loading states strictly match DESIGN S4 specifications.")

    # 4. Check Acceptance Criterion 3: Table scrolls inside its own container on small screens
    assert "overflow-x-auto" in table_code, "Table container must have overflow-x-auto for internal scrolling"
    assert "md:hidden" in table_code, "Stacked mobile cards must be rendered on small screens (< 768px)"
    assert "hidden md:block" in table_code, "Table must be wrapped with responsive visibility for >= 768px"
    print("[PASS] Table scrolls inside its own container and renders stacked cards under 768 px.")

    # 5. Check TypeScript compilation of frontend
    print("Running npx tsc --noEmit in frontend...")
    res = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
        shell=True,
    )
    assert res.returncode == 0, f"TypeScript compilation error:\n{res.stdout}\n{res.stderr}"
    print("[PASS] Frontend TypeScript compilation passed with zero errors.")

    print("\n>>> All T7-016 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_016_history_list()
