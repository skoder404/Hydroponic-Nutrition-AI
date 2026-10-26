"""Unit tests for Data Leakage Demonstration Notebook (T4-001)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "01_leakage_demo.ipynb"
CONCEPT_NOTE_PATH = ROOT / "docs" / "learning" / "data-leakage-and-grouped-timebased-splits.md"


def test_notebook_exists_and_is_valid_format() -> None:
    """Verify 01_leakage_demo.ipynb exists and adheres to nbformat 4."""
    assert NOTEBOOK_PATH.is_file()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    assert len(nb.get("cells", [])) >= 6


def test_notebook_cells_contain_executed_outputs() -> None:
    """Verify code cells were executed and have output payloads."""
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) > 0
    for cell in code_cells:
        assert len(cell.get("outputs", [])) > 0, f"Unexecuted cell found: {cell.get('source')[:1]}"


def test_no_test_file_references_in_notebook() -> None:
    """Verify no references to held-out test file exist in the notebook."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8").lower()
    assert "hydroponic_lettuce_test" not in content
    assert "lettuce/test.csv" not in content
    assert "sensor/lettuce/test" not in content


def test_concept_note_exists_with_required_sections() -> None:
    """Verify navigator concept note exists with all 7 standard sections."""
    assert CONCEPT_NOTE_PATH.is_file()
    text = CONCEPT_NOTE_PATH.read_text(encoding="utf-8")
    for sec_num in range(1, 8):
        assert f"## {sec_num}." in text
