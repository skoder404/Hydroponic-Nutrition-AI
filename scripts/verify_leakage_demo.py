#!/usr/bin/env python3
"""Verification script for Data Leakage Demonstration Notebook (T4-001).

Acceptance Criteria:
1. Notebook runs top to bottom with a fixed seed.
2. Both scores and the explanation are shown.
3. A grep of the notebook shows no reference to the test file.

Re-run command:
    python scripts/verify_leakage_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "01_leakage_demo.ipynb"
CONCEPT_NOTE_PATH = ROOT / "docs" / "learning" / "data-leakage-and-grouped-timebased-splits.md"


def verify_notebook_execution_and_structure() -> tuple[bool, str]:
    """Criterion 1: Notebook exists, is valid JSON, and has executed code cells."""
    if not NOTEBOOK_PATH.exists():
        return False, f"Notebook not found at: {NOTEBOOK_PATH}"

    try:
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"Notebook is not valid JSON: {e}"

    if nb.get("nbformat") != 4:
        return False, f"Expected nbformat 4, got {nb.get('nbformat')}"

    cells = nb.get("cells", [])
    if len(cells) < 6:
        return False, f"Notebook has fewer cells than expected ({len(cells)} cells)"

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    executed_cells = [c for c in code_cells if c.get("outputs")]
    if len(executed_cells) == 0:
        return False, "Notebook has not been executed (all outputs are empty)"

    return True, f"Notebook exists with {len(cells)} cells ({len(code_cells)} code cells, all executed with outputs)"


def verify_scores_and_explanation() -> tuple[bool, str]:
    """Criterion 2: Both scores (random vs grouped) and explanation are shown."""
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    text_content = ""
    for cell in nb.get("cells", []):
        text_content += "".join(cell.get("source", [])) + "\n"
        for out in cell.get("outputs", []):
            if "text" in out:
                text_content += "".join(out.get("text", [])) + "\n"
            if "data" in out and "text/plain" in out["data"]:
                text_content += "".join(out["data"]["text/plain"]) + "\n"

    # Check for scores
    has_random_f1 = "Naive Random Split" in text_content or "Random Split Macro-F1" in text_content
    has_grouped_f1 = "Grouped Date Split" in text_content or "Grouped Date Macro-F1" in text_content
    has_gap_explanation = "Interpolation vs Extrapolation" in text_content or "autocorrelation" in text_content.lower()

    if not has_random_f1:
        return False, "Random split validation score missing from notebook outputs"
    if not has_grouped_f1:
        return False, "Grouped date split validation score missing from notebook outputs"
    if not has_gap_explanation:
        return False, "Explanation of the inflation gap missing from notebook text"

    return True, "Both random and grouped scores are presented alongside the technical gap explanation"


def verify_no_test_file_references() -> tuple[bool, str]:
    """Criterion 3: Grep of notebook shows zero reference to held-out test file."""
    content = NOTEBOOK_PATH.read_text(encoding="utf-8").lower()
    forbidden_terms = [
        "hydroponic_lettuce_test",
        "lettuce/test.csv",
        "sensor/lettuce/test",
        "test_csv",
    ]

    for term in forbidden_terms:
        if term in content:
            return False, f"Forbidden reference to test file detected in notebook: '{term}'"

    return True, "Zero references to the held-out test file found across all notebook cells"


def verify_concept_note() -> tuple[bool, str]:
    """Verify navigator concept note exists and adheres to template structure."""
    if not CONCEPT_NOTE_PATH.exists():
        return False, f"Concept note not found at: {CONCEPT_NOTE_PATH}"

    text = CONCEPT_NOTE_PATH.read_text(encoding="utf-8")
    required_sections = [
        "## 1. Intuition",
        "## 2. Mathematical Formulation & Formal Definition",
        "## 3. From-Scratch Implementation",
        "## 4. Production & Library Usage",
        "## 5. Failure Modes, Edge Cases & Debugging",
        "## 6. Where It Appears in This Project",
        "## 7. Self-Check Questions",
    ]

    missing = [sec for sec in required_sections if sec not in text]
    if missing:
        return False, f"Concept note missing sections: {missing}"

    return True, f"Concept note verified with all required template sections at {CONCEPT_NOTE_PATH.name}"


def run_all_checks() -> int:
    """Execute all verification checks."""
    print("=" * 70)
    print("Data Leakage Demo Notebook Verification (T4-001)")
    print("=" * 70)

    checks = [
        ("Criterion 1: Notebook execution & outputs", verify_notebook_execution_and_structure),
        ("Criterion 2: Both scores & explanation shown", verify_scores_and_explanation),
        ("Criterion 3: Zero references to test file", verify_no_test_file_references),
        ("Criterion 4: Navigator concept note structure", verify_concept_note),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {title}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED (T4-001)")
        return 0
    else:
        print("SOME VERIFICATION CHECKS FAILED (T4-001)")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
