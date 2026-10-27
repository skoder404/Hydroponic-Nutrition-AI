"""Unit tests for Decision Tree vs Rules Analysis (T4-021 / PRD LR-5)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_tree_vs_rules_reports_exist():
    """Verify that both JSON and Markdown analysis reports for T4-021 exist and are populated."""
    json_path = ROOT / "reports" / "analysis" / "tree_vs_rules.json"
    md_path = ROOT / "reports" / "analysis" / "tree_vs_rules.md"

    assert json_path.is_file(), f"Missing {json_path}"
    assert md_path.is_file(), f"Missing {md_path}"
    assert json_path.stat().st_size > 1000
    assert md_path.stat().st_size > 1000


def test_all_three_crops_tabulated_beside_documented_bands():
    """Verify acceptance criterion: Learned thresholds are tabulated beside documented bands for all crops."""
    json_path = ROOT / "reports" / "analysis" / "tree_vs_rules.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    crops = data.get("crops", {})
    assert "lettuce" in crops, "Lettuce missing in analysis"
    assert "spinach" in crops, "Spinach missing in analysis"
    assert "tomato" in crops, "Tomato missing in analysis"

    for crop_name, analysis in crops.items():
        comparisons = analysis.get("comparisons", [])
        assert len(comparisons) > 0, f"No feature comparisons for {crop_name}"

        # Ensure at least one primary feature has verified threshold recovery
        recovered = [c for c in comparisons if c.get("recovery_status") == "Recovered"]
        assert len(recovered) > 0, (
            f"Expected recovered rule thresholds for {crop_name}, found none"
        )

    # Verify Lettuce specific recovery
    lettuce_comp = {c["feature"]: c for c in crops["lettuce"]["comparisons"]}
    assert lettuce_comp["ph"]["learned_tree_lower_split"] is not None
    assert lettuce_comp["ec"]["learned_tree_upper_split"] is not None

    # Verify Markdown contains tables for all three crops
    md_path = ROOT / "reports" / "analysis" / "tree_vs_rules.md"
    md_content = md_path.read_text(encoding="utf-8")
    assert "Lettuce Sensor Threshold Recovery" in md_content
    assert "Spinach Sensor Threshold Recovery" in md_content
    assert "Tomato Sensor Threshold Recovery" in md_content


def test_document_answers_lr5_in_plain_language():
    """Verify acceptance criterion: Document answers LR-5 from the PRD in plain language."""
    md_path = ROOT / "reports" / "analysis" / "tree_vs_rules.md"
    md_content = md_path.read_text(encoding="utf-8")

    assert "LR-5" in md_content
    assert "biological" in md_content.lower()
    assert "Plain-Language Explanation of PRD LR-5" in md_content
    assert "Sensor Calibration Drift" in md_content


def test_one_paragraph_conclusion_present():
    """Verify acceptance criterion: Both members can restate conclusion in one paragraph."""
    md_path = ROOT / "reports" / "analysis" / "tree_vs_rules.md"
    md_content = md_path.read_text(encoding="utf-8")

    assert (
        "The One-Paragraph Takeaway (Mastery Statement for Sushil & Yashwant):"
        in md_content
    )
    assert "reverse-engineering" in md_content or "reconstruct" in md_content


def test_concept_note_exists_and_complete():
    """Verify that concept note docs/learning/why-rulederived-labels-inflate-scores.md exists and is complete."""
    note_path = ROOT / "docs" / "learning" / "why-rulederived-labels-inflate-scores.md"
    assert note_path.is_file(), f"Missing concept note at {note_path}"

    content = note_path.read_text(encoding="utf-8")
    assert "LR-5" in content
    assert "## 1. Intuition" in content
    assert "## 2. Mathematical Formulation" in content
    assert "## 3. From-Scratch Implementation" in content
    assert "## 7. Self-Check Questions" in content
