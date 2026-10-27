"""Unit tests for lettuce grouped 5-fold cross-validation by calendar date (T4-022)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_lettuce_grouped_cv_reports_exist():
    """Verify that both JSON and Markdown reports for T4-022 exist and are populated."""
    json_path = ROOT / "reports" / "lettuce" / "lettuce_grouped_cv_report.json"
    md_path = ROOT / "reports" / "lettuce" / "lettuce_grouped_cv_report.md"

    assert json_path.is_file(), f"Missing {json_path}"
    assert md_path.is_file(), f"Missing {md_path}"
    assert json_path.stat().st_size > 1000
    assert md_path.stat().st_size > 1000


def test_fold_composition_no_date_overlap():
    """Verify acceptance criterion: Fold composition lists dates per fold with no overlap."""
    json_path = ROOT / "reports" / "lettuce" / "lettuce_grouped_cv_report.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    folds = data.get("fold_compositions", [])
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    all_dates = []
    seen_dates = set()
    for f in folds:
        dates = f.get("val_dates", [])
        assert len(dates) > 0, f"Fold {f['fold']} has no dates"
        for d in dates:
            assert d not in seen_dates, (
                f"Date {d} appears in multiple validation folds!"
            )
            seen_dates.add(d)
            all_dates.append(d)

    assert len(all_dates) == data["total_calendar_days"]
    assert len(seen_dates) == 69


def test_report_shows_mean_std_and_compares_with_readme():
    """Verify acceptance criterion: Report shows mean and std and compares with README figure."""
    json_path = ROOT / "reports" / "lettuce" / "lettuce_grouped_cv_report.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    models = data["model_results"]
    assert "random_forest" in models
    rf = models["random_forest"]

    assert "mean_macro_f1" in rf
    assert "std_macro_f1" in rf
    assert 0.5 <= rf["mean_macro_f1"] <= 1.0
    assert rf["std_macro_f1"] >= 0.0

    readme = data["readme_baseline"]
    assert readme["reported_macro_f1_mean"] == 0.837
    assert readme["reported_macro_f1_std"] == 0.075

    md_path = ROOT / "reports" / "lettuce" / "lettuce_grouped_cv_report.md"
    md_content = md_path.read_text(encoding="utf-8")
    assert "0.837" in md_content
    assert "0.075" in md_content
    assert "one favourable split" in md_content


def test_result_is_not_described_as_field_accuracy():
    """Verify acceptance criterion: Result is explicitly not described as field accuracy."""
    md_path = ROOT / "reports" / "lettuce" / "lettuce_grouped_cv_report.md"
    md_content = md_path.read_text(encoding="utf-8")

    assert (
        "Not field accuracy" in md_content or "not field accuracy" in md_content.lower()
    )
    assert "biological plant health" in md_content
    assert "one favourable split" in md_content
