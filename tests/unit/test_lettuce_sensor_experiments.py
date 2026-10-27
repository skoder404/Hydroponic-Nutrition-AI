"""Unit tests for Lettuce Sensor Experiments (T4-017).

Verifies Acceptance Criteria:
- Report covers all five families with required Section 18.2 metrics.
- No test metrics appear in any report or metadata.
- Limitations block states rule-derived labels and observational scope.
- Comparison against source README baseline is included and documented.
"""

from __future__ import annotations

import json
from pathlib import Path


from scripts.run_lettuce_experiments import (
    MODEL_FAMILIES,
    verify_lettuce_split_integrity,
)

ROOT = Path(__file__).resolve().parent.parent.parent
LETTUCE_DATA_DIR = ROOT / "processed_data" / "sensor" / "lettuce"
REPORTS_DIR = ROOT / "reports" / "lettuce"
EXP_JSON_PATH = REPORTS_DIR / "lettuce_sensor_experiments.json"
EXP_MD_PATH = REPORTS_DIR / "lettuce_sensor_experiments.md"


def test_report_covers_all_five_families_with_required_metrics() -> None:
    """Acceptance Criterion 1: Report covers all five families with required metrics."""
    assert EXP_MD_PATH.is_file(), f"Missing report at {EXP_MD_PATH}"
    assert EXP_JSON_PATH.is_file(), f"Missing JSON report at {EXP_JSON_PATH}"

    md_content = EXP_MD_PATH.read_text(encoding="utf-8")
    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    reported_models = {m["model_type"] for m in json_data["models"]}
    assert reported_models == set(MODEL_FAMILIES), (
        f"Missing model families: {set(MODEL_FAMILIES) - reported_models}"
    )

    for m in json_data["models"]:
        # Verify required metrics per Section 18.2
        assert "val_macro_f1" in m and m["val_macro_f1"] > 0.0
        assert "val_accuracy" in m and m["val_accuracy"] > 0.0
        assert "val_macro_precision" in m
        assert "val_macro_recall" in m
        assert "cv_macro_f1" in m
        assert "cv_std_macro_f1" in m
        assert "best_hyperparameters" in m

        # Model name appears in Markdown table
        assert f"**{m['model_type']}**" in md_content or m["model_type"] in md_content


def test_no_test_metrics_appear() -> None:
    """Acceptance Criterion 2: No test metrics appear in comparison report or JSON."""
    assert EXP_JSON_PATH.is_file()
    assert EXP_MD_PATH.is_file()

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_text = f.read()

    md_text = EXP_MD_PATH.read_text(encoding="utf-8")

    # Assert that no test metric keys appear
    forbidden_keys = [
        "test_macro_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_score",
    ]
    for key in forbidden_keys:
        assert key not in json_text.lower(), (
            f"Forbidden key '{key}' found in JSON report!"
        )
        assert key not in md_text.lower(), (
            f"Forbidden key '{key}' found in Markdown report!"
        )


def test_limitations_block_states_rule_derived_labels() -> None:
    """Acceptance Criterion 3: Limitations block states rule-derived labels."""
    assert EXP_JSON_PATH.is_file()
    assert EXP_MD_PATH.is_file()

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    md_text = EXP_MD_PATH.read_text(encoding="utf-8")

    limitations = json_data.get("limitations", [])
    assert len(limitations) > 0, "Limitations list is empty!"

    # Must discuss rule-derived labels
    rule_mentioned = any("rule-derived" in lim.lower() for lim in limitations)
    assert rule_mentioned, "Limitations must state rule-derived labels!"

    assert "rule-derived" in md_text.lower()
    assert "action bands" in md_text.lower()


def test_readme_baseline_comparison_included() -> None:
    """Report compares numbers against source README (0.837 +/- 0.075 and 0.994/0.986)."""
    assert EXP_MD_PATH.is_file()
    md_text = EXP_MD_PATH.read_text(encoding="utf-8")

    assert "0.837" in md_text, (
        "Source README grouped CV score 0.837 missing from report"
    )
    assert "0.986" in md_text or "0.994" in md_text, (
        "Source README single-split score 0.986/0.994 missing from report"
    )
    assert "favourable" in md_text.lower() or "favorable" in md_text.lower(), (
        "Explanation of single-split as favorable split missing from report"
    )


def test_group_separation_and_split_integrity() -> None:
    """Verify calendar-day separation between train, val, and test partitions."""
    split_info = verify_lettuce_split_integrity(LETTUCE_DATA_DIR)
    assert split_info["group_separation_verified"] is True
    assert split_info["overlap_days"] == 0
    assert split_info["train_days"] == 41
    assert split_info["val_days"] == 12
    assert split_info["test_days"] == 16
