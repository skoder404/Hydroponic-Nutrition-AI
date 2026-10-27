#!/usr/bin/env python3
"""Verification script for model selection by validation macro-F1 (T4-023).

Acceptance Criteria:
1. Script fails if test metrics are supplied.
2. Selection JSON lists runners-up.
3. Per-class weaknesses are noted.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

from scripts.select_model import (
    run_selection,
    validate_report_is_validation_only,
)


def verify_criterion_1_fails_on_test_metrics() -> None:
    """Criterion 1: Script fails if test metrics are supplied."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Report explicitly evaluating test split
        test_split_report = {
            "crop": "lettuce",
            "model_name": "lettuce_rf_test",
            "split_evaluated": "test",
            "metrics": {"macro_f1": 0.85},
        }
        f1 = tmp_path / "model_test.json"
        with open(f1, "w", encoding="utf-8") as f:
            json.dump(test_split_report, f)

        failed_split = False
        try:
            validate_report_is_validation_only(test_split_report, f1)
        except ValueError as err:
            failed_split = True
            assert "Criterion 1 Violation" in str(err)
        assert failed_split, "Expected ValueError when split_evaluated='test'!"

        # 2. Report with test metrics injected
        test_metric_report = {
            "crop": "lettuce",
            "model_name": "lettuce_rf_val",
            "split_evaluated": "validation",
            "metrics": {"macro_f1": 0.85, "test_macro_f1": 0.82},
        }
        f2 = tmp_path / "model_val.json"
        with open(f2, "w", encoding="utf-8") as f:
            json.dump(test_metric_report, f)

        failed_metric = False
        try:
            validate_report_is_validation_only(test_metric_report, f2)
        except ValueError as err:
            failed_metric = True
            assert "Criterion 1 Violation" in str(err)
        assert failed_metric, "Expected ValueError when test_macro_f1 is present!"

    print("[PASS] Criterion 1: Script fails if test metrics are supplied")
    print(
        "       Verified strict rejection of test splits and forbidden test metric keys."
    )


def verify_criterion_2_selection_json_lists_runners_up() -> None:
    """Criterion 2: Selection JSON lists runners-up."""
    crops = ["lettuce", "spinach", "tomato"]

    for crop in crops:
        out_file = run_selection(crop)
        assert out_file.exists(), f"Output file '{out_file}' was not generated!"

        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "selected_model" in data, "Selection JSON missing 'selected_model'"
        assert "runners_up" in data, "Selection JSON missing 'runners_up'"
        runners_up = data["runners_up"]
        assert len(runners_up) >= 4, (
            f"Expected at least 4 runners-up for {crop}, found {len(runners_up)}"
        )

        # Verify ranks and descending macro-F1 order
        prev_f1 = data["selected_model"]["metrics"]["macro_f1"]
        for idx, runner in enumerate(runners_up, start=2):
            assert runner["rank"] == idx, f"Expected rank {idx}, got {runner['rank']}"
            assert "macro_f1_delta_from_selected" in runner
            assert runner["val_macro_f1"] <= prev_f1 + 1e-5, (
                "Runners-up not sorted in descending order"
            )
            assert "model_name" in runner
            assert "model_family" in runner
            prev_f1 = runner["val_macro_f1"]

    print("[PASS] Criterion 2: Selection JSON lists runners-up")
    print(
        "       Verified ranked runners-up with macro-F1 deltas across lettuce, spinach, and tomato."
    )


def verify_criterion_3_per_class_weaknesses_noted() -> None:
    """Criterion 3: Per-class weaknesses are noted."""
    crops = ["lettuce", "spinach", "tomato"]

    for crop in crops:
        out_file = ROOT / "reports" / crop / "sensor_selection.json"
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "per_class_weaknesses" in data, "Missing 'per_class_weaknesses' key"
        weaknesses = data["per_class_weaknesses"]
        assert "summary" in weaknesses
        assert "selected_model_weaknesses" in weaknesses

        sel_weak = weaknesses["selected_model_weaknesses"]
        assert "unsupported_classes_in_val" in sel_weak
        assert "low_recall_classes" in sel_weak
        assert "low_precision_classes" in sel_weak
        assert "diagnostic_notes" in sel_weak

        # Verify lettuce specifically catches zero-support classes in hold-out
        if crop == "lettuce":
            assert "EC_Low" in sel_weak["unsupported_classes_in_val"]
            assert "WaterTemp_Low" in sel_weak["unsupported_classes_in_val"]
            assert len(sel_weak["diagnostic_notes"]) >= 2

    print("[PASS] Criterion 3: Per-class weaknesses are noted")
    print(
        "       Verified zero-support classes, low-recall stress hazards, and agronomic impact notes."
    )


def main() -> None:
    """Run all T4-023 criteria verifications."""
    print("=" * 70)
    print("T4-023 Verification: Model Selection by Validation Macro-F1")
    print("=" * 70)

    verify_criterion_1_fails_on_test_metrics()
    verify_criterion_2_selection_json_lists_runners_up()
    verify_criterion_3_per_class_weaknesses_noted()

    print("\nALL CRITERIA PASSED! T4-023 is fully verified.")


if __name__ == "__main__":
    main()
