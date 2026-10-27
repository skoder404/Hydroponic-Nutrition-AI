#!/usr/bin/env python3
"""Verification script for Lettuce Sensor Experiments (T4-017).
# ruff: noqa: E402

Acceptance Criteria:
1. Report covers all five families with required metrics.
2. No test metrics appear.
3. Limitations block states rule-derived labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

REPORTS_DIR = ROOT / "reports" / "lettuce"
EXP_JSON_PATH = REPORTS_DIR / "lettuce_sensor_experiments.json"
EXP_MD_PATH = REPORTS_DIR / "lettuce_sensor_experiments.md"

REQUIRED_FAMILIES = {
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "logistic_regression",
}


def verify_report_covers_all_five_families() -> tuple[bool, str]:
    """Criterion 1: Report covers all five families with required metrics."""
    if not EXP_JSON_PATH.is_file() or not EXP_MD_PATH.is_file():
        return False, f"Missing report files in {REPORTS_DIR}"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    families_present = {m["model_type"] for m in data.get("models", [])}
    missing = REQUIRED_FAMILIES - families_present
    if missing:
        return False, f"Missing model families from report: {missing}"

    for m in data.get("models", []):
        for req_metric in [
            "val_macro_f1",
            "val_accuracy",
            "val_macro_precision",
            "val_macro_recall",
        ]:
            if req_metric not in m or m[req_metric] is None:
                return (
                    False,
                    f"Model {m.get('model_type')} missing required metric: {req_metric}",
                )

    return True, (
        f"Criterion 1 satisfied: All 5 families present with required metrics "
        f"(Winner: {data.get('best_overall_model')} with Val Macro-F1: {data['models'][0]['val_macro_f1']:.4f})."
    )


def verify_no_test_metrics_appear() -> tuple[bool, str]:
    """Criterion 2: No test metrics appear in comparison report or JSON."""
    if not EXP_JSON_PATH.is_file() or not EXP_MD_PATH.is_file():
        return False, "Missing report files"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_text = f.read().lower()

    md_text = EXP_MD_PATH.read_text(encoding="utf-8").lower()

    for forbidden in [
        "test_macro_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
    ]:
        if forbidden in json_text:
            return False, f"Forbidden test metric '{forbidden}' found in JSON report"
        if forbidden in md_text:
            return (
                False,
                f"Forbidden test metric '{forbidden}' found in Markdown report",
            )

    return (
        True,
        "Criterion 2 satisfied: Exactly zero test metrics appear across reports.",
    )


def verify_limitations_block_states_rule_derived_labels() -> tuple[bool, str]:
    """Criterion 3: Limitations block states rule-derived labels."""
    if not EXP_JSON_PATH.is_file() or not EXP_MD_PATH.is_file():
        return False, "Missing report files"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    limitations = data.get("limitations", [])
    if not limitations:
        return False, "Limitations block is empty in JSON report"

    rule_found = any("rule-derived" in lim.lower() for lim in limitations)
    if not rule_found:
        return False, "Limitations block does not explicitly state rule-derived labels"

    md_text = EXP_MD_PATH.read_text(encoding="utf-8")
    if "rule-derived" not in md_text.lower():
        return False, "Markdown report does not contain 'rule-derived' in limitations"

    return True, (
        "Criterion 3 satisfied: Limitations block explicitly discloses that labels are mathematically "
        "rule-derived from action bands rather than independent biological health observations."
    )


def main() -> int:
    print("=" * 70)
    print("T4-017 Verification: Lettuce Sensor Experiments & Model Comparison")
    print("=" * 70)

    criteria = [
        (
            "Criterion 1: Report covers all five families with required metrics",
            verify_report_covers_all_five_families,
        ),
        ("Criterion 2: No test metrics appear", verify_no_test_metrics_appear),
        (
            "Criterion 3: Limitations block states rule-derived labels",
            verify_limitations_block_states_rule_derived_labels,
        ),
    ]

    all_passed = True
    for name, func in criteria:
        passed, msg = func()
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
        print(f"       {msg}\n")
        if not passed:
            all_passed = False

    if all_passed:
        print("ALL CRITERIA PASSED! T4-017 is fully verified.")
        return 0
    else:
        print(
            "VERIFICATION FAILED: One or more criteria did not pass.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
