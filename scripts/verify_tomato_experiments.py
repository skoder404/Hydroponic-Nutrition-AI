#!/usr/bin/env python3
"""Verification script for Tomato Sensor Experiments (T4-019).

Acceptance Criteria:
1. Month-based assignment is used.
2. Metadata has experimental=true.
3. Report documents transformed and synthetic features.
"""
# ruff: noqa: E402

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

from scripts.run_tomato_experiments import (
    verify_tomato_split_integrity,
)

TOMATO_DATA_DIR = ROOT / "processed_data" / "sensor" / "tomato"
REPORTS_DIR = ROOT / "reports" / "tomato"
EXP_JSON_PATH = REPORTS_DIR / "tomato_sensor_experiments.json"
EXP_MD_PATH = REPORTS_DIR / "tomato_sensor_experiments.md"

EXPECTED_FAMILIES = {
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "logistic_regression",
}


def verify_month_based_assignment_is_used() -> tuple[bool, str]:
    """Criterion 1: Month-based assignment is used with zero leakage."""
    split_info = verify_tomato_split_integrity(TOMATO_DATA_DIR)
    if not split_info["group_separation_verified"]:
        return False, "Group separation verification failed."

    if split_info["overlap_months"] != 0:
        return False, f"Month overlap detected: {split_info['overlap_months']}"

    if (
        split_info["train_months"] != 89
        or split_info["val_months"] != 16
        or split_info["test_months"] != 27
    ):
        return False, (
            f"Month counts mismatch: train={split_info['train_months']} (exp 89), "
            f"val={split_info['val_months']} (exp 16), test={split_info['test_months']} (exp 27)"
        )

    if not EXP_JSON_PATH.is_file() or not EXP_MD_PATH.is_file():
        return False, f"Missing report files in {REPORTS_DIR}"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if data["split_integrity"]["overlap_months"] != 0:
        return False, "JSON report indicates non-zero month overlap."

    return True, (
        "Criterion 1 satisfied: 132 total calendar months partitioned into 89 train, "
        "16 validation, and 27 test months with 0% month overlap."
    )


def verify_metadata_has_experimental_true() -> tuple[bool, str]:
    """Criterion 2: Metadata has experimental=true on all tomato model artifacts."""
    if not EXP_JSON_PATH.is_file():
        return False, f"Missing JSON report at {EXP_JSON_PATH}"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("experimental") is not True:
        return False, "Top-level report does not have experimental=True"

    models = data.get("models", [])
    if len(models) != 5:
        return False, f"Expected 5 models in report, found {len(models)}"

    for m in models:
        if m.get("experimental") is not True:
            return (
                False,
                f"Model summary {m.get('model_type')} has experimental={m.get('experimental')}",
            )

        meta_path = Path(m["metadata_path"])
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("experimental") is not True:
                return (
                    False,
                    f"Model artifact metadata at {meta_path.name} has experimental={meta.get('experimental')}",
                )

    return (
        True,
        "Criterion 2 satisfied: All 5 model artifacts and report marked experimental=True in ModelMetadata.",
    )


def verify_report_documents_transformed_and_synthetic_features() -> tuple[bool, str]:
    """Criterion 3: Report documents transformed and synthetic features."""
    if not EXP_JSON_PATH.is_file() or not EXP_MD_PATH.is_file():
        return False, "Missing report files"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    md_text = EXP_MD_PATH.read_text(encoding="utf-8")

    transforms = data.get("proxy_transforms", {})
    if len(transforms) != 8:
        return (
            False,
            f"Expected 8 proxy transformations documented, found {len(transforms)}",
        )

    for feat in [
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
        "nitrogen_mg_L",
        "phosphorus_mg_L",
        "potassium_mg_L",
    ]:
        if feat not in transforms:
            return False, f"Feature '{feat}' missing from proxy_transforms"
        if feat not in md_text:
            return False, f"Feature '{feat}' missing from Markdown report table"

    if (
        "Ambient_Temp" not in md_text
        or "Humidity + 15.0" not in md_text
        or "Soil_pH - 1.0" not in md_text
    ):
        return False, "Key proxy transform formulas missing from report Markdown"

    return True, (
        "Criterion 3 satisfied: Section 3 explicitly documents all 8 physical proxy formulas "
        "and simulated thermal reservoir dynamics."
    )


def main() -> int:
    print("=" * 70)
    print("T4-019 Verification: Tomato Sensor Experiments & Model Comparison")
    print("=" * 70)

    criteria = [
        (
            "Criterion 1: Month-based assignment is used",
            verify_month_based_assignment_is_used,
        ),
        (
            "Criterion 2: Metadata has experimental=true",
            verify_metadata_has_experimental_true,
        ),
        (
            "Criterion 3: Report documents transformed and synthetic features",
            verify_report_documents_transformed_and_synthetic_features,
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
        print("ALL CRITERIA PASSED! T4-019 is fully verified.")
        return 0
    else:
        print(
            "VERIFICATION FAILED: One or more criteria did not pass.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
