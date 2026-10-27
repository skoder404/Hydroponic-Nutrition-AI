#!/usr/bin/env python3
"""Model selection by validation macro-F1 (T4-023).

Selects the best sensor model for a crop strictly using validation split reports,
breaking ties deterministically, identifying per-class failure weaknesses,
and recording runner-up comparisons in reports/<crop>/sensor_selection.json.

Acceptance Criteria:
1. Script fails if test metrics are supplied (Overview §17.2, §18.3).
2. Selection JSON lists runners-up with performance deltas and tie-break rationale.
3. Per-class weaknesses and agronomic failure impacts are thoroughly documented.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

# Model complexity / parsimony preference order for identical performance
MODEL_COMPLEXITY_PREFERENCE: dict[str, int] = {
    "decision_tree": 1,
    "decisiontreeclassifier": 1,
    "logistic_regression": 2,
    "logisticregression": 2,
    "random_forest": 3,
    "randomforestclassifier": 3,
    "svm": 4,
    "svc": 4,
    "knn": 5,
    "kneighborsclassifier": 5,
}

# Domain-specific agronomic impact notes for class-level prediction failures (Overview §17.2)
AGRONOMIC_IMPACT_MAP: dict[str, dict[str, str]] = {
    "EC_High": {
        "hazard": "Root scorch and osmotic stress",
        "impact": "False negative permits toxic salt build-up scorching root tips; false positive triggers unnecessary freshwater dilution, starving plants.",
    },
    "EC_High_Toxicity": {
        "hazard": "Root scorch and osmotic stress",
        "impact": "False negative permits toxic salt build-up scorching root tips; false positive triggers unnecessary freshwater dilution, starving plants.",
    },
    "EC_Low": {
        "hazard": "General nutrient starvation and stunting",
        "impact": "False negative delays nutrient dosing leading to reduced vegetative yield; false positive causes premature nutrient addition.",
    },
    "EC_Low_Deficiency": {
        "hazard": "General nutrient starvation and stunting",
        "impact": "False negative delays nutrient dosing leading to reduced vegetative yield; false positive causes premature nutrient addition.",
    },
    "pH_Low": {
        "hazard": "Cation lockout and root acid burn",
        "impact": "False negative risks acidic root damage and calcium/magnesium lockout; false positive adds unnecessary base, precipitating minerals.",
    },
    "pH_High": {
        "hazard": "Micronutrient lockout and precipitation",
        "impact": "False negative leads to iron/zinc chlorosis and insoluble phosphate salts; false positive overdoses acid buffer.",
    },
    "pH_Imbalance": {
        "hazard": "Nutrient solution boundary deviation",
        "impact": "Failure to detect pH imbalance locks out vital cations/anions; corrective action must precede EC adjustments.",
    },
    "WaterTemp_High": {
        "hazard": "Root hypoxia and Pythium pathogen bloom",
        "impact": "False negative risks rapid root rot due to dropped dissolved oxygen; false positive runs chillers unnecessarily.",
    },
    "Water_Temp_High": {
        "hazard": "Root hypoxia and Pythium pathogen bloom",
        "impact": "False negative risks rapid root rot due to dropped dissolved oxygen; false positive runs chillers unnecessarily.",
    },
    "WaterTemp_Low": {
        "hazard": "Metabolic slowdown",
        "impact": "False negative suppresses nutrient uptake kinetics during cold solution periods.",
    },
    "Humidity_Low": {
        "hazard": "Excessive VPD and stomatal shutdown",
        "impact": "False negative causes transpirational stress and calcium deficiency (tipburn); false positive mists unnecessarily.",
    },
    "Humidity_Low_Stress": {
        "hazard": "Excessive VPD and stomatal shutdown",
        "impact": "False negative causes transpirational stress and calcium deficiency (tipburn); false positive mists unnecessarily.",
    },
    "Humidity_High_Mildew_Risk": {
        "hazard": "Guttation and fungal spore germination",
        "impact": "False negative permits powdery/downy mildew infestation; false positive increases ventilation heating costs.",
    },
    "Multiple_Stress": {
        "hazard": "Compound ecological breakdown",
        "impact": "Misclassifying compound stress as single-stress leads to partial or counterproductive dosing intervention.",
    },
    "N_Deficiency": {
        "hazard": "Macronutrient nitrogen chlorosis",
        "impact": "False negative delays nitrogenous fertilizer dosing causing chlorosis in older foliage and stunted vegetative canopy.",
    },
    "P_Deficiency": {
        "hazard": "Macronutrient phosphorus deficit",
        "impact": "False negative causes root growth stagnation and characteristic dark/purple foliar pigmentation.",
    },
    "K_Deficiency": {
        "hazard": "Macronutrient potassium necrosis",
        "impact": "False negative causes marginal leaf burn, weak stems, and poor fruit development.",
    },
    "Normal": {
        "hazard": "False alarm vs undetected stress",
        "impact": "False positive masks stress as healthy; false negative initiates unneeded chemical corrections on a stable reservoir.",
    },
}


def validate_report_is_validation_only(
    report: dict[str, Any], filepath: Path | str
) -> None:
    """Ensure report contains only validation metrics; reject if test metrics are present (Criterion 1).

    Raises:
        ValueError: If test metrics, test split evaluation, or test counts are detected.
    """
    path_str = str(filepath)

    # 1. Reject by filename if it explicitly indicates a test split report
    if "_test.json" in path_str.lower() or "-test.json" in path_str.lower():
        raise ValueError(
            f"Criterion 1 Violation: Test split report '{filepath}' supplied. "
            f"Model selection must strictly evaluate validation reports only (Overview §17.2, §18.3)."
        )

    # 2. Reject if split_evaluated field is 'test' or not 'validation'
    split_eval = report.get("split_evaluated", "").strip().lower()
    if split_eval == "test":
        raise ValueError(
            f"Criterion 1 Violation: Report '{filepath}' has split_evaluated='test'. "
            f"Model selection on test set before promotion is strictly forbidden."
        )
    if split_eval and split_eval not in ("validation", "val"):
        raise ValueError(
            f"Criterion 1 Violation: Report '{filepath}' evaluated unsupported split '{split_eval}'. "
            f"Only validation reports may be used for model selection."
        )

    # 3. Reject if test metric keys are present in top-level or metrics dictionary
    forbidden_metric_keys = {
        "test_macro_f1",
        "test_accuracy",
        "test_f1",
        "test_loss",
        "test_metrics",
        "metrics_test",
        "test_precision",
        "test_recall",
    }
    present_keys = forbidden_metric_keys.intersection(report.keys())
    if present_keys:
        raise ValueError(
            f"Criterion 1 Violation: Forbidden test metric keys {sorted(present_keys)} "
            f"found in '{filepath}'. Test metrics must never influence model selection."
        )

    metrics = report.get("metrics", {})
    present_subkeys = forbidden_metric_keys.intersection(metrics.keys())
    if present_subkeys:
        raise ValueError(
            f"Criterion 1 Violation: Forbidden test metric keys {sorted(present_subkeys)} "
            f"found in metrics dictionary of '{filepath}'."
        )

    # 4. Check split counts: if test split has evaluated sample counts > 0 and metrics exist
    split_counts = report.get("split_counts", {})
    if split_counts.get("test", 0) > 0 and split_eval == "test":
        raise ValueError(
            f"Criterion 1 Violation: Test samples were evaluated in report '{filepath}'."
        )


def analyze_per_class_weaknesses(
    metrics: dict[str, Any], crop: str, class_counts: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Inspect class-wise metrics, zero support, low recall, and map failure impacts (Criterion 3)."""
    per_class = metrics.get("per_class", {})
    unsupported_classes: list[str] = []
    low_recall_classes: list[dict[str, Any]] = []
    low_precision_classes: list[dict[str, Any]] = []
    healthy_classes: list[dict[str, Any]] = []

    for cls_name, cls_metrics in per_class.items():
        support = cls_metrics.get("support", 0)
        precision = cls_metrics.get("precision", 0.0)
        recall = cls_metrics.get("recall", 0.0)
        f1 = cls_metrics.get("f1", 0.0)

        impact_info = AGRONOMIC_IMPACT_MAP.get(
            cls_name,
            {
                "hazard": "Nutritional/environmental stress boundary breach",
                "impact": f"Misclassification of {cls_name} risks incorrect reservoir adjustment.",
            },
        )

        entry = {
            "class": cls_name,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hazard": impact_info["hazard"],
            "failure_impact": impact_info["impact"],
        }

        if support == 0:
            unsupported_classes.append(cls_name)
        elif recall < 0.85:
            low_recall_classes.append(entry)
        elif precision < 0.85:
            low_precision_classes.append(entry)
        else:
            healthy_classes.append(entry)

    notes: list[str] = []
    if unsupported_classes:
        notes.append(
            f"Classes with 0 validation support in hold-out: {unsupported_classes}. "
            f"Generalization on these minority classes cannot be verified from validation data alone."
        )
    if low_recall_classes:
        low_names = [item["class"] for item in low_recall_classes]
        notes.append(
            f"Classes with recall < 0.85: {low_names}. Risk of false negatives under stress conditions."
        )
    if low_precision_classes:
        prec_names = [item["class"] for item in low_precision_classes]
        notes.append(
            f"Classes with precision < 0.85: {prec_names}. Risk of false positive nuisance alarms."
        )

    return {
        "unsupported_classes_in_val": unsupported_classes,
        "low_recall_classes": low_recall_classes,
        "low_precision_classes": low_precision_classes,
        "healthy_classes": [item["class"] for item in healthy_classes],
        "diagnostic_notes": notes,
    }


def select_best_model_for_crop(
    crop: str,
    reports_dir: Path | None = None,
    explicit_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Load validation reports for a crop, select the best model by macro-F1, and rank runners-up."""
    crop_clean = crop.strip().lower()
    if reports_dir is None:
        reports_dir = ROOT / "reports" / crop_clean

    if explicit_files:
        report_files = [Path(p) for p in explicit_files]
    else:
        # Match all individual model validation reports (exclude comparison summaries and grouped reports)
        pattern = f"{crop_clean}_sensor_*_validation.json"
        report_files = sorted(reports_dir.glob(pattern))
        if not report_files:
            # Fallback to any validation JSON reports in custom directory
            report_files = sorted(
                f
                for f in reports_dir.glob("*_validation.json")
                if not f.name.endswith(("_experiments.json", "_grouped_cv_report.json"))
            )

    if not report_files:
        raise FileNotFoundError(
            f"No validation reports found for crop '{crop_clean}' in '{reports_dir}'"
        )

    candidates: list[dict[str, Any]] = []

    for fpath in report_files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        # Enforce Criterion 1 check on every report
        validate_report_is_validation_only(data, fpath)

        metrics = data.get("metrics", {})
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        macro_recall = float(metrics.get("macro_recall", 0.0))
        accuracy = float(metrics.get("accuracy", 0.0))
        macro_precision = float(metrics.get("macro_precision", 0.0))

        model_name = data.get("model_name", fpath.stem)
        model_family = data.get("model_family", "Unknown")
        model_family_lower = model_family.lower().replace(" ", "").replace("_", "")

        # Look up complexity rank for tie-break
        parsimony_rank = 99
        for key, rank in MODEL_COMPLEXITY_PREFERENCE.items():
            if key in model_family_lower or key in model_name.lower():
                parsimony_rank = rank
                break

        # Check for cross-validation metrics if recorded
        cv_macro_f1 = float(data.get("cv_macro_f1", 0.0))

        weaknesses = analyze_per_class_weaknesses(
            metrics=metrics,
            crop=crop_clean,
            class_counts=data.get("class_counts_per_split", {}).get("validation"),
        )

        candidates.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "model_family_raw": model_family,
                "report_file": str(
                    fpath.relative_to(ROOT) if fpath.is_relative_to(ROOT) else fpath
                ),
                "val_macro_f1": round(macro_f1, 6),
                "val_macro_recall": round(macro_recall, 6),
                "val_accuracy": round(accuracy, 6),
                "val_macro_precision": round(macro_precision, 6),
                "cv_macro_f1": round(cv_macro_f1, 6),
                "parsimony_rank": parsimony_rank,
                "hyperparameters": data.get("hyperparameters", {}),
                "split_evaluated": data.get("split_evaluated", "validation"),
                "artifact_version": data.get("artifact_version", "1.0.0"),
                "device_used": data.get("device_used", "cpu"),
                "weaknesses": weaknesses,
                "metrics_summary": {
                    "macro_f1": round(macro_f1, 6),
                    "macro_recall": round(macro_recall, 6),
                    "accuracy": round(accuracy, 6),
                    "macro_precision": round(macro_precision, 6),
                },
            }
        )

    # Deterministic Multi-Tier Ranking:
    # 1. Primary: Validation Macro-F1 (descending)
    # 2. Tie-break 1: Validation Macro-Recall (descending)
    # 3. Tie-break 2: Validation Accuracy (descending)
    # 4. Tie-break 3: Model Parsimony / Simplicity (ascending order: 1=Decision Tree, 2=LogReg, 3=RF, 4=SVM, 5=KNN)
    candidates.sort(
        key=lambda m: (
            -m["val_macro_f1"],
            -m["val_macro_recall"],
            -m["val_accuracy"],
            m["parsimony_rank"],
        )
    )

    winner = candidates[0]
    runners_up: list[dict[str, Any]] = []

    # Build runners-up list with explicit rank and delta (Criterion 2)
    for rank_idx, candidate in enumerate(candidates[1:], start=2):
        delta = round(candidate["val_macro_f1"] - winner["val_macro_f1"], 6)
        runners_up.append(
            {
                "rank": rank_idx,
                "model_name": candidate["model_name"],
                "model_family": candidate["model_family"],
                "val_macro_f1": candidate["val_macro_f1"],
                "macro_f1_delta_from_selected": delta,
                "val_accuracy": candidate["val_accuracy"],
                "val_macro_recall": candidate["val_macro_recall"],
                "val_macro_precision": candidate["val_macro_precision"],
                "report_file": candidate["report_file"],
                "hyperparameters": candidate["hyperparameters"],
                "unsupported_classes_count": len(
                    candidate["weaknesses"]["unsupported_classes_in_val"]
                ),
                "low_recall_classes_count": len(
                    candidate["weaknesses"]["low_recall_classes"]
                ),
            }
        )

    # Check if a tie-break was applied against second place
    tie_break_applied = False
    tie_break_reason: str | None = None
    if (
        runners_up
        and abs(winner["val_macro_f1"] - runners_up[0]["val_macro_f1"]) < 1e-5
    ):
        tie_break_applied = True
        r1 = runners_up[0]
        tie_break_reason = (
            f"Validation macro-F1 tied at {winner['val_macro_f1']:.4f} between "
            f"'{winner['model_name']}' ({winner['model_family']}) and "
            f"'{r1['model_name']}' ({r1['model_family']}). "
            f"Selection decided by secondary criteria: "
            f"Recall ({winner['val_macro_recall']:.4f} vs {r1['val_macro_recall']:.4f}), "
            f"Accuracy ({winner['val_accuracy']:.4f} vs {r1['val_accuracy']:.4f}), and "
            f"model parsimony preference (rank {winner['parsimony_rank']} vs {candidates[1]['parsimony_rank']})."
        )

    # Compile comprehensive selection payload
    selection_record = {
        "crop": crop_clean,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "T4-023",
        "selection_metric": "validation_macro_f1",
        "tie_break_hierarchy": [
            "1. Primary: validation_macro_f1 (descending)",
            "2. Tie-Break 1: validation_macro_recall (descending, ensures balanced stress coverage)",
            "3. Tie-Break 2: validation_accuracy (descending)",
            "4. Tie-Break 3: model parsimony and explainability (Decision Tree > LogReg > Random Forest > SVM > KNN)",
        ],
        "selected_model": {
            "model_name": winner["model_name"],
            "model_family": winner["model_family"],
            "report_file": winner["report_file"],
            "artifact_version": winner["artifact_version"],
            "metrics": winner["metrics_summary"],
            "hyperparameters": winner["hyperparameters"],
            "device_used": winner["device_used"],
        },
        "tie_break_applied": tie_break_applied,
        "tie_break_notes": tie_break_reason
        or "Selected strictly by highest validation macro-F1 score without tie.",
        "exceptions_to_macro_f1": None,
        "runners_up": runners_up,
        "per_class_weaknesses": {
            "summary": (
                f"Evaluation of '{winner['model_name']}' on {crop_clean} validation split identified "
                f"{len(winner['weaknesses']['unsupported_classes_in_val'])} unsupported minority classes, "
                f"{len(winner['weaknesses']['low_recall_classes'])} low-recall classes (<0.85), and "
                f"{len(winner['weaknesses']['low_precision_classes'])} low-precision classes (<0.85)."
            ),
            "selected_model_weaknesses": winner["weaknesses"],
            "runners_up_weaknesses_summary": [
                {
                    "rank": r["rank"],
                    "model_name": r["model_name"],
                    "low_recall_count": r["low_recall_classes_count"],
                    "unsupported_count": r["unsupported_classes_count"],
                }
                for r in runners_up
            ],
        },
        "limits_and_safety_notes": (
            "Model selection is based solely on validation split metrics. "
            "Rule-derived synthetic features and hold-out zero-support classes must be disclosed in UI. "
            "Test split evaluation must occur only in run-once final evaluation (T4-024)."
        ),
    }

    return selection_record


def run_selection(
    crop: str,
    reports_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Execute model selection for crop and write sensor_selection.json."""
    crop_clean = crop.strip().lower()
    record = select_best_model_for_crop(crop_clean, reports_dir=reports_dir)

    if output_path is None:
        target_dir = reports_dir or (ROOT / "reports" / crop_clean)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / "sensor_selection.json"
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # Print clean summary table
    sel = record["selected_model"]
    m = sel["metrics"]
    print("\n======================================================================")
    print(f"SENSOR MODEL SELECTION RESULTS: {crop_clean.upper()}")
    print("======================================================================")
    print(f"  Selected Model  : {sel['model_name']} ({sel['model_family']})")
    print(f"  Validation F1   : {m['macro_f1']:.4f} (Macro-F1)")
    print(f"  Validation Rec  : {m['macro_recall']:.4f} (Macro-Recall)")
    print(f"  Validation Acc  : {m['accuracy']:.4f} (Accuracy)")
    print(f"  Tie-Break Note  : {record['tie_break_notes']}")
    print(
        f"  Selection Output: {out_file.relative_to(ROOT) if out_file.is_relative_to(ROOT) else out_file}"
    )
    print(f"\n  Runners-up ({len(record['runners_up'])} models):")
    for r in record["runners_up"]:
        print(
            f"    #{r['rank']:d} {r['model_name']:45s} "
            f"F1={r['val_macro_f1']:.4f} (Δ={r['macro_f1_delta_from_selected']:+.4f}) "
            f"Acc={r['val_accuracy']:.4f}"
        )

    diag = record["per_class_weaknesses"]["selected_model_weaknesses"][
        "diagnostic_notes"
    ]
    if diag:
        print("\n  Per-Class Weakness Notes:")
        for note in diag:
            print(f"    - {note}")
    print("======================================================================\n")

    return out_file


def main() -> None:
    """CLI entry point for model selection."""
    parser = argparse.ArgumentParser(
        description="Select best sensor model by validation macro-F1 (T4-023)"
    )
    parser.add_argument(
        "crop",
        nargs="?",
        default=None,
        choices=["lettuce", "spinach", "tomato"],
        help="Target crop for model selection (lettuce, spinach, tomato)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute model selection across all three crops (lettuce, spinach, tomato)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Custom reports directory (defaults to reports/<crop>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output JSON path (defaults to reports/<crop>/sensor_selection.json)",
    )

    args = parser.parse_args()

    if args.all:
        crops = ["lettuce", "spinach", "tomato"]
    elif args.crop:
        crops = [args.crop]
    else:
        parser.print_help()
        sys.exit(1)

    for c in crops:
        run_selection(c, reports_dir=args.reports_dir, output_path=args.output)


if __name__ == "__main__":
    main()
