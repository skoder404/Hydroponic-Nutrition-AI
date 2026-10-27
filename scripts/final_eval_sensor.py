#!/usr/bin/env python3
"""Final run-once test evaluation of selected sensor models (T4-024).

Adheres to Canonical Overview §18.2, §18.3, §18.5:
- Evaluates ONLY the model selected by validation macro-F1 (reports/<crop>/sensor_selection.json).
- Evaluates on the held-out test split (test.csv).
- Run-once guard: second run without --force refuses.
- Re-running requires both --force AND a non-empty written --reason.
- Appends every evaluation to reports/final_eval_log.jsonl (making re-runs visible).
- Generates Section 18.2 final test reports (JSON & Markdown) explicitly labelled as test results.
- Updates model metadata in ModelRegistry with final_test_metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

from ml.evaluation.metrics import calculate_metrics, to_serializable_metrics
from ml.evaluation.report import write_evaluation_report
from ml.registry import ModelMetadata, ModelRegistry
from ml.sensor_models.schemas import (
    assert_no_leakage,
    extract_feature_matrix,
    get_sensor_schema,
)
from ml.sensor_models.train_sensor import train_sensor_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("final_eval_sensor")

SUPPORTED_CROPS: tuple[str, ...] = ("lettuce", "spinach", "tomato")

FAMILY_TO_MODEL_TYPE: dict[str, str] = {
    "decisiontreeclassifier": "decision_tree",
    "randomforestclassifier": "random_forest",
    "svc": "svm",
    "kneighborsclassifier": "knn",
    "logisticregression": "logistic_regression",
}

CROP_SPLIT_PROTOCOLS: dict[str, str] = {
    "lettuce": (
        "Preserved calendar day / group holdout (test.csv untouched until final"
        " evaluation)"
    ),
    "spinach": (
        "Preserved 6-hour time-block holdout (test.csv untouched until final"
        " evaluation)"
    ),
    "tomato": (
        "Preserved calendar month holdout (test.csv untouched until final evaluation)"
    ),
}


def get_default_crop_limitations(crop: str) -> list[str]:
    """Return mandatory limitations disclosure for the final test evaluation."""
    clean_crop = crop.strip().lower()
    base = [
        (
            "Final Test Evaluation: Performance evaluated strictly on the"
            " preserved held-out test partition after all model selection"
            " decisions were finalized."
        ),
        (
            "Rule-Derived Diagnostic Targets: Sensor labels are mathematically"
            " mapped from published physical action bands. Models learn the"
            " decision threshold geometry rather than independent biological"
            " plant health validated by agronomists."
        ),
        (
            "Un-oversampled Test Evaluation: The test split contains the"
            " untouched natural class distribution without synthetic"
            " oversampling."
        ),
        (
            "Strict Zero Leakage: Test partition remained unread during feature"
            " exploration, hyperparameter tuning, and candidate model"
            " selection."
        ),
    ]
    if clean_crop == "spinach":
        base.append(
            "Mixed Real and Synthetic Features: Air temperature is real"
            " greenhouse telemetry; relative humidity is 85.7% real and 14.3%"
            " adjusted; water temperature, pH, and EC are synthetic"
            " distributions."
        )
    elif clean_crop == "tomato":
        base.append(
            "Experimental Status: Prepared from simulated Kolar precision"
            " agriculture dataset with synthetic water temperature proxy shifts;"
            " not validated for live campus hardware."
        )
    elif clean_crop == "lettuce":
        base.append(
            "Small Physical Dataset: Held-out test set contains 162 physical"
            " sensor readings across 12 disjoint calendar days. Low-frequency"
            " classes EC_Low and WaterTemp_Low were excluded due to lack of"
            " support."
        )
    return base


def load_selection_metadata(
    crop: str, reports_dir: Path | str = "reports"
) -> dict[str, Any]:
    """Load the model selection record produced by T4-023.

    Raises:
        FileNotFoundError: If sensor_selection.json does not exist.
        ValueError: If selected_model is missing or malformed.
    """
    clean_crop = crop.strip().lower()
    selection_path = Path(reports_dir) / clean_crop / "sensor_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Missing selection record at '{selection_path}'. Run"
            f" scripts/select_model.py {clean_crop} first."
        )

    with open(selection_path, encoding="utf-8") as f:
        data = json.load(f)

    if "selected_model" not in data or not data["selected_model"]:
        raise ValueError(
            f"Selection record '{selection_path}' contains no valid 'selected_model'."
        )

    return data


def read_evaluation_log(log_path: Path) -> list[dict[str, Any]]:
    """Read all entries from the final evaluation log file."""
    if not log_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
    return entries


def check_already_evaluated(crop: str, log_path: Path) -> bool:
    """Check if the given crop has already been evaluated on test data."""
    clean_crop = crop.strip().lower()
    entries = read_evaluation_log(log_path)
    return any(e.get("crop") == clean_crop for e in entries)


def ensure_model_trained(
    crop: str,
    selected_model_info: dict[str, Any],
    data_dir: Path,
    registry_root: Path,
    reports_dir: Path,
) -> tuple[Any, ModelMetadata]:
    """Load the selected model from ModelRegistry or train it on train.csv."""
    clean_crop = crop.strip().lower()
    registry = ModelRegistry(root_dir=registry_root)
    version = str(selected_model_info.get("artifact_version", "1.0.0"))
    schema = get_sensor_schema(clean_crop)
    expected_features = list(schema.feature_names)

    # 1. Attempt to load from registry
    try:
        model_obj, model_meta = registry.load_model(
            crop=clean_crop,
            modality="sensor",
            version=version,
            expected_features=expected_features,
        )
        LOGGER.info(
            "Loaded existing candidate model '%s' (v%s) from ModelRegistry.",
            model_meta.model_id,
            version,
        )
        return model_obj, model_meta
    except (FileNotFoundError, KeyError):
        LOGGER.info(
            "Model artifact v%s not found in registry. Training selected model"
            " on train.csv...",
            version,
        )

    # 2. Train selected model strictly on train.csv using selected hyperparameters
    family = str(selected_model_info["model_family"]).lower()
    model_type = FAMILY_TO_MODEL_TYPE.get(family, "decision_tree")
    hyperparams = dict(selected_model_info.get("hyperparameters", {}))

    seed = hyperparams.pop("random_state", 42)
    experimental = clean_crop == "tomato"
    limitations = get_default_crop_limitations(clean_crop)

    train_res = train_sensor_model(
        crop=clean_crop,
        model_type=model_type,
        seed=seed,
        data_dir=data_dir,
        registry_root=registry_root,
        reports_dir=reports_dir / clean_crop,
        artifact_version=version,
        experimental=experimental,
        limitations=limitations,
        **hyperparams,
    )

    LOGGER.info(
        "Trained and saved candidate model '%s' at: %s",
        train_res.get("model_type"),
        train_res.get("artifact_path"),
    )

    model_obj, model_meta = registry.load_model(
        crop=clean_crop,
        modality="sensor",
        version=version,
        expected_features=expected_features,
    )
    return model_obj, model_meta


def run_final_test_evaluation(
    crop: str,
    data_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    registry_root: Path | str | None = None,
    log_file: Path | str | None = None,
    force: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Execute the final test evaluation for a crop.

    Parameters:
        crop: Target crop family ('lettuce', 'spinach', 'tomato').
        data_dir: Directory containing processed sensor data splits (default: processed_data/sensor/<crop>).
        reports_dir: Root directory for reports (default: reports).
        registry_root: Root directory for model artifacts (default: models).
        log_file: Path to final evaluation append-log (default: reports/final_eval_log.jsonl).
        force: Allow re-running evaluation if already executed.
        reason: Mandatory written justification when force=True.

    Returns:
        Summary dictionary containing test metrics, report paths, and run details.
    """
    clean_crop = crop.strip().lower()
    if clean_crop not in SUPPORTED_CROPS:
        raise ValueError(
            f"Unsupported crop '{crop}'. Must be one of {SUPPORTED_CROPS}."
        )

    resolved_reports_dir = (
        Path(reports_dir).resolve() if reports_dir else ROOT / "reports"
    )
    resolved_registry_root = (
        Path(registry_root).resolve() if registry_root else ROOT / "models"
    )
    resolved_data_dir = (
        Path(data_dir).resolve()
        if data_dir
        else ROOT / "processed_data" / "sensor" / clean_crop
    )
    resolved_log_path = (
        Path(log_file).resolve()
        if log_file
        else resolved_reports_dir / "final_eval_log.jsonl"
    )

    # 1. Run-once guard (Acceptance Criterion 2)
    already_evaluated = check_already_evaluated(clean_crop, log_path=resolved_log_path)
    if already_evaluated:
        if not force:
            raise RuntimeError(
                f"Final test evaluation has already been run for '{clean_crop}'"
                f" (see {resolved_log_path}). Re-running requires --force and a"
                " non-empty written --reason (Overview §18.3, §18.5)."
            )
        if reason is None or not reason.strip():
            raise ValueError(
                f"Re-running final test evaluation for '{clean_crop}' with"
                " --force requires a non-empty written --reason explaining why"
                " re-evaluation is necessary."
            )
        LOGGER.warning(
            "Re-running final test evaluation for '%s' under --force. Reason: '%s'",
            clean_crop,
            reason.strip(),
        )

    # 2. Load model selection record (Acceptance Criterion: evaluates ONLY selected model)
    selection_data = load_selection_metadata(
        clean_crop, reports_dir=resolved_reports_dir
    )
    selected_info = selection_data["selected_model"]
    model_name = selected_info["model_name"]
    model_family = selected_info["model_family"]
    artifact_version = str(selected_info.get("artifact_version", "1.0.0"))
    LOGGER.info(
        "Selected model for final evaluation: '%s' (%s v%s)",
        model_name,
        model_family,
        artifact_version,
    )

    # 3. Ensure selected model is available in ModelRegistry
    model_obj, model_meta = ensure_model_trained(
        crop=clean_crop,
        selected_model_info=selected_info,
        data_dir=resolved_data_dir,
        registry_root=resolved_registry_root,
        reports_dir=resolved_reports_dir,
    )

    # 4. Load held-out test split
    test_csv = resolved_data_dir / "test.csv"
    train_csv = resolved_data_dir / "train.csv"
    val_csv = resolved_data_dir / "val.csv"
    if not val_csv.is_file():
        val_csv = resolved_data_dir / "validation.csv"

    if not test_csv.is_file():
        raise FileNotFoundError(f"Held-out test split not found at '{test_csv}'")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training split not found at '{train_csv}'")
    if not val_csv.is_file():
        raise FileNotFoundError(f"Validation split not found at '{val_csv}'")

    test_df = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    LOGGER.info(
        "Loaded held-out test split for '%s': %d rows.",
        clean_crop,
        len(test_df),
    )

    # 5. Extract features and enforce strict leakage checks on model inputs
    X_test = extract_feature_matrix(test_df, clean_crop, check_bounds=True)
    assert_no_leakage(X_test, clean_crop)
    y_test = test_df["label"]

    # 6. Evaluate strictly on held-out test partition
    classes = list(model_meta.classes)
    y_test_pred = model_obj.predict(X_test)
    test_metrics = calculate_metrics(
        y_test, y_test_pred, labels=classes, zero_division=0.0
    )

    LOGGER.info(
        "Final Test Evaluation: Macro-F1 = %.4f | Accuracy = %.4f"
        " | Macro-Recall = %.4f",
        test_metrics["macro_f1"],
        test_metrics["accuracy"],
        test_metrics["macro_recall"],
    )

    # 7. Write final test report (Acceptance Criterion 3)
    split_protocol = CROP_SPLIT_PROTOCOLS.get(
        clean_crop,
        "Preserved holdout (test.csv untouched until final evaluation)",
    )
    split_counts = {
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
    }
    class_counts_per_split = {
        "train": {str(k): int(v) for k, v in train_df["label"].value_counts().items()},
        "validation": {
            str(k): int(v) for k, v in val_df["label"].value_counts().items()
        },
        "test": {str(k): int(v) for k, v in test_df["label"].value_counts().items()},
    }
    limitations = get_default_crop_limitations(clean_crop)

    crop_reports_dir = resolved_reports_dir / clean_crop
    crop_reports_dir.mkdir(parents=True, exist_ok=True)

    json_report_path, md_report_path = write_evaluation_report(
        crop=clean_crop,
        model_name=model_name,
        split_evaluated="test",
        metrics=test_metrics,
        dataset_manifest=f"processed_data/sensor/{clean_crop}/manifest.json",
        split_protocol=split_protocol,
        split_counts=split_counts,
        class_counts_per_split=class_counts_per_split,
        label_generation_method=(
            "Rule-derived thresholds mapped from physical hydroponic solution bands"
        ),
        model_family=model_family,
        hyperparameters=model_meta.hyperparameters,
        limitations=limitations,
        seeds=model_meta.seeds if hasattr(model_meta, "seeds") else {"seed": 42},
        device_used="cpu",
        artifact_version=artifact_version,
        preprocessing_version="1.0.0",
        output_dir=crop_reports_dir,
    )
    LOGGER.info("Wrote test evaluation JSON report to: %s", json_report_path)
    LOGGER.info("Wrote test evaluation Markdown report to: %s", md_report_path)

    # 8. Update ModelRegistry metadata with final_test_metrics
    meta_path = (
        resolved_registry_root
        / "sensor"
        / clean_crop
        / f"v{artifact_version}"
        / "metadata.json"
    )
    if meta_path.is_file():
        meta_dict = model_meta.model_dump()
        meta_dict["final_test_metrics"] = to_serializable_metrics(test_metrics)
        updated_meta = ModelMetadata.model_validate(meta_dict)
        updated_meta.save(meta_path)
        LOGGER.info("Updated ModelRegistry metadata at: %s", meta_path)

    # 9. Append to final_eval_log.jsonl (Acceptance Criterion 1)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crop": clean_crop,
        "model_name": model_name,
        "model_family": model_family,
        "artifact_version": artifact_version,
        "split_evaluated": "test",
        "test_metrics": {
            "macro_f1": round(float(test_metrics["macro_f1"]), 6),
            "accuracy": round(float(test_metrics["accuracy"]), 6),
            "macro_precision": round(float(test_metrics["macro_precision"]), 6),
            "macro_recall": round(float(test_metrics["macro_recall"]), 6),
        },
        "test_sample_count": len(test_df),
        "test_file": str(test_csv.as_posix()),
        "report_json": str(json_report_path.as_posix()),
        "report_md": str(md_report_path.as_posix()),
        "is_rerun": bool(already_evaluated),
        "rerun_reason": reason.strip() if (already_evaluated and reason) else None,
    }

    with open(resolved_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    LOGGER.info(
        "Recorded evaluation in log: %s (is_rerun=%s)",
        resolved_log_path,
        log_entry["is_rerun"],
    )

    return {
        "crop": clean_crop,
        "model_name": model_name,
        "artifact_version": artifact_version,
        "test_metrics": test_metrics,
        "report_json": json_report_path,
        "report_md": md_report_path,
        "log_entry": log_entry,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for final sensor test evaluation."""
    parser = argparse.ArgumentParser(
        description=("Run-once final test evaluation for selected crop sensor model.")
    )
    parser.add_argument(
        "crop_positional",
        nargs="?",
        choices=["lettuce", "spinach", "tomato", "all"],
        help="Target crop family ('lettuce', 'spinach', 'tomato', or 'all').",
    )
    parser.add_argument(
        "--crop",
        dest="crop_flag",
        choices=["lettuce", "spinach", "tomato", "all"],
        help="Target crop family via flag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation if an entry already exists in final_eval_log.jsonl.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Mandatory written justification when using --force.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional path to data split directory.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Optional root reports directory.",
    )
    parser.add_argument(
        "--registry-root",
        type=str,
        default=None,
        help="Optional model registry root directory.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to final_eval_log.jsonl.",
    )

    args = parser.parse_args(argv)
    target_crop = args.crop_positional or args.crop_flag

    if not target_crop:
        parser.error("A target crop must be specified (e.g. 'lettuce' or 'all').")

    crops_to_evaluate = list(SUPPORTED_CROPS) if target_crop == "all" else [target_crop]

    for c in crops_to_evaluate:
        print("=" * 70)
        print(f"Executing Final Test Evaluation for: {c.upper()}")
        print("=" * 70)
        try:
            res = run_final_test_evaluation(
                crop=c,
                data_dir=args.data_dir,
                reports_dir=args.reports_dir,
                registry_root=args.registry_root,
                log_file=args.log_file,
                force=args.force,
                reason=args.reason,
            )
            metrics = res["test_metrics"]
            print(f"\nFinal Test Evaluation Summary: {c.capitalize()}")
            print(f"- Selected Model: {res['model_name']} (v{res['artifact_version']})")
            print(f"- Test Macro-F1:  {metrics['macro_f1']:.4f}")
            print(f"- Test Accuracy:  {metrics['accuracy']:.4f}")
            print(f"- Test Recall:    {metrics['macro_recall']:.4f}")
            print(f"- Report JSON:    {res['report_json']}")
            print(f"- Report MD:      {res['report_md']}\n")
        except Exception as err:  # noqa: BLE001
            LOGGER.error("Evaluation failed for %s: %s", c, err)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
