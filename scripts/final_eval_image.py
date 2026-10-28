#!/usr/bin/env python3
"""Final run-once test evaluation of selected image models (T5-032).

Adheres to Canonical Overview §18.2, §18.3, §18.5:
- Evaluates ONLY the model selected by validation macro-F1 (reports/<crop>/image_comparison.{json,md}).
- Evaluates on the held-out test split (split="test").
- Run-once guard: second run without --force refuses.
- Re-running requires both --force AND a non-empty written --reason.
- Appends every evaluation to reports/final_eval_log.jsonl (making re-runs visible).
- Generates Section 18.2 final test reports (JSON & Markdown) explicitly labelled as test results with split named.
- Updates model metadata in ModelRegistry with final_test_metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ml.evaluation.metrics import calculate_metrics, to_serializable_metrics
from ml.evaluation.report import write_evaluation_report
from ml.image_models.data import CropLeafDataset
from ml.image_models.efficientnet import build_efficientnet_b0
from ml.labels import IMAGE_CLASSES

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("final_eval_image")

SUPPORTED_CROPS: tuple[str, ...] = ("lettuce", "spinach", "tomato")

SELECTED_MODELS: dict[str, dict[str, Any]] = {
    "lettuce": {
        "model_name": "lettuce_efficientnet_b0_v1.0.0",
        "model_family": "EfficientNetB0",
        "artifact_version": "v1.0.0",
        "artifact_file": "models/image/lettuce/efficientnet_b0/v1.0.0/model.pt",
        "metadata_file": "models/image/lettuce/efficientnet_b0/v1.0.0/metadata.json",
        "split_protocol": "Stratified split; train, validation, and test partitions isolated",
        "split_manifest_file": "processed_data/splits/image_splits_lettuce.csv",
    },
    "spinach": {
        "model_name": "spinach_efficientnet_b0_v1.0.0",
        "model_family": "EfficientNetB0",
        "artifact_version": "v1.0.0",
        "artifact_file": "models/image/spinach/efficientnet_b0/v1.0.0/model.pt",
        "metadata_file": "models/image/spinach/efficientnet_b0/v1.0.0/metadata.json",
        "split_protocol": "Stratified split; train, validation, and test partitions isolated",
        "split_manifest_file": "processed_data/splits/image_splits_spinach.csv",
    },
    "tomato": {
        "model_name": "tomato_efficientnet_b0_v1.0.0",
        "model_family": "EfficientNetB0",
        "artifact_version": "v1.0.0",
        "artifact_file": "models/image/tomato/efficientnet_b0/v1.0.0/model.pt",
        "metadata_file": "models/image/tomato/efficientnet_b0/v1.0.0/metadata.json",
        "split_protocol": "Stratified split; train, validation, and test partitions isolated",
        "split_manifest_file": "processed_data/splits/image_splits_tomato.csv",
    },
}


def get_default_crop_limitations(crop: str) -> list[str]:
    """Return mandatory limitations disclosure for the final test evaluation."""
    clean_crop = crop.strip().lower()
    base = [
        (
            "Final Test Evaluation: Evaluated strictly on the preserved held-out test partition "
            "after all model selection decisions were finalized."
        ),
        (
            "Strict Zero Leakage: Test partition remained unread during transfer learning fine-tuning, "
            "hyperparameter exploration, and temperature calibration."
        ),
        (
            "Inherited Plant-Level Target Labels: Ground-truth target annotations represent whole-plant "
            "or specimen pathology; individual leaf crops inherit these labels which can introduce label "
            "noise for unaffected leaves on stressed plants."
        ),
        (
            "Un-oversampled Test Evaluation: The test split contains the untouched natural class distribution "
            "without synthetic oversampling or artificial balancing."
        ),
    ]
    if clean_crop == "lettuce":
        base.append(
            "Leaf Crops Instance Diagnosis: Evaluated on 156 segmented individual leaf crops isolated "
            "from whole-plant canopy photographs."
        )
    elif clean_crop == "spinach":
        base.append(
            "Leaf Photography Dataset: Evaluated on 2,745 leaf images across 6 target pathology and freshness classes."
        )
    elif clean_crop == "tomato":
        base.append(
            "Extensive Multi-Class Foliar Diagnosis: Evaluated on 5,034 leaf images across 11 target pathology classes."
        )
    return base


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


def is_image_entry(entry: dict[str, Any]) -> bool:
    """Return True if the log entry corresponds to an image model evaluation."""
    modality = str(entry.get("modality", "")).lower()
    model_name = str(entry.get("model_name", "")).lower()
    return modality == "image" or "efficientnet" in model_name or "image" in model_name


def check_already_evaluated(crop: str, log_path: Path) -> bool:
    """Check if the given crop image model has already been evaluated on test data."""
    clean_crop = crop.strip().lower()
    entries = read_evaluation_log(log_path)
    return any(e.get("crop") == clean_crop and is_image_entry(e) for e in entries)


def run_final_test_evaluation(
    crop: str,
    reports_dir: Path | str | None = None,
    registry_root: Path | str | None = None,
    log_file: Path | str | None = None,
    force: bool = False,
    reason: str | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute run-once test evaluation of the selected image model for a crop.

    Parameters:
        crop: Target crop family ('lettuce', 'spinach', 'tomato').
        reports_dir: Root directory for reports (default: reports).
        registry_root: Root directory for model artifacts (default: models).
        log_file: Path to final evaluation append-log (default: reports/final_eval_log.jsonl).
        force: Allow re-running evaluation if already executed.
        reason: Mandatory written justification when force=True.
        batch_size: DataLoader mini-batch size.
        device: 'cpu' or 'cuda' (defaults to cpu or detected).

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
    resolved_log_path = (
        Path(log_file).resolve()
        if log_file
        else resolved_reports_dir / "final_eval_log.jsonl"
    )

    # 1. Run-once guard (Acceptance Criterion 2: Re-run without --force refuses)
    already_evaluated = check_already_evaluated(clean_crop, log_path=resolved_log_path)
    if already_evaluated:
        if not force:
            raise RuntimeError(
                f"Final test evaluation has already been run for '{clean_crop}' image model"
                f" (see {resolved_log_path}). Re-running requires --force and a"
                " non-empty written --reason (Overview §18.3, §18.5)."
            )
        if reason is None or not reason.strip():
            raise ValueError(
                f"Re-running final test evaluation for '{clean_crop}' image model with"
                " --force requires a non-empty written --reason explaining why"
                " re-evaluation is necessary."
            )
        LOGGER.warning(
            "Re-running final test evaluation for '%s' image model under --force. Reason: '%s'",
            clean_crop,
            reason.strip(),
        )

    # 2. Identify selected model
    model_info = SELECTED_MODELS[clean_crop]
    model_name = model_info["model_name"]
    model_family = model_info["model_family"]
    artifact_version = model_info["artifact_version"]
    split_protocol = model_info["split_protocol"]

    artifact_path = (
        resolved_registry_root
        / "image"
        / clean_crop
        / "efficientnet_b0"
        / artifact_version
        / "model.pt"
    )
    metadata_path = (
        resolved_registry_root
        / "image"
        / clean_crop
        / "efficientnet_b0"
        / artifact_version
        / "metadata.json"
    )

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {artifact_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # 3. Load Model and Weights
    eval_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Evaluating '%s' on device: %s", model_name, eval_device)

    num_classes = len(IMAGE_CLASSES[clean_crop])
    model = build_efficientnet_b0(num_classes=num_classes)

    ckpt = torch.load(artifact_path, map_location=eval_device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.to(eval_device)
    model.eval()

    # 4. Load Held-Out Test Dataset
    LOGGER.info("Loading held-out test split for crop '%s'...", clean_crop)
    test_dataset = CropLeafDataset(crop=clean_crop, split="test", project_root=ROOT)
    test_sample_count = len(test_dataset)
    LOGGER.info("Loaded %d test samples for '%s'", test_sample_count, clean_crop)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 5. Execute Evaluation Inference
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(eval_device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    class_names = IMAGE_CLASSES[clean_crop]
    y_true = [class_names[t] for t in all_targets]
    y_pred = [class_names[p] for p in all_preds]

    # Calculate metrics using canonical library
    metrics = calculate_metrics(y_true, y_pred, labels=class_names)
    serializable_metrics = to_serializable_metrics(metrics)

    LOGGER.info(
        "Final Test Evaluation Results for '%s': Macro-F1=%.6f, Accuracy=%.6f",
        clean_crop,
        serializable_metrics["macro_f1"],
        serializable_metrics["accuracy"],
    )

    # 6. Gather Split Counts & Class Counts
    test_class_counts = dict(Counter(s["class_name"] for s in test_dataset.samples))
    val_dataset = CropLeafDataset(crop=clean_crop, split="val", project_root=ROOT)
    train_dataset = CropLeafDataset(crop=clean_crop, split="train", project_root=ROOT)

    split_counts = {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": test_sample_count,
    }

    class_counts_per_split = {
        "train": dict(Counter(s["class_name"] for s in train_dataset.samples)),
        "validation": dict(Counter(s["class_name"] for s in val_dataset.samples)),
        "test": test_class_counts,
    }

    # 7. Write Evaluation Report (Acceptance Criterion 3: Reports label metrics as test results with split named)
    limitations = get_default_crop_limitations(clean_crop)
    manifest_rel_path = model_info["split_manifest_file"]

    json_path, md_path = write_evaluation_report(
        crop=clean_crop,
        model_name=model_name,
        split_evaluated="test",
        metrics=metrics,
        dataset_manifest=manifest_rel_path,
        split_protocol=split_protocol,
        split_counts=split_counts,
        class_counts_per_split=class_counts_per_split,
        label_generation_method=(
            "Leaf crop instance diagnosis; crop labels inherited from whole-plant labels"
            if clean_crop == "lettuce"
            else "Individual leaf photography diagnosis"
        ),
        model_family=model_family,
        hyperparameters=metadata.get("hyperparameters", {}),
        limitations=limitations,
        seeds=metadata.get("hyperparameters", {}).get("seed", 42),
        device_used=str(eval_device),
        artifact_version=artifact_version,
        preprocessing_version="0.1.0",
        output_dir=resolved_reports_dir / clean_crop,
    )

    LOGGER.info("Generated test evaluation report JSON: %s", json_path)
    LOGGER.info("Generated test evaluation report Markdown: %s", md_path)

    # 8. Update Model Metadata with Final Test Metrics
    metadata["test_metrics"] = {
        "macro_f1": round(float(serializable_metrics["macro_f1"]), 6),
        "accuracy": round(float(serializable_metrics["accuracy"]), 6),
        "macro_precision": round(float(serializable_metrics["macro_precision"]), 6),
        "macro_recall": round(float(serializable_metrics["macro_recall"]), 6),
    }
    metadata["test_evaluation_date"] = datetime.now(timezone.utc).isoformat()
    metadata["test_sample_count"] = test_sample_count
    metadata["test_split_file"] = manifest_rel_path

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    LOGGER.info("Updated model metadata at %s", metadata_path)

    # 9. Append Log Entry to reports/final_eval_log.jsonl (Acceptance Criterion 1)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crop": clean_crop,
        "modality": "image",
        "model_name": model_name,
        "model_family": model_family,
        "artifact_version": artifact_version,
        "split_evaluated": "test",
        "test_metrics": {
            "macro_f1": round(float(serializable_metrics["macro_f1"]), 6),
            "accuracy": round(float(serializable_metrics["accuracy"]), 6),
            "macro_precision": round(float(serializable_metrics["macro_precision"]), 6),
            "macro_recall": round(float(serializable_metrics["macro_recall"]), 6),
        },
        "test_sample_count": test_sample_count,
        "test_file": str(ROOT / manifest_rel_path).replace("\\", "/"),
        "report_json": str(json_path).replace("\\", "/"),
        "report_md": str(md_path).replace("\\", "/"),
        "is_rerun": bool(already_evaluated),
        "rerun_reason": reason if already_evaluated else None,
    }

    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    LOGGER.info("Appended final test log entry to %s", resolved_log_path)

    return {
        "crop": clean_crop,
        "model_name": model_name,
        "split_evaluated": "test",
        "sample_count": test_sample_count,
        "metrics": serializable_metrics,
        "report_json": str(json_path),
        "report_md": str(md_path),
        "log_entry": log_entry,
    }


def main() -> int:
    """CLI entrypoint for final_eval_image.py."""
    parser = argparse.ArgumentParser(
        description="Run-once final test evaluation for image models (T5-032)."
    )
    parser.add_argument(
        "crop",
        type=str,
        help="Target crop: 'lettuce', 'spinach', 'tomato', or 'all'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation if this crop was already evaluated.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Written explanation required when using --force.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DataLoader batch size (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Reports output directory.",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="Models registry root directory.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to final_eval_log.jsonl.",
    )

    args = parser.parse_args()
    crop_arg = args.crop.strip().lower()

    crops_to_evaluate: list[str]
    if crop_arg == "all":
        crops_to_evaluate = list(SUPPORTED_CROPS)
    elif crop_arg in SUPPORTED_CROPS:
        crops_to_evaluate = [crop_arg]
    else:
        LOGGER.error(
            "Invalid crop '%s'. Must be one of %s or 'all'.",
            args.crop,
            SUPPORTED_CROPS,
        )
        return 1

    overall_success = True
    for c in crops_to_evaluate:
        LOGGER.info("==================================================")
        LOGGER.info("Starting Final Test Evaluation for: %s", c.upper())
        LOGGER.info("==================================================")
        try:
            res = run_final_test_evaluation(
                crop=c,
                reports_dir=args.reports_dir,
                registry_root=args.registry_root,
                log_file=args.log_file,
                force=args.force,
                reason=args.reason,
                batch_size=args.batch_size,
                device=args.device,
            )
            print(
                f"\n[OK] {c.capitalize()} Final Test Evaluation Complete:\n"
                f"     Model: {res['model_name']}\n"
                f"     Split: {res['split_evaluated']} (N={res['sample_count']})\n"
                f"     Macro-F1: {res['metrics']['macro_f1']:.6f}\n"
                f"     Accuracy: {res['metrics']['accuracy']:.6f}\n"
                f"     Report JSON: {res['report_json']}\n"
                f"     Report MD:   {res['report_md']}\n"
            )
        except Exception as err:  # noqa: BLE001
            LOGGER.error("Failed evaluation for '%s': %s", c, err)
            overall_success = False

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
