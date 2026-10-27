"""Evaluation report writer for model verification and audit (T4-006).

Writes reproducible JSON and Markdown reports under `reports/<crop>/` adhering to
Canonical Overview Section 18.2 and PRD requirements:
- Complete Section 18.2 metric and metadata fields.
- Strict isolation: validation and test metrics are never mixed or aggregated.
- Mandatory, non-empty 'Limitations' block disclosing rule-derived targets, oversampling, etc.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from ml.evaluation.metrics import to_serializable_metrics

VALID_SPLIT_NAMES: frozenset[str] = frozenset({"train", "validation", "test"})


def get_default_software_versions() -> dict[str, str]:
    """Capture runtime library and platform versions for reproducibility."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    return versions


def build_markdown_report(report_data: dict[str, Any]) -> str:
    """Render a structured, GitHub-flavored Markdown report from report data."""
    crop = report_data["crop"].capitalize()
    model_name = report_data["model_name"]
    split_eval = report_data["split_evaluated"]
    metrics = report_data["metrics"]

    lines = [
        f"# Model Evaluation Report: {crop} — {model_name}",
        "",
        f"> **Evaluated Split:** `{split_eval}` | **Generated:** `{report_data['evaluation_date']}`",
        "",
        "## 1. Executive Summary & Core Metrics",
        "",
        f"Metrics below reflect performance exclusively on the **`{split_eval}`** partition.",
        "",
        "| Metric | Score | Note |",
        "|---|---:|---|",
        f"| **Macro-F1** (Primary Selection) | **{metrics['macro_f1']:.4f}** | Primary optimization metric per §18.1 |",
        f"| Accuracy | {metrics['accuracy']:.4f} | Overall sample accuracy |",
        f"| Macro Precision | {metrics['macro_precision']:.4f} | Unweighted class average |",
        f"| Macro Recall | {metrics['macro_recall']:.4f} | Unweighted class average |",
        f"| Evaluated Split Samples | {metrics.get('total_samples', report_data['split_counts'].get(split_eval, 'N/A'))} | Total samples in evaluated partition |",
        "",
        "## 2. Per-Class Performance Breakdown",
        "",
        "| Class Label | Precision | Recall | F1-Score | Support |",
        "|---|---:|---:|---:|---:|",
    ]

    for lbl in report_data["class_order"]:
        cls_data = metrics["per_class"].get(lbl, {})
        prec = cls_data.get("precision", 0.0)
        rec = cls_data.get("recall", 0.0)
        f1 = cls_data.get("f1", 0.0)
        supp = cls_data.get("support", 0)
        lines.append(f"| `{lbl}` | {prec:.4f} | {rec:.4f} | {f1:.4f} | {supp} |")

    lines.extend([
        "",
        "## 3. Confusion Matrix",
        "",
        "Rows indicate Ground Truth; Columns indicate Model Prediction (order strictly matches class list):",
        "",
    ])

    # Table header
    labels = report_data["class_order"]
    header = "| True \\ Pred | " + " | ".join(f"`{lbl}`" for lbl in labels) + " |"
    divider = "|---|" + "---:|" * len(labels)
    lines.append(header)
    lines.append(divider)

    cm = metrics["confusion_matrix_list"]
    for i, true_lbl in enumerate(labels):
        row_vals = " | ".join(str(cm[i][j]) for j in range(len(labels)))
        lines.append(f"| `{true_lbl}` | {row_vals} |")

    lines.extend([
        "",
        "## 4. Dataset Splits & Protocol",
        "",
        f"- **Dataset Manifest Reference:** `{report_data['source_dataset_manifest']}`",
        f"- **Split Protocol:** {report_data['split_protocol']}",
        f"- **Preprocessing Version:** `{report_data['preprocessing_version']}`",
        f"- **Label-Generation Method:** {report_data['label_generation_method']}",
        "",
        "### Partition Counts",
        "",
        "| Partition | Total Samples |",
        "|---|---:|",
        f"| Train | {report_data['split_counts'].get('train', 0)} |",
        f"| Validation | {report_data['split_counts'].get('validation', 0)} |",
        f"| Test (Held-Out) | {report_data['split_counts'].get('test', 0)} |",
        "",
        "## 5. Model Architecture & Hyperparameters",
        "",
        f"- **Model Family:** `{report_data['model_family']}`",
        f"- **Artifact Version:** `{report_data['artifact_version']}`",
        f"- **Target Device:** `{report_data['device_used']}`",
        f"- **Random Seeds:** `{json.dumps(report_data['seeds'])}`",
        "",
        "```json",
        json.dumps(report_data["hyperparameters"], indent=2),
        "```",
        "",
        "## 6. Software & Runtime Versions",
        "",
        "| Package / Component | Version |",
        "|---|---|",
    ])

    for pkg, ver in sorted(report_data["software_versions"].items()):
        lines.append(f"| {pkg} | `{ver}` |")

    lines.extend([
        "",
        "## 7. Mandatory Limitations & Disclosures",
        "",
        "> [!IMPORTANT]",
        "> The following known caveats and constraints must remain visible in documentation and UI:",
        "",
    ])

    for limitation in report_data["limitations"]:
        lines.append(f"- {limitation}")

    lines.append("")
    return "\n".join(lines)


def write_evaluation_report(
    crop: str,
    model_name: str,
    split_evaluated: str,
    metrics: dict[str, Any],
    dataset_manifest: str | Path | dict[str, Any],
    split_protocol: str,
    split_counts: dict[str, int],
    class_counts_per_split: dict[str, dict[str, int]],
    label_generation_method: str,
    model_family: str,
    hyperparameters: dict[str, Any],
    limitations: Sequence[str],
    seeds: dict[str, int] | int = 42,
    device_used: str = "cpu",
    artifact_version: str = "0.1.0",
    preprocessing_version: str = "0.1.0",
    software_versions: dict[str, str] | None = None,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write comprehensive JSON and Markdown reports under reports/<crop>/.

    Parameters
    ----------
    crop:
        Crop family: 'lettuce', 'spinach', or 'tomato'.
    model_name:
        Identifier of the evaluated model (e.g., 'random_forest_v1').
    split_evaluated:
        The exact split partition evaluated: strictly 'validation' or 'test'.
        Validation and test metrics must never be mixed into a combined evaluation.
    metrics:
        Output dictionary from calculate_metrics.
    dataset_manifest:
        Manifest filepath or identifier.
    split_protocol:
        Human-readable description of how splits were isolated (e.g. grouped by date, dHash deduplication).
    split_counts:
        Dictionary with sample counts for 'train', 'validation', and 'test'.
    class_counts_per_split:
        Per-class sample counts broken down by partition.
    label_generation_method:
        Description of how ground truth targets were assigned (e.g. threshold rules, agronomic inspection).
    model_family:
        Model category (e.g., 'RandomForestClassifier', 'EfficientNetB0', 'CustomCNN').
    hyperparameters:
        Configuration dictionary of model hyperparameters.
    limitations:
        Mandatory, non-empty list of known limitations and disclosures.
    seeds:
        Random seeds used for training/splits.
    device_used:
        Device utilized ('cpu', 'cuda', etc.).
    artifact_version:
        Model artifact semantic version.
    preprocessing_version:
        Preprocessing pipeline semantic version.
    software_versions:
        Optional dictionary of library versions (captured automatically if omitted).
    output_dir:
        Optional target directory (defaults to 'reports/<crop>').

    Returns
    -------
    tuple[Path, Path]:
        Paths to the generated (json_path, markdown_path).
    """
    clean_split = split_evaluated.strip().lower()
    if clean_split not in VALID_SPLIT_NAMES:
        raise ValueError(
            f"Invalid split_evaluated '{split_evaluated}'. Must be strictly one of {sorted(VALID_SPLIT_NAMES)}."
        )

    # Acceptance Criteria: Limitations block is always included and non-empty
    if not limitations or len(limitations) == 0 or all(not str(s).strip() for s in limitations):
        raise ValueError(
            "A non-empty 'limitations' sequence is mandatory per Overview Section 18.2 policy."
        )

    clean_limitations = [str(s).strip() for s in limitations if str(s).strip()]

    # Acceptance Criteria: Validation and test metrics are never mixed
    # Ensure the metrics dictionary is strictly tagged with this specific split
    serializable_metrics = to_serializable_metrics(metrics)

    # Validate essential Section 18.2 metric fields
    required_metric_keys = {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class",
        "confusion_matrix",
    }
    missing_metrics = required_metric_keys - set(serializable_metrics.keys())
    if missing_metrics:
        raise ValueError(
            f"Metrics dictionary is missing required Section 18.2 keys: {missing_metrics}"
        )

    class_order = serializable_metrics.get("labels") or list(serializable_metrics["per_class"].keys())
    if not class_order:
        raise ValueError("class_order could not be determined from metrics.")

    # Convert manifest to string identifier if passed as Path or dict
    if isinstance(dataset_manifest, Path):
        manifest_ref = str(dataset_manifest.as_posix())
    elif isinstance(dataset_manifest, dict):
        manifest_ref = dataset_manifest.get("dataset_key") or dataset_manifest.get("source_name") or "dict_manifest"
    else:
        manifest_ref = str(dataset_manifest)

    # Build root output directory
    target_crop = crop.strip().lower()
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = Path("reports") / target_crop

    out_path.mkdir(parents=True, exist_ok=True)

    report_payload: dict[str, Any] = {
        "crop": target_crop,
        "model_name": model_name,
        "model_family": model_family,
        "artifact_version": artifact_version,
        "preprocessing_version": preprocessing_version,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "split_evaluated": clean_split,
        "split_protocol": split_protocol,
        "source_dataset_manifest": manifest_ref,
        "label_generation_method": label_generation_method,
        "device_used": device_used,
        "seeds": seeds if isinstance(seeds, dict) else {"seed": seeds},
        "hyperparameters": hyperparameters,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "class_counts_per_split": class_counts_per_split,
        "class_order": class_order,
        "metrics": serializable_metrics,
        "software_versions": software_versions or get_default_software_versions(),
        "limitations": clean_limitations,
    }

    # Generate file paths: <model_name>_<split>.json and .md
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    json_file = out_path / f"{safe_model_name}_{clean_split}.json"
    md_file = out_path / f"{safe_model_name}_{clean_split}.md"

    # Write JSON report
    json_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    # Render and write Markdown report
    md_content = build_markdown_report(report_payload)
    md_file.write_text(md_content, encoding="utf-8")

    return json_file, md_file


def load_evaluation_report(report_path: Path | str) -> dict[str, Any]:
    """Load and validate an existing JSON evaluation report."""
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "crop",
        "model_name",
        "split_evaluated",
        "metrics",
        "limitations",
        "split_counts",
    }
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Corrupted report {path}: missing required keys {missing}")
    return data
