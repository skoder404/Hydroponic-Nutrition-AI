"""Shared training and evaluation pipeline for tabular leaf feature classifiers (T5-014).

Adheres to TECH_RULES and Canonical Overview §18.2, §24.1:
- Scaler is fitted exclusively on training data (never on validation or test).
- Feature ordering strictly matches the crop's canonical specification and is persisted in metadata.
- Evaluates on validation partition only (test.csv is strictly never read during training or tuning).
- Produces Section 18.2 compliant markdown and JSON evaluation reports with per-class support.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.metrics import calculate_metrics, to_serializable_metrics
from ml.evaluation.report import get_default_software_versions, write_evaluation_report
from ml.features.lettuce import LETTUCE_FEATURE_NAMES
from ml.features.spinach import SPINACH_FEATURE_NAMES
from ml.features.tomato import TOMATO_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_features")

CROP_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "lettuce": LETTUCE_FEATURE_NAMES,
    "spinach": SPINACH_FEATURE_NAMES,
    "tomato": TOMATO_FEATURE_NAMES,
}


def get_current_git_commit() -> str:
    """Retrieve current git commit hash for metadata audit."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


class FeaturePipeline(Pipeline):
    """Pipeline bundling a fitted StandardScaler with a trained ClassifierMixin."""

    @property
    def scaler(self) -> StandardScaler | None:
        return self.named_steps.get("scaler")

    @property
    def classifier(self) -> BaseEstimator:
        return self.named_steps.get("classifier")


def load_feature_splits(
    crop: str,
    features_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Load train.csv and val.csv for the specified crop, strictly verifying feature order."""
    if features_dir is None:
        features_dir = PROJECT_ROOT / f"processed_data/features/{crop}"

    train_path = features_dir / "train.csv"
    val_path = features_dir / "val.csv"
    meta_path = features_dir / "metadata.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Feature split files missing in {features_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    canonical_features = list(CROP_FEATURE_NAMES.get(crop, []))
    if not canonical_features and "feature_names" in metadata:
        canonical_features = list(metadata["feature_names"])

    # Ensure all canonical feature columns are present and correctly ordered
    for fname in canonical_features:
        if fname not in train_df.columns:
            raise ValueError(f"Required feature '{fname}' missing from train.csv")
        if fname not in val_df.columns:
            raise ValueError(f"Required feature '{fname}' missing from val.csv")

    LOGGER.info(
        f"Loaded {crop} features: train={len(train_df)}, val={len(val_df)}, features={len(canonical_features)}"
    )
    return train_df, val_df, canonical_features, metadata


def train_and_evaluate_classifier(
    crop: str,
    model_family: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: list[str],
    hyperparameters: dict[str, Any] | None = None,
    output_model_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Train classifier on train partition and evaluate exclusively on validation partition."""
    if hyperparameters is None:
        hyperparameters = {}

    target_col = "class"
    X_train = train_df[feature_names].to_numpy(dtype=np.float64)
    y_train = train_df[target_col].to_numpy()

    X_val = val_df[feature_names].to_numpy(dtype=np.float64)
    y_val = val_df[target_col].to_numpy()

    classes = sorted(list(set(y_train) | set(y_val)))

    # Fit scaler EXCLUSIVELY on training features
    scaler = StandardScaler()
    scaler.fit(X_train)

    if model_family.lower() == "random_forest":
        clf_params = {
            "n_estimators": hyperparameters.get("n_estimators", 150),
            "max_depth": hyperparameters.get("max_depth", 8),
            "min_samples_split": hyperparameters.get("min_samples_split", 3),
            "min_samples_leaf": hyperparameters.get("min_samples_leaf", 2),
            "class_weight": hyperparameters.get("class_weight", "balanced"),
            "random_state": 42,
        }
        classifier = RandomForestClassifier(**clf_params)
        display_name = "RandomForest"
    elif model_family.lower() in ("svm", "svc"):
        clf_params = {
            "C": hyperparameters.get("C", 2.0),
            "gamma": hyperparameters.get("gamma", "scale"),
            "kernel": hyperparameters.get("kernel", "rbf"),
            "class_weight": hyperparameters.get("class_weight", "balanced"),
            "probability": True,
            "random_state": 42,
        }
        classifier = SVC(**clf_params)
        display_name = "SVM"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    # Build and fit pipeline
    pipeline = FeaturePipeline([("scaler", scaler), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)

    # Predict on validation partition exclusively
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)

    # Calculate Section 18.2 metrics (including per-class support and confusion matrix)
    metrics_result = calculate_metrics(
        y_true=y_val,
        y_pred=val_preds,
        labels=classes,
    )
    serializable_metrics = to_serializable_metrics(metrics_result)

    val_macro_f1 = serializable_metrics["macro_f1"]
    val_accuracy = serializable_metrics["accuracy"]
    LOGGER.info(
        f"[{crop.upper()} {display_name}] Validation Macro-F1: {val_macro_f1:.4f} | Accuracy: {val_accuracy:.4f}"
    )

    # Save model artifact and metadata
    artifact_version = "v1.0.0"
    model_name = f"{crop}_feature_{model_family.lower()}_{artifact_version}"

    if output_model_dir is None:
        output_model_dir = PROJECT_ROOT / f"models/image/{crop}/{model_family.lower()}/{artifact_version}"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_model_dir / "model.joblib"
    joblib.dump(pipeline, artifact_path)
    LOGGER.info(f"Saved trained pipeline to {artifact_path}")

    # Class counts and support tracking
    train_class_counts = train_df[target_col].value_counts().to_dict()
    val_class_counts = val_df[target_col].value_counts().to_dict()

    model_metadata = {
        "model_id": f"{crop}_feature_{model_family.lower()}",
        "crop": crop,
        "modality": "feature",
        "family": display_name,
        "artifact_version": artifact_version,
        "artifact_file": str(artifact_path.relative_to(PROJECT_ROOT).as_posix()),
        "commit": get_current_git_commit(),
        "classes": classes,
        "feature_order": feature_names,
        "num_features": len(feature_names),
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
        },
        "class_support_validation": val_class_counts,
        "hyperparameters": clf_params,
        "val_macro_f1": round(float(val_macro_f1), 4),
        "val_accuracy": round(float(val_accuracy), 4),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "software_versions": get_default_software_versions(),
    }

    metadata_file = output_model_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Write Section 18.2 Evaluation Report
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / f"reports/{crop}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    limitations = [
        "Features are extracted from segmented lettuce leaf crops derived from whole-plant imagery.",
        "Ground truth labels inherit the whole-plant condition (weak labels); an estimated 15.8% of leaf crops on deficient plants present normal green morphology.",
        "Model performance is evaluated exclusively on the validation holdout split; test.csv remains untouched.",
    ]

    write_evaluation_report(
        crop=crop,
        model_name=f"{crop}_feature_{model_family.lower()}_{artifact_version}",
        model_family=display_name,
        artifact_version=artifact_version,
        split_evaluated="validation",
        metrics=serializable_metrics,
        dataset_manifest=f"processed_data/features/{crop}",
        split_protocol="Stratified split inherited from parent plant images; train/val isolated",
        split_counts={"train": len(train_df), "validation": len(val_df), "test": 0},
        class_counts_per_split={
            "train": train_class_counts,
            "validation": val_class_counts,
        },
        label_generation_method="Whole-plant agronomic condition inherited by individual leaf crops (weak label)",
        hyperparameters=clf_params,
        limitations=limitations,
        output_dir=reports_dir,
    )

    return {
        "model_family": display_name,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_file),
        "metrics": serializable_metrics,
        "val_macro_f1": val_macro_f1,
        "val_accuracy": val_accuracy,
        "metadata": model_metadata,
    }


def write_comparison_report(
    crop: str,
    results: list[dict[str, Any]],
    reports_dir: Path | None = None,
) -> Path:
    """Generate comparative markdown and JSON report across evaluated models."""
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / f"reports/{crop}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Find best model by validation macro-F1
    best_res = max(results, key=lambda r: r["val_macro_f1"])

    comp_data = {
        "crop": crop,
        "evaluation_split": "validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model": {
            "family": best_res["model_family"],
            "val_macro_f1": round(float(best_res["val_macro_f1"]), 4),
            "val_accuracy": round(float(best_res["val_accuracy"]), 4),
        },
        "models": [
            {
                "family": r["model_family"],
                "val_macro_f1": round(float(r["val_macro_f1"]), 4),
                "val_accuracy": round(float(r["val_accuracy"]), 4),
                "macro_precision": round(float(r["metrics"]["macro_precision"]), 4),
                "macro_recall": round(float(r["metrics"]["macro_recall"]), 4),
                "per_class": r["metrics"]["per_class"],
                "confusion_matrix": r["metrics"]["confusion_matrix"],
            }
            for r in results
        ],
    }

    json_path = reports_dir / f"{crop}_feature_experiments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)

    # Build Markdown table
    md_lines = [
        f"# Feature Classification Comparison Report: {crop.capitalize()}",
        "",
        "> **Split Evaluated:** `validation` (strictly isolated from test holdout)  ",
        f"> **Generated:** `{comp_data['generated_at']}`  ",
        f"> **Selected Champion Model:** `{best_res['model_family']}` (Val Macro-F1: `{best_res['val_macro_f1']:.4f}`)  ",
        "",
        "## 1. Model Family Comparison Table",
        "",
        "| Model Family | Val Macro-F1 | Val Accuracy | Macro Precision | Macro Recall | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in results:
        fam = r["model_family"]
        f1 = r["val_macro_f1"]
        acc = r["val_accuracy"]
        prec = r["metrics"]["macro_precision"]
        rec = r["metrics"]["macro_recall"]
        status = "**CHAMPION**" if fam == best_res["model_family"] else "Candidate"
        md_lines.append(f"| **{fam}** | **{f1:.4f}** | {acc:.4f} | {prec:.4f} | {rec:.4f} | {status} |")

    md_lines.extend([
        "",
        "## 2. Per-Class Metrics & Support (Validation Partition)",
        "",
    ])

    for r in results:
        fam = r["model_family"]
        md_lines.extend([
            f"### {fam} Per-Class Performance",
            "",
            "| Class | Precision | Recall | F1-Score | Support (Validation Instances) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        per_class = r["metrics"]["per_class"]
        for cls_name, p_metrics in per_class.items():
            md_lines.append(
                f"| `{cls_name}` | {p_metrics['precision']:.4f} | {p_metrics['recall']:.4f} | "
                f"{p_metrics['f1']:.4f} | {p_metrics['support']} |"
            )
        md_lines.append("")

    md_lines.extend([
        "## 3. Confusion Matrices",
        "",
    ])

    for r in results:
        fam = r["model_family"]
        md_lines.append(f"#### {fam} Confusion Matrix")
        md_lines.append("```json")
        md_lines.append(json.dumps(r["metrics"]["confusion_matrix"], indent=2))
        md_lines.append("```")
        md_lines.append("")

    md_lines.extend([
        "## 4. Limitations & Weak Label Note",
        "",
        "- Training and validation features were extracted from individual leaf crops generated by SAM 2.1 from whole-plant images (`image_data/Lettuce_Images/`).",
        "- Because whole-plant nutrient deficiencies manifest gradually across the canopy, roughly 15.8% of leaf crops on deficient plants present normal green morphology (documented in `reports/analysis/crop_label_review.md`).",
        "- Test holdout partition (`test.csv`) was strictly never evaluated during training or model selection.",
    ])

    md_path = reports_dir / f"{crop}_feature_experiments.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    LOGGER.info(f"Saved comparison report to {md_path}")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate leaf feature classifiers (T5-014)")
    parser.add_argument("--crop", default="lettuce", choices=["lettuce", "spinach", "tomato"], help="Crop to train on")
    parser.add_argument("--families", nargs="+", default=["random_forest", "svm"], help="Model families to train")
    args = parser.parse_args()

    train_df, val_df, feature_names, _ = load_feature_splits(args.crop)

    results = []
    for fam in args.families:
        LOGGER.info(f"Training {fam} on {args.crop}...")
        res = train_and_evaluate_classifier(
            crop=args.crop,
            model_family=fam,
            train_df=train_df,
            val_df=val_df,
            feature_names=feature_names,
        )
        results.append(res)

    write_comparison_report(args.crop, results)
    LOGGER.info("Training and comparative reporting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
