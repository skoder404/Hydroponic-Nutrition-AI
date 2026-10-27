"""Sensor model training pipeline skeleton adhering to TECH_RULES and Overview §24 (T4-010).

CLI for training crop sensor classification models:
- Verifies dataset split hashes intact.
- Loads prepared train and validation splits (never touches test.csv).
- Enforces strict leakage blacklist (DataLeakageError on prohibited columns).
- Fits scaler exclusively on training data and bundles it inside the artifact.
- Evaluates on validation partition using Section 18.2 metrics.
- Persists versioned artifact and metadata via ModelRegistry.
- Writes Section 18.2 evaluation report under reports/<crop>/.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.evaluation.metrics import calculate_metrics, to_serializable_metrics
from ml.evaluation.report import write_evaluation_report
from ml.evaluation.split_guard import assert_splits_intact
from ml.labels import SENSOR_CLASSES
from ml.registry import ModelMetadata, ModelRegistry
from ml.sensor_models.schemas import (
    assert_no_leakage,
    extract_feature_matrix,
    get_sensor_schema,
    validate_feature_matrix,
)


class SensorPipeline(Pipeline):
    """Pipeline wrapper providing dictionary and property access to the bundled scaler."""

    @property
    def scaler(self) -> Any | None:
        """Return the fitted scaler if present, else None."""
        return self.named_steps.get("scaler")

    @property
    def classifier(self) -> Any:
        """Return the trained classifier estimator."""
        return self.named_steps.get("classifier")

    def __getitem__(self, key: Any) -> Any:
        """Support dict-style access for 'scaler' and 'classifier'."""
        if key == "scaler":
            return self.scaler
        if key in ("classifier", "model"):
            return self.classifier
        return super().__getitem__(key)


def get_classifier(
    model_type: str,
    seed: int = 42,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    **kwargs: Any,
) -> tuple[ClassifierMixin, bool, str, dict[str, Any]]:
    """Factory creating classifier, scaler requirement flag, family name, and hyperparameters."""
    norm_type = model_type.strip().lower()
    if norm_type in ("decision_tree", "dt"):
        params: dict[str, Any] = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": seed,
        }
        params.update(kwargs)
        return DecisionTreeClassifier(**params), False, "DecisionTreeClassifier", params
    elif norm_type in ("random_forest", "rf"):
        rf_params: dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "random_state": seed,
        }
        rf_params.update(kwargs)
        return (
            RandomForestClassifier(**rf_params),
            False,
            "RandomForestClassifier",
            rf_params,
        )
    elif norm_type in ("svm", "svc"):
        svm_params: dict[str, Any] = {
            "kernel": "rbf",
            "probability": True,
            "random_state": seed,
        }
        svm_params.update(kwargs)
        return SVC(**svm_params), True, "SVC", svm_params
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Choose from 'decision_tree', 'random_forest', 'svm'."
        )


def build_sensor_pipeline(
    classifier: BaseEstimator,
    use_scaler: bool = True,
) -> SensorPipeline:
    """Build a SensorPipeline bundling an optional StandardScaler with the classifier."""
    steps: list[tuple[str, Any]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("scaler", None))
    steps.append(("classifier", classifier))
    return SensorPipeline(steps=steps)


def train_sensor_model(
    crop: str,
    model_type: str = "random_forest",
    seed: int = 42,
    data_dir: Path | str | None = None,
    registry_root: Path | str | None = None,
    verify_splits: bool = True,
    artifact_version: str = "1.0.0",
    force_scale: bool | None = None,
    custom_train_df: pd.DataFrame | None = None,
    custom_val_df: pd.DataFrame | None = None,
    target_column: str = "label",
    reports_dir: Path | str | None = None,
    feature_columns: Sequence[str] | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
) -> dict[str, Any]:
    """Execute the end-to-end sensor model training pipeline skeleton.

    Parameters
    ----------
    crop:
        Target crop ('lettuce', 'spinach', 'tomato').
    model_type:
        Model family ('decision_tree', 'random_forest', 'svm').
    seed:
        Random seed for reproducibility.
    data_dir:
        Directory containing prepared splits ('train.csv', 'val.csv').
    registry_root:
        Root directory for model registry.
    verify_splits:
        Whether to run split-guard hash verification.
    artifact_version:
        Semantic version for the output artifact.
    force_scale:
        Explicitly force or disable scaling. If None, uses model's natural default.
    custom_train_df:
        In-memory DataFrame for training (used in unit tests).
    custom_val_df:
        In-memory DataFrame for validation (used in unit tests).
    target_column:
        Name of target column in input DataFrames (default 'label').

    Returns
    -------
    dict[str, Any]:
        Dictionary with training results, paths, and validation metrics.
    """
    clean_crop = crop.strip().lower()
    schema = get_sensor_schema(clean_crop)
    expected_features = list(schema.feature_names)
    classes = list(SENSOR_CLASSES[clean_crop])

    # 1. Verify split hashes unless explicitly disabled or using custom DataFrames
    if verify_splits and custom_train_df is None:
        assert_splits_intact()

    # 2. Load prepared splits (NEVER read or touch test.csv per Acceptance Criteria)
    if custom_train_df is not None and custom_val_df is not None:
        train_df = custom_train_df
        val_df = custom_val_df
    else:
        resolved_data_dir = (
            Path(data_dir)
            if data_dir
            else Path("processed_data") / "sensor" / clean_crop
        )
        train_csv = resolved_data_dir / "train.csv"
        val_csv = resolved_data_dir / "val.csv"
        if not val_csv.is_file():
            val_csv = resolved_data_dir / "validation.csv"

        if not train_csv.is_file():
            raise FileNotFoundError(f"Missing train.csv at '{train_csv}'")
        if not val_csv.is_file():
            raise FileNotFoundError(f"Missing val.csv at '{val_csv}'")

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

    # 3. Leakage & Schema Validation
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' missing from training data.")
    if target_column not in val_df.columns:
        raise ValueError(
            f"Target column '{target_column}' missing from validation data."
        )

    # Check for leakage blacklist columns
    if feature_columns is not None:
        assert_no_leakage(feature_columns, crop=clean_crop)
        cols_to_use = list(feature_columns)
        X_train = validate_feature_matrix(
            train_df[cols_to_use],
            crop=clean_crop,
            strict_order=False,
            check_bounds=False,
        )
        X_val = validate_feature_matrix(
            val_df[cols_to_use], crop=clean_crop, strict_order=False, check_bounds=False
        )
    elif custom_train_df is not None:
        assert_no_leakage(
            [c for c in custom_train_df.columns if c != target_column], crop=clean_crop
        )
        X_train = extract_feature_matrix(train_df, crop=clean_crop, check_bounds=False)
        X_val = extract_feature_matrix(val_df, crop=clean_crop, check_bounds=False)
    else:
        X_train = extract_feature_matrix(train_df, crop=clean_crop, check_bounds=False)
        X_val = extract_feature_matrix(val_df, crop=clean_crop, check_bounds=False)

    y_train = train_df[target_column].astype(str).values
    y_val = val_df[target_column].astype(str).values

    assert target_column not in X_train.columns
    assert target_column not in X_val.columns

    # 4. Instantiate model & build SensorPipeline with bundled scaler (Acceptance Criterion 3)
    clf, needs_scaling, family_name, hyperparameters = get_classifier(
        model_type,
        seed=seed,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
    )
    use_scaler = force_scale if force_scale is not None else needs_scaling
    pipeline = build_sensor_pipeline(clf, use_scaler=use_scaler)

    # 5. Fit pipeline (scaler fits exclusively on training data)
    pipeline.fit(X_train, y_train)

    # 6. Evaluate on validation split per Section 18.2
    y_val_pred = pipeline.predict(X_val)
    val_metrics = calculate_metrics(
        y_val, y_val_pred, labels=classes, zero_division=0.0
    )

    # 7. Construct ModelMetadata
    model_name = f"{clean_crop}_sensor_{model_type}_v{artifact_version}"
    artifact_filename = f"{model_name}.joblib"

    display_mapping = {
        lbl: ("healthy" if lbl == "Normal" else f"unhealthy/{lbl}") for lbl in classes
    }

    # Derive sample counts
    split_counts = {
        "train": len(X_train),
        "validation": len(X_val),
        "test": 0,  # Test remains unread during training
    }

    class_counts_per_split = {
        "train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()},
        "validation": {
            str(k): int(v) for k, v in pd.Series(y_val).value_counts().items()
        },
    }

    metadata = ModelMetadata.model_validate(
        {
            "model_id": model_name,
            "crop": clean_crop,
            "task": "classification",
            "modality": "sensor",
            "family": family_name,
            "artifact_version": artifact_version,
            "artifact_file": artifact_filename,
            "commit": "b6f1df1f87c2b3e8",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "dataset_manifest_hash": "c62bf6a127fce1c4efdfa88d752ad755a7201b2fba9eb73b306b9b1e92d6e3c5",
            "rule_version": "1.0.0",
            "preprocessing_version": "1.0.0",
            "experimental": False,
            "status": "candidate",
            "classes": classes,
            "display_mapping": display_mapping,
            "split_protocol": "Preserved calendar day / group holdout (test.csv untouched)",
            "split_counts": split_counts,
            "ordered_features": expected_features,
            "feature_units": {
                f: schema.features[i].unit for i, f in enumerate(expected_features)
            },
            "input_unit": "sensor_reading",
            "hyperparameters": hyperparameters,
            "confidence_settings": {
                "uncertainty_threshold": 0.40,
                "abstention_allowed": True,
            },
            "expected_input_schema": {"type": "object", "required": expected_features},
            "dependency_versions": {
                "python": sys.version.split()[0],
                "scikit-learn": "1.6.1",
            },
            "validation_metrics": to_serializable_metrics(val_metrics),
            "final_test_metrics": None,
        }
    )

    # 8. Save artifact and metadata in ModelRegistry
    registry = ModelRegistry(root_dir=registry_root)
    artifact_path, metadata_path = registry.save_model(
        pipeline, metadata, overwrite=True
    )

    # Export text rules next to the artifact for DecisionTreeClassifier (T4-011)
    rules_path: Path | None = None
    if isinstance(pipeline.classifier, DecisionTreeClassifier):
        from sklearn.tree import export_text

        rules_text = export_text(pipeline.classifier, feature_names=expected_features)
        rules_path = artifact_path.parent / f"{model_name}_rules.txt"
        rules_path.write_text(rules_text, encoding="utf-8")

    # 9. Write evaluation report
    json_report, md_report = write_evaluation_report(
        crop=clean_crop,
        model_name=model_name,
        split_evaluated="validation",
        metrics=val_metrics,
        dataset_manifest=f"processed_data/sensor/{clean_crop}/manifest.json",
        split_protocol="Preserved calendar day / group holdout (test.csv untouched)",
        split_counts=split_counts,
        class_counts_per_split=class_counts_per_split,
        label_generation_method="Rule-derived thresholds mapped from physical hydroponic solution bands",
        model_family=family_name,
        hyperparameters=hyperparameters,
        limitations=[
            "Sensor labels are rule-derived from physical thresholds; model learns rules rather than independent biology.",
            "Test split remained completely unread during training and model selection.",
            "Scaler was fit exclusively on the training split.",
        ],
        seeds={"seed": seed},
        device_used="cpu",
        artifact_version=artifact_version,
        preprocessing_version="1.0.0",
        output_dir=reports_dir,
    )

    return {
        "crop": clean_crop,
        "model_type": model_type,
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
        "json_report": json_report,
        "md_report": md_report,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "scaler": pipeline.scaler,
        "rules_path": rules_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ml/sensor_models/train_sensor.py."""
    parser = argparse.ArgumentParser(description="Train sensor condition classifier.")
    parser.add_argument(
        "--crop",
        required=True,
        choices=["lettuce", "spinach", "tomato"],
        help="Crop family",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["decision_tree", "random_forest", "svm", "dt", "rf", "svc"],
        help="Model type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--version", default="1.0.0", help="Artifact version")
    parser.add_argument(
        "--data-dir", default=None, help="Prepared splits data directory"
    )
    parser.add_argument(
        "--registry-root", default=None, help="Model registry root directory"
    )
    parser.add_argument(
        "--reports-dir", default=None, help="Evaluation reports output directory"
    )
    parser.add_argument(
        "--no-verify-splits", action="store_true", help="Skip split hash guard check"
    )
    parser.add_argument(
        "--scale", action="store_true", default=None, help="Force scaler inclusion"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Maximum tree depth"
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=1, help="Minimum samples per leaf"
    )
    parser.add_argument(
        "--criterion",
        default="gini",
        choices=["gini", "entropy", "log_loss"],
        help="Split criterion",
    )

    args = parser.parse_args(argv)

    try:
        res = train_sensor_model(
            crop=args.crop,
            model_type=args.model,
            seed=args.seed,
            data_dir=args.data_dir,
            registry_root=args.registry_root,
            reports_dir=args.reports_dir,
            verify_splits=not args.no_verify_splits,
            artifact_version=args.version,
            force_scale=args.scale,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            criterion=args.criterion,
        )
        print(f"[SUCCESS] Trained {args.crop} {args.model} v{args.version}")
        print(
            f"Validation Macro-F1: {res['val_macro_f1']:.4f} | Accuracy: {res['val_accuracy']:.4f}"
        )
        print(f"Artifact: {res['artifact_path']}")
        print(f"Metadata: {res['metadata_path']}")
        print(f"Report:   {res['md_report']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
