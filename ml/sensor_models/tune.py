"""Hyperparameter search pipeline with grouped cross-validation (T4-015).

Implements grid search per model family using group-aware CV wrappers (ADR 0006)
and macro-F1 optimization. Search spaces are loaded from versioned YAML files
under ml/sensor_models/configs/. Test split is NEVER loaded or evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from ml.evaluation.cv import GuardedGroupKFold, evaluate_grouped_cv
from ml.labels import SENSOR_CLASSES
from ml.sensor_models.schemas import extract_feature_matrix
from ml.sensor_models.train_sensor import build_sensor_pipeline, get_classifier


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load and validate a versioned hyperparameter search YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at '{cfg_file}'")

    with open(cfg_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Invalid YAML content in '{cfg_file}'. Expected mapping.")

    required_keys = ["version", "model_family", "search_space"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config '{cfg_file}' missing required keys: {missing}")

    return config


def resolve_groups(
    df: pd.DataFrame,
    custom_groups: Sequence[Any] | np.ndarray | None = None,
) -> np.ndarray:
    """Extract or construct strict group identifiers for group-aware cross-validation.

    Priority:
    1. Explicit custom_groups parameter.
    2. 'bench' + 'source_timestamp_EDT' (spinach 6-hour temporal block holdout).
    3. 'experiment' column (physical hydroponic batch/tank).
    4. 'date' column (calendar day holdout).
    5. 'day_index' column.
    """
    if custom_groups is not None:
        return np.asarray(custom_groups)

    if "bench" in df.columns and "source_timestamp_EDT" in df.columns:
        ts = pd.to_datetime(df["source_timestamp_EDT"])
        blocks = (
            df["bench"].astype(str)
            + "|"
            + ts.dt.strftime("%Y-%m-%d")
            + "|"
            + (ts.dt.hour // 6).astype(str)
        )
        return blocks.to_numpy()

    if "source_timestamp" in df.columns:
        return df["source_timestamp"].astype(str).str[:7].to_numpy()

    for col in ["experiment", "date", "day_index"]:
        if col in df.columns:
            return df[col].astype(str).to_numpy()

    # Fallback for synthetic/unit-test datasets: partition into contiguous blocks
    n_samples = len(df)
    n_blocks = min(5, max(2, n_samples // 10))
    return np.repeat(np.arange(n_blocks), int(np.ceil(n_samples / n_blocks)))[
        :n_samples
    ]


def tune_sensor_model(
    crop: str,
    model_type: str,
    config_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    n_splits: int = 5,
    seed: int = 42,
    custom_df: pd.DataFrame | None = None,
    custom_groups: Sequence[Any] | np.ndarray | None = None,
    target_column: str = "label",
    override_grid: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute hyperparameter grid search optimizing validation macro-F1 under Grouped CV.

    Parameters
    ----------
    crop:
        Target crop ('lettuce', 'spinach', 'tomato').
    model_type:
        Model family ('decision_tree', 'random_forest', 'svm', 'knn', 'logistic_regression').
    config_path:
        Optional path to YAML search space. Defaults to ml/sensor_models/configs/<model_type>.yaml.
    data_dir:
        Directory containing prepared splits ('train.csv'). Never touches test.csv.
    output_dir:
        Directory where tuning JSON report is written.
    n_splits:
        Number of group-aware folds for GuardedGroupKFold.
    seed:
        Master random seed for reproducibility.
    custom_df:
        In-memory training DataFrame for testing.
    custom_groups:
        Optional explicit group array matching rows.
    target_column:
        Name of class target column.
    override_grid:
        Optional custom parameter list for testing.

    Returns
    -------
    dict[str, Any]:
        Dictionary with best parameters, best CV score, runtime, and output path.
    """
    start_time = time.perf_counter()
    clean_crop = crop.strip().lower()
    classes = list(SENSOR_CLASSES[clean_crop])

    # 1. Resolve configuration & search space
    if config_path is None:
        norm_name = model_type.strip().lower()
        if norm_name in ("dt", "decision_tree"):
            fname = "decision_tree.yaml"
        elif norm_name in ("rf", "random_forest"):
            fname = "random_forest.yaml"
        elif norm_name in ("svc", "svm"):
            fname = "svm.yaml"
        elif norm_name in ("kneighbors", "knn"):
            fname = "knn.yaml"
        elif norm_name in ("lr", "logreg", "logistic_regression"):
            fname = "logistic_regression.yaml"
        else:
            fname = f"{norm_name}.yaml"
        cfg_file = Path(__file__).parent / "configs" / fname
    else:
        cfg_file = Path(config_path)

    config = load_config(cfg_file)
    config_version = config.get("version", "1.0.0")
    search_space = config.get("search_space", {})

    # 2. Build parameter grid
    if override_grid is not None:
        param_grid = override_grid
    else:
        param_grid = list(ParameterGrid(search_space))

    if not param_grid:
        raise ValueError(
            f"Empty parameter grid derived from search space: {search_space}"
        )

    # 3. Load training data (NEVER touch or read test.csv!)
    if custom_df is not None:
        train_df = custom_df.copy()
    else:
        resolved_data_dir = (
            Path(data_dir)
            if data_dir
            else Path("processed_data") / "sensor" / clean_crop
        )
        train_csv = resolved_data_dir / "train.csv"
        if not train_csv.is_file():
            raise FileNotFoundError(f"Missing train.csv at '{train_csv}'")
        # Ensure test.csv is never accidentally accessed
        train_df = pd.read_csv(train_csv)

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' missing from training data.")

    # 4. Extract features and group identifiers
    groups = resolve_groups(train_df, custom_groups=custom_groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        n_splits = max(2, len(unique_groups))

    # Verify splitter can partition groups without error
    _ = GuardedGroupKFold(n_splits=n_splits)

    X = extract_feature_matrix(train_df, crop=clean_crop, check_bounds=False)
    y = train_df[target_column].astype(str).values

    # 5. Grid Search Loop over Candidate Hyperparameters
    all_results: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_macro_f1 = -1.0
    best_std_macro_f1 = 0.0

    for candidate_params in param_grid:
        tune_kwargs: dict[str, Any] = dict(candidate_params)
        if model_type.lower() in ("svm", "svc") and "probability" not in tune_kwargs:
            tune_kwargs["probability"] = False

        # Build classifier with candidate hyperparameters
        clf, needs_scaling, _family_name, _ = get_classifier(
            model_type=model_type,
            seed=seed,
            **tune_kwargs,
        )
        pipeline = build_sensor_pipeline(clf, use_scaler=needs_scaling)

        # Run Group-Aware Cross-Validation
        cv_res = evaluate_grouped_cv(
            estimator=pipeline,
            X=X,
            y=y,
            groups=groups,
            labels=classes,
            n_splits=n_splits,
            oversample_train=False,  # Evaluates natural candidate robustness
            random_state=seed,
        )

        mean_f1 = cv_res["mean_macro_f1"]
        std_f1 = cv_res["std_macro_f1"]

        result_entry = {
            "params": candidate_params,
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "mean_accuracy": round(cv_res["mean_accuracy"], 4),
            "std_accuracy": round(cv_res["std_accuracy"], 4),
        }
        all_results.append(result_entry)

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_std_macro_f1 = std_f1
            best_params = candidate_params

    runtime_seconds = round(time.perf_counter() - start_time, 2)

    # 6. Save Tuning Results as JSON
    out_dir = Path(output_dir) if output_dir else Path("reports") / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{clean_crop}_{model_type}_tuning.json"

    tuning_payload = {
        "crop": clean_crop,
        "model_family": model_type,
        "config_version": config_version,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "n_splits": n_splits,
        "n_candidates": len(param_grid),
        "best_params": best_params,
        "best_macro_f1": round(best_macro_f1, 4),
        "best_std_macro_f1": round(best_std_macro_f1, 4),
        "all_results": all_results,
        "search_space": search_space,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    out_file.write_text(json.dumps(tuning_payload, indent=2), encoding="utf-8")

    return {
        "crop": clean_crop,
        "model_type": model_type,
        "best_params": best_params,
        "best_macro_f1": best_macro_f1,
        "best_std_macro_f1": best_std_macro_f1,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "output_path": out_file,
        "all_results": all_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Run group-aware hyperparameter grid search on sensor models."
    )
    parser.add_argument(
        "--crop",
        required=True,
        choices=["lettuce", "spinach", "tomato"],
        help="Target crop",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "decision_tree",
            "random_forest",
            "svm",
            "knn",
            "logistic_regression",
            "dt",
            "rf",
            "svc",
        ],
        help="Model family to tune",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to YAML search space configuration",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Prepared dataset directory containing train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where tuning JSON report is written",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of grouped folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args(argv)

    try:
        res = tune_sensor_model(
            crop=args.crop,
            model_type=args.model,
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        print(f"[SUCCESS] Tuned {args.crop} {args.model}")
        print(
            f"Best Macro-F1: {res['best_macro_f1']:.4f} (+/- {res['best_std_macro_f1']:.4f})"
        )
        print(f"Best Params:   {res['best_params']}")
        print(f"Runtime:       {res['runtime_seconds']}s")
        print(f"Report:        {res['output_path']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Tuning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
