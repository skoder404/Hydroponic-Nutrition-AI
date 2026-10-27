"""Evaluation utilities, split guard, and cross-validation module."""

from ml.evaluation.cv import (
    CVValidationError,
    GroupOverlapError,
    GuardedGroupKFold,
    assert_no_group_overlap,
    evaluate_grouped_cv,
    get_fold_data,
    iterate_grouped_folds,
    oversample_minority_classes,
)
from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_metrics,
    to_serializable_metrics,
)
from ml.evaluation.report import (
    build_markdown_report,
    load_evaluation_report,
    write_evaluation_report,
)
from ml.evaluation.split_guard import (
    DEFAULT_HASH_FILE,
    SplitTamperError,
    assert_splits_intact,
    compute_file_sha256,
    generate_split_hashes,
    verify_split_hashes,
)

__all__ = [
    "CVValidationError",
    "DEFAULT_HASH_FILE",
    "GroupOverlapError",
    "GuardedGroupKFold",
    "SplitTamperError",
    "assert_no_group_overlap",
    "assert_splits_intact",
    "build_markdown_report",
    "calculate_classification_metrics",
    "calculate_metrics",
    "compute_file_sha256",
    "evaluate_grouped_cv",
    "generate_split_hashes",
    "get_fold_data",
    "iterate_grouped_folds",
    "load_evaluation_report",
    "oversample_minority_classes",
    "to_serializable_metrics",
    "verify_split_hashes",
    "write_evaluation_report",
]
