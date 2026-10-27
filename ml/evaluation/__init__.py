"""Evaluation utilities and split guard module."""

from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_metrics,
    to_serializable_metrics,
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
    "DEFAULT_HASH_FILE",
    "SplitTamperError",
    "assert_splits_intact",
    "calculate_classification_metrics",
    "calculate_metrics",
    "compute_file_sha256",
    "generate_split_hashes",
    "to_serializable_metrics",
    "verify_split_hashes",
]
