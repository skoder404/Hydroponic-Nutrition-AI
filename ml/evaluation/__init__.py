"""Evaluation utilities and split guard module."""

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
    "compute_file_sha256",
    "generate_split_hashes",
    "verify_split_hashes",
]
