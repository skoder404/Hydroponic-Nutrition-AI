"""Unit tests for split manifest hash guard (T2-019)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.evaluation.split_guard import (
    DEFAULT_HASH_FILE,
    SplitTamperError,
    assert_splits_intact,
    compute_file_sha256,
    verify_split_hashes,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def test_default_hash_file_exists_and_valid() -> None:
    """Verify processed_data/split_hashes.json exists and has valid structure."""
    assert DEFAULT_HASH_FILE.is_file(), f"Missing hash file: {DEFAULT_HASH_FILE}"
    content = json.loads(DEFAULT_HASH_FILE.read_text(encoding="utf-8"))
    assert content.get("version") == "1.0"
    assert content.get("total_files") >= 19
    files = content.get("files", {})
    assert len(files) >= 19

    # Verify key expected split files are tracked
    expected_tracked = [
        "processed_data/sensor/lettuce/train.csv",
        "processed_data/sensor/lettuce/val.csv",
        "processed_data/sensor/lettuce/test.csv",
        "processed_data/sensor/spinach/train.csv",
        "processed_data/sensor/spinach/val.csv",
        "processed_data/sensor/spinach/test.csv",
        "processed_data/sensor/tomato/train.csv",
        "processed_data/sensor/tomato/val.csv",
        "processed_data/sensor/tomato/test.csv",
        "processed_data/splits/image_splits_manifest.json",
    ]
    for exp in expected_tracked:
        assert exp in files, f"Expected tracked file {exp} missing from hash file"


def test_verify_split_hashes_passes_on_repo() -> None:
    """Verify that current workspace split files match hashes perfectly."""
    is_valid, errors = verify_split_hashes(DEFAULT_HASH_FILE, raise_on_failure=False)
    assert is_valid, f"Verification failed with errors: {errors}"
    assert len(errors) == 0


def test_assert_splits_intact_does_not_raise_on_valid_repo() -> None:
    """Verify assert_splits_intact passes without exception."""
    assert_splits_intact()
    assert_splits_intact(crop="lettuce", modality="sensor")
    assert_splits_intact(crop="tomato", modality="sensor")
    assert_splits_intact(crop="spinach", modality="sensor")


def test_tamper_detection_triggers_error(tmp_path: Path) -> None:
    """Verify altering a split file causes verification to fail and raises SplitTamperError."""
    fake_split = tmp_path / "mock.csv"
    fake_split.write_text("x,y\n1,2\n", encoding="utf-8")

    hash_file = tmp_path / "split_hashes.json"
    hashes = {"mock.csv": compute_file_sha256(fake_split)}
    hash_file.write_text(json.dumps({"version": "1.0", "files": hashes}), encoding="utf-8")

    # Initial verification passes
    is_valid, errors = verify_split_hashes(hash_file, root=tmp_path, raise_on_failure=False)
    assert is_valid
    assert len(errors) == 0

    # Tamper with file
    fake_split.write_text("x,y\n1,999\n", encoding="utf-8")

    is_valid, errors = verify_split_hashes(hash_file, root=tmp_path, raise_on_failure=False)
    assert not is_valid
    assert "mock.csv" in errors

    with pytest.raises(SplitTamperError, match="Split integrity violation"):
        assert_splits_intact(hash_file=hash_file, root=tmp_path)


def test_missing_split_file_triggers_error(tmp_path: Path) -> None:
    """Verify missing split file triggers error."""
    fake_split = tmp_path / "mock.csv"
    fake_split.write_text("x,y\n1,2\n", encoding="utf-8")

    hash_file = tmp_path / "split_hashes.json"
    hashes = {"mock.csv": compute_file_sha256(fake_split)}
    hash_file.write_text(json.dumps({"version": "1.0", "files": hashes}), encoding="utf-8")

    # Remove file
    fake_split.unlink()

    is_valid, errors = verify_split_hashes(hash_file, root=tmp_path, raise_on_failure=False)
    assert not is_valid
    assert "mock.csv" in errors
    assert "MISSING" in errors["mock.csv"]

    with pytest.raises(SplitTamperError, match="Split integrity violation"):
        verify_split_hashes(hash_file, root=tmp_path, raise_on_failure=True)
