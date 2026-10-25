"""Unit tests for trusted model-path guard and safe loader (T3-007)."""
from scripts.verify_paths_guard import (
    check_loader_uses_guard,
    check_outside_path_rejected,
    check_symlink_escape_rejected,
)


def test_outside_model_path_rejected():
    ok, msg = check_outside_path_rejected()
    assert ok, msg


def test_model_path_symlink_escape_rejected():
    ok, msg = check_symlink_escape_rejected()
    assert ok, msg


def test_safe_model_loader_uses_guard():
    ok, msg = check_loader_uses_guard()
    assert ok, msg
