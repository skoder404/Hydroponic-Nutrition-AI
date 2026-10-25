"""Unit tests for gitignore and gitattributes rules (T1-003)."""

from scripts.verify_git_rules import (
    verify_datasets_tracked,
    verify_env_rules,
    verify_gitattributes,
    verify_models_and_processed_data_trackable,
)


def test_env_rules():
    ok, msg = verify_env_rules()
    assert ok, msg


def test_datasets_tracked():
    ok, msg = verify_datasets_tracked()
    assert ok, msg


def test_models_and_processed_data_trackable():
    ok, msg = verify_models_and_processed_data_trackable()
    assert ok, msg


def test_gitattributes():
    ok, msg = verify_gitattributes()
    assert ok, msg
