"""Unit tests for repository layout and rules."""
from scripts.verify_folder_structure import (
    check_deviations_logged,
    check_directories,
    check_duplicates,
    check_fusion_readme,
)


def test_required_directories_exist():
    missing, untracked = check_directories()
    assert not missing, f"Missing directories: {missing}"
    assert not untracked, f"Untracked empty directories: {untracked}"


def test_no_duplicate_directories():
    dupes = check_duplicates()
    assert not dupes, f"Found duplicate directories: {dupes}"


def test_fusion_readme_policy():
    ok, msg = check_fusion_readme()
    assert ok, msg


def test_deviations_logged():
    ok, msg = check_deviations_logged()
    assert ok, msg
