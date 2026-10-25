"""Unit tests for learning scaffold and progress tracker (T1-015)."""
from scripts.verify_learning_scaffold import (
    check_concept_template,
    check_progress_tracker,
    check_teachback_template,
)


def test_concept_template_structure():
    ok, msg = check_concept_template()
    assert ok, msg


def test_teachback_template_structure():
    ok, msg = check_teachback_template()
    assert ok, msg


def test_progress_tracker_contents_and_columns():
    ok, msg = check_progress_tracker()
    assert ok, msg
