"""Unit tests for dataset provenance template (T2-001)."""
from scripts.verify_dataset_template import check_template


def test_dataset_provenance_template():
    ok, msg = check_template()
    assert ok, msg
