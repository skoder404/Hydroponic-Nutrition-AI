"""Unit tests for spinach image and sensor provenance sheets (T2-003)."""
from scripts.verify_spinach_provenance import (
    check_canonical_url,
    check_column_table_and_tds,
    check_files_exist,
)


def test_spinach_provenance_files_exist():
    ok, msg = check_files_exist()
    assert ok, msg


def test_spinach_canonical_url():
    ok, msg = check_canonical_url()
    assert ok, msg


def test_spinach_column_table_and_tds():
    ok, msg = check_column_table_and_tds()
    assert ok, msg
