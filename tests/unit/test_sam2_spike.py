"""Unit tests for SAM 2.1 feasibility spike and ADR 0003 (T1-007)."""
from scripts.verify_sam2_spike import (
    check_adr_0003,
    check_metrics_report,
    check_sam2_checkpoint_exists,
)


def test_sam2_metrics_report():
    ok, msg = check_metrics_report()
    assert ok, msg


def test_sam2_adr_documentation():
    ok, msg = check_adr_0003()
    assert ok, msg


def test_sam2_checkpoint():
    ok, msg = check_sam2_checkpoint_exists()
    assert ok, msg
