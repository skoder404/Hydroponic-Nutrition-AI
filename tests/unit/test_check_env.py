"""Unit tests for environment diagnostic reporter and ADR 0002 appendices (T1-006)."""
from scripts.verify_check_env import (
    check_adr_appendices,
    check_default_execution_honesty,
    check_force_cpu_override,
)


def test_check_env_device_honesty():
    ok, msg = check_default_execution_honesty()
    assert ok, msg


def test_check_env_force_cpu_override():
    ok, msg = check_force_cpu_override()
    assert ok, msg


def test_check_env_adr_appendices_saved():
    ok, msg = check_adr_appendices()
    assert ok, msg
