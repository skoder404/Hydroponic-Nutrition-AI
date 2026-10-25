"""Unit tests for typed settings loader and .env.example (T1-004)."""
from scripts.verify_settings import (
    check_env_example,
    check_secret_redaction,
    check_settings_load_without_api_key,
)


def test_env_example_placeholders():
    ok, msg = check_env_example()
    assert ok, msg


def test_settings_load_without_gemini_api_key():
    ok, msg = check_settings_load_without_api_key()
    assert ok, msg


def test_settings_secret_redaction():
    ok, msg = check_secret_redaction()
    assert ok, msg
