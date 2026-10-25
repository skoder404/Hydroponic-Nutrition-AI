"""Unit tests for docker-compose configuration and rules (T1-011)."""
from scripts.verify_docker_compose import (
    check_compose_file_and_secrets,
    check_env_example_database_match,
)


def test_docker_compose_structure_and_no_committed_secrets():
    ok, msg = check_compose_file_and_secrets()
    assert ok, msg


def test_env_example_database_match():
    ok, msg = check_env_example_database_match()
    assert ok, msg
