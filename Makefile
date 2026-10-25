.PHONY: lint type-check format secret-scan pre-commit-install pre-commit-run all-checks

## Python linting (ruff)
lint:
	python -m ruff check backend ml scripts tests

## Python type checking (mypy)
type-check:
	python -m mypy backend --ignore-missing-imports

## Python auto-format (ruff)
format:
	python -m ruff format backend ml scripts tests

## Secret scanning
secret-scan:
	python scripts/secret_scan.py

## Install pre-commit hooks
pre-commit-install:
	pre-commit install

## Run all pre-commit hooks on all files
pre-commit-run:
	pre-commit run --all-files

## Run all checks (lint + type-check + secret-scan)
all-checks: lint type-check secret-scan
	@echo "All checks passed!"
