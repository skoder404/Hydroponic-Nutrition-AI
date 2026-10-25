"""Tests for GitHub Actions CI skeleton (T1-013).

Verifies:
1. Workflow YAML schema, structure, triggers, jobs, and caching.
2. Zero-secret requirement (no secrets needed).
3. Intentional lint failure verification (guaranteeing CI fails on bad code).
4. Heavy training / SAM exclusion in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_file_exists() -> None:
    """CI workflow file must exist at .github/workflows/ci.yml."""
    assert CI_WORKFLOW.is_file(), f"Expected {CI_WORKFLOW} to exist"


def test_ci_workflow_valid_yaml() -> None:
    """CI workflow must parse as valid YAML."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "ci.yml must parse to a dictionary"

    # Triggers
    triggers = data.get("on") if "on" in data else data.get(True)
    assert triggers is not None
    assert "push" in triggers
    assert "pull_request" in triggers

    push_branches = triggers["push"].get("branches", [])
    assert "main" in push_branches


def test_ci_workflow_jobs_and_caching() -> None:
    """Workflow must contain secret-scan, backend, and frontend jobs with caching."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    jobs = data.get("jobs", {})

    assert "secret-scan" in jobs, "Must have secret-scan job"
    assert "backend" in jobs, "Must have backend job"
    assert "frontend" in jobs, "Must have frontend job"

    # Check caching
    backend_steps_str = yaml.dump(jobs["backend"].get("steps", []))
    assert "cache: pip" in backend_steps_str or "cache: 'pip'" in backend_steps_str or '"pip"' in backend_steps_str

    frontend_steps_str = yaml.dump(jobs["frontend"].get("steps", []))
    assert "cache: npm" in frontend_steps_str or "cache: 'npm'" in frontend_steps_str or '"npm"' in frontend_steps_str


def test_ci_workflow_no_secrets_required() -> None:
    """Workflow must not require or reference any repository secrets."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in content, "CI workflow must not reference any GitHub secrets"
    assert "SECRETS." not in content


def test_ci_workflow_excludes_heavy_training_and_sam() -> None:
    """CI must exclude heavy GPU training and SAM model downloads."""
    content = CI_WORKFLOW.read_text(encoding="utf-8")
    # Must use FORCE_CPU
    assert "FORCE_CPU" in content
    # Must not invoke heavy training commands
    assert "train_image_model" not in content
    assert "sam2.1_hiera_large" not in content


def test_ci_lint_fails_on_intentional_lint_error(tmp_path: Path) -> None:
    """Workflow verification: Ruff must fail when run on an intentional lint violation."""
    bad_code = "import os\nimport sys\ndef bad_fn():\n  x = 1\n"
    bad_file = tmp_path / "bad_syntax.py"
    bad_file.write_text(bad_code, encoding="utf-8")

    # Run ruff check on the temporary bad file
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "Ruff must fail on intentional unused imports and unused variables"
    assert "F401" in result.stdout or "F841" in result.stdout or result.returncode != 0
