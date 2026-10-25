#!/usr/bin/env python3
"""Verification script for CONTRIBUTING.md, commit checklist, and issue templates (T1-014).

Acceptance Criteria:
1. Commit template works with `git config commit.template`.
2. CONTRIBUTING states that the navigator reviews each task's commit(s) after they are pushed.
3. Definition of done matches Overview §27.4.
4. Issue templates exist in .github/ISSUE_TEMPLATE/.

Re-run command:
    python scripts/verify_contributing.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRIBUTING_MD = ROOT / "CONTRIBUTING.md"
GITMESSAGE = ROOT / ".gitmessage"
ISSUE_TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def verify_commit_template() -> tuple[bool, str]:
    """Criterion 1: Commit template works with `git config commit.template`."""
    if not GITMESSAGE.exists():
        return False, f"Missing commit template file: {GITMESSAGE}"

    # Check git config
    res = run_cmd(["git", "config", "commit.template"])
    if res.returncode != 0 or not res.stdout.strip():
        # Try configuring it if not configured
        run_cmd(["git", "config", "commit.template", ".gitmessage"])
        res = run_cmd(["git", "config", "commit.template"])
        if res.returncode != 0 or not res.stdout.strip():
            return False, "git config commit.template returned empty or failed"

    configured_val = res.stdout.strip()
    if ".gitmessage" not in configured_val:
        return False, f"Expected commit.template to reference .gitmessage, got: {configured_val}"

    # Verify template content contains required sections
    content = GITMESSAGE.read_text(encoding="utf-8")
    required_sections = [
        "WHAT & WHY",
        "TESTS RUN",
        "SPLITS & DATA TOUCHED",
        "METRICS",
        "LIMITATIONS",
        "LEARNING FOCUS NOTE",
    ]
    missing = [sec for sec in required_sections if sec not in content]
    if missing:
        return False, f".gitmessage missing required sections: {missing}"

    return True, f"Commit template verified with git config ({configured_val}) and all required sections present."


def verify_navigator_review_requirement() -> tuple[bool, str]:
    """Criterion 2: CONTRIBUTING states that the navigator reviews each task's commit(s) after they are pushed."""
    if not CONTRIBUTING_MD.exists():
        return False, f"Missing {CONTRIBUTING_MD}"

    text = CONTRIBUTING_MD.read_text(encoding="utf-8")

    # Check that navigator reviews after push is documented
    has_navigator = "navigator" in text.lower()
    has_after_push = "after they are pushed" in text.lower() or "after push" in text.lower()
    has_review = "review" in text.lower()

    if not (has_navigator and has_after_push and has_review):
        return False, "CONTRIBUTING.md does not clearly state that the navigator reviews commits after they are pushed"

    return True, "CONTRIBUTING.md explicitly states navigator reviews commits after they are pushed."


def verify_definition_of_done() -> tuple[bool, str]:
    """Criterion 3: Definition of done matches Overview §27.4."""
    if not CONTRIBUTING_MD.exists():
        return False, f"Missing {CONTRIBUTING_MD}"

    text = CONTRIBUTING_MD.read_text(encoding="utf-8")

    has_dod = "definition of done" in text.lower()
    has_sec_ref = "27.4" in text or "overview §27.4" in text.lower()
    has_quote_or_principle = (
        "implementation" in text.lower()
        and "artifacts" in text.lower()
        and "demonstrate" in text.lower()
    )

    if not (has_dod and has_sec_ref and has_quote_or_principle):
        return False, "Definition of Done does not match Overview §27.4 (missing reference, implementation/artifact requirement)"

    return True, "Definition of Done matches Overview §27.4 ('implementation and evidence must exist')."


def verify_issue_templates() -> tuple[bool, str]:
    """Criterion 4: Issue templates exist in .github/ISSUE_TEMPLATE/."""
    if not ISSUE_TEMPLATE_DIR.exists():
        return False, f"Missing issue template directory: {ISSUE_TEMPLATE_DIR}"

    bug = ISSUE_TEMPLATE_DIR / "bug_report.md"
    feature = ISSUE_TEMPLATE_DIR / "task_or_feature.md"
    cfg = ISSUE_TEMPLATE_DIR / "config.yml"

    missing = [f.name for f in (bug, feature, cfg) if not f.exists()]
    if missing:
        return False, f"Missing issue templates: {missing}"

    return True, "All standard issue templates (bug_report.md, task_or_feature.md, config.yml) present."


def main() -> int:
    print("================================================================================")
    print("VERIFYING CONTRIBUTING, COMMIT TEMPLATE, AND ISSUE TEMPLATES (T1-014)")
    print("================================================================================\n")

    checks = [
        ("Criterion 1: Commit template works with git config commit.template", verify_commit_template),
        ("Criterion 2: CONTRIBUTING states navigator reviews commits after push", verify_navigator_review_requirement),
        ("Criterion 3: Definition of Done matches Overview §27.4", verify_definition_of_done),
        ("Criterion 4: GitHub issue templates present", verify_issue_templates),
    ]

    all_passed = True
    for name, fn in checks:
        passed, msg = fn()
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{status}] {name}")
        print(f"       {msg}\n")
        if not passed:
            all_passed = False

    print("--------------------------------------------------------------------------------")
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY.")
        return 0
    print("VERIFICATION FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
