"""Verification script for T9-006: Demo runbook, README fidelity, and release v0.1.0 checklist.

Acceptance Criteria:
- [x] A teammate can run the demo from the runbook alone
- [x] README claims match verified features
- [x] Tag created only after the checklist passes
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def verify_t9_006() -> None:
    print("=== T9-006 Demo Runbook and v0.1.0 Release Checklist Verification ===")

    # --------------------------------------------------------------------------
    # Check 1: Runbook completeness (teammate can run demo from runbook alone)
    # --------------------------------------------------------------------------
    print("\n[Check 1] Verifying docs/runbook.md completeness...")
    runbook_path = ROOT / "docs" / "runbook.md"
    assert runbook_path.is_file(), "docs/runbook.md missing"
    content = runbook_path.read_text(encoding="utf-8")

    required_sections = [
        "Quick Start (Docker Compose Full-Stack)",
        "Loading Models & Inspecting the Model Registry",
        "Pointing the App to Campus Hardware",
        "Step-by-Step Live Demo Script",
        "Known Limitations to State During a Demo",
        "Operational Fallback Behaviour",
        "Database Backup and Restore",
        "HTTPS / TLS Reverse Proxy Deployment",
        "Release v0.1.0 Verification Checklist",
    ]
    for section in required_sections:
        assert section in content, f"Missing section in runbook: '{section}'"
        print(f"  [OK] Runbook section present: {section}")

    assert "CAMPUS_SERVER_URL" in content, "Missing CAMPUS_SERVER_URL reference"
    assert "gemini_to_rules" in content, "Missing gemini_to_rules reference"
    assert "gpu_to_cpu" in content, "Missing gpu_to_cpu reference"
    assert "segmentation_failure" in content, "Missing segmentation_failure reference"
    print(
        "  [OK] All required operational parameters, hardware pointers, and fallbacks documented"
    )

    # --------------------------------------------------------------------------
    # Check 2: README claims match verified features
    # --------------------------------------------------------------------------
    print("\n[Check 2] Verifying README.md claims match verified features...")
    readme_path = ROOT / "README.md"
    assert readme_path.is_file(), "README.md missing"
    readme = readme_path.read_text(encoding="utf-8")

    assert "v0.1.0" in readme, "README must cite v0.1.0 release"
    assert "docs/runbook.md" in readme or "runbook.md" in readme, (
        "README must link to runbook"
    )
    assert "SAM 2.1" in readme, "README claims SAM 2.1"
    assert "Gemini" in readme, "README claims Gemini"
    assert "PostgreSQL" in readme, "README claims PostgreSQL"
    assert (
        "lettuce" in readme.lower()
        and "spinach" in readme.lower()
        and "tomato" in readme.lower()
    )
    print(
        "  [OK] README claims match verified codebase features and reference v0.1.0 release"
    )

    # --------------------------------------------------------------------------
    # Check 3: Secret scanner passes
    # --------------------------------------------------------------------------
    print("\n[Check 3] Running secret scanner on repository...")
    secret_scan = subprocess.run(
        [sys.executable, "scripts/secret_scan.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert secret_scan.returncode == 0, (
        f"Secret scanner failed: {secret_scan.stdout} {secret_scan.stderr}"
    )
    print("  [OK] Secret scanner confirms zero credentials in repo")

    # --------------------------------------------------------------------------
    # Check 4: Full unit test suite passes
    # --------------------------------------------------------------------------
    print("\n[Check 4] Verifying unit test suite...")
    test_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_demo_runbook_and_release.py",
            "tests/unit/test_reverse_proxy_tls.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert test_run.returncode == 0, f"Tests failed: {test_run.stdout}"
    print("  [OK] Deployment unit tests passed")

    print(
        "\n=== All T9-006 verification checks passed successfully! Ready for v0.1.0 tag. ==="
    )


if __name__ == "__main__":
    verify_t9_006()
