#!/usr/bin/env python3
"""Verification script for T1-008: Gemini API smoke test and configuration.

Validates:
1. scripts/gemini_smoke_test.py exists.
2. Missing key produces clean 'not configured' message and exits with code 0.
3. Secret key is never leaked in output when configured with a test token.
4. docs/decisions/0004-gemini-config.md exists, is Accepted, and documents
   model name, SDK, and live call / unverified fallback status.
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_script_exists() -> bool:
    script_path = PROJECT_ROOT / "scripts" / "gemini_smoke_test.py"
    if not script_path.exists():
        print(f"[FAIL] Missing {script_path}")
        return False
    print("[PASS] scripts/gemini_smoke_test.py exists.")
    return True


def check_missing_key_behavior() -> bool:
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = ""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gemini_smoke_test.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if result.returncode != 0:
        print(f"[FAIL] Expected exit code 0 when key is missing, got {result.returncode}")
        print(f"Stderr: {result.stderr}")
        return False

    output = result.stdout
    if "not configured" not in output.lower():
        print(f"[FAIL] Expected 'not configured' message in stdout, got:\n{output}")
        return False

    print("[PASS] Missing key exits cleanly with code 0 and clear message.")
    return True


def check_secret_redaction() -> bool:
    secret_probe = "SUPER_SECRET_CANARY_TOKEN_ABC123XYZ"
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = secret_probe

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gemini_smoke_test.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    combined_output = result.stdout + result.stderr
    if secret_probe in combined_output:
        print("[FAIL] CRITICAL: API key was exposed in smoke test output!")
        return False

    print("[PASS] Secret token is strictly redacted and never printed or logged.")
    return True


def check_adr_contents() -> bool:
    adr_path = PROJECT_ROOT / "docs" / "decisions" / "0004-gemini-config.md"
    if not adr_path.exists():
        print(f"[FAIL] Missing {adr_path}")
        return False

    content = adr_path.read_text(encoding="utf-8")
    errors = []

    if "Status: Accepted" not in content and "**Status:** Accepted" not in content:
        errors.append("ADR 0004 status is not 'Accepted'")

    if "gemini-1.5-flash" not in content:
        errors.append("ADR 0004 does not mention default model 'gemini-1.5-flash'")

    if "google-genai" not in content:
        errors.append("ADR 0004 does not mention 'google-genai' SDK")

    if "could not be verified" not in content and "verified" not in content:
        errors.append("ADR 0004 does not record verification status")

    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    print("[PASS] ADR 0004 properly documents model name, SDK, and verification status.")
    return True


def main() -> None:
    print("=== Running T1-008 Verification ===")
    ok = True
    ok = check_script_exists() and ok
    ok = check_missing_key_behavior() and ok
    ok = check_secret_redaction() and ok
    ok = check_adr_contents() and ok

    if ok:
        print("\n[ALL CHECKS PASSED] T1-008 acceptance criteria successfully verified.")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILED] One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
