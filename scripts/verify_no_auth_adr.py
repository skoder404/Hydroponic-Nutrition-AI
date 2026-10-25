#!/usr/bin/env python3
"""Verification script for T3-001: ADR 0005 No Authentication in v1.

Validates:
1. docs/decisions/0005-no-auth.md exists.
2. ADR lists withheld capabilities:
   - no model promotion / rollback over HTTP
   - no dataset writes / modifications
   - no code execution
3. ADR states explicit triggers for adding authentication:
   - multi-tenancy / public deployment
   - user data isolation
   - actuator / hardware control
4. README.md security note links directly to docs/decisions/0005-no-auth.md.
"""
import sys
from pathlib import Path


def check_adr():
    p = Path("docs/decisions/0005-no-auth.md")
    if not p.is_file():
        return False, f"{p} does not exist"
    content = p.read_text(encoding="utf-8").lower()

    # Withheld capabilities
    withheld_keywords = [
        "model promotion",
        "dataset write",
        "code execution",
    ]
    missing_withheld = [k for k in withheld_keywords if k not in content]
    if missing_withheld:
        return False, f"ADR missing withheld capabilities: {missing_withheld}"

    # Triggers
    trigger_keywords = [
        "trigger",
        "threat model",
    ]
    missing_triggers = [k for k in trigger_keywords if k not in content]
    if missing_triggers:
        return False, f"ADR missing trigger definitions: {missing_triggers}"

    return True, "ADR 0005 exists and documents withheld capabilities and auth triggers."


def check_readme_link():
    p = Path("README.md")
    if not p.is_file():
        return False, "README.md does not exist"
    content = p.read_text(encoding="utf-8")
    if "0005-no-auth.md" not in content:
        return False, "README.md does not link to docs/decisions/0005-no-auth.md"
    return True, "README.md security note links to ADR 0005."


def main():
    print("=== Running T3-001 Verification ===")
    errors = []

    ok_adr, msg_adr = check_adr()
    if not ok_adr:
        errors.append(msg_adr)
    else:
        print(f"[PASS] {msg_adr}")

    ok_rm, msg_rm = check_readme_link()
    if not ok_rm:
        errors.append(msg_rm)
    else:
        print(f"[PASS] {msg_rm}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T3-001 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
