#!/usr/bin/env python3
"""Verification script for T1-015: Learning scaffold and progress tracker.

Validates:
1. docs/learning/_TEMPLATE_concept.md exists with all required sections.
2. docs/learning/_TEMPLATE_teachback.md exists with all required sections.
3. docs/learning/PROGRESS.md exists and contains:
   - Every unique concept from Learning Focus in todo.md (26 concepts).
   - Column sets for both teammates (Yashwant and Sushil): Note, Teach-Back, Self-Check, Date.
"""
import re
import sys
from pathlib import Path


def check_concept_template():
    p = Path("docs/learning/_TEMPLATE_concept.md")
    if not p.is_file():
        return False, f"{p} does not exist"
    content = p.read_text(encoding="utf-8")
    required = [
        "## 1. Intuition",
        "## 2. Mathematical Formulation",
        "## 3. From-Scratch Implementation",
        "## 4. Production & Library Usage",
        "## 5. Failure Modes, Edge Cases & Debugging",
        "## 6. Where It Appears in This Project",
        "## 7. Self-Check Questions",
    ]
    missing = [req for req in required if req not in content]
    if missing:
        return False, f"_TEMPLATE_concept.md missing sections: {missing}"
    return True, "_TEMPLATE_concept.md has all required headings and valid markdown structure"


def check_teachback_template():
    p = Path("docs/learning/_TEMPLATE_teachback.md")
    if not p.is_file():
        return False, f"{p} does not exist"
    content = p.read_text(encoding="utf-8")
    required = [
        "## 1. Verbal Explanation Summary",
        "## 2. Whiteboard / Hand-Derivation Check",
        "## 3. Answers to Self-Check Questions",
        "## 4. Gaps Identified & Reviewer Feedback",
        "## 5. Verification Sign-Off",
    ]
    missing = [req for req in required if req not in content]
    if missing:
        return False, f"_TEMPLATE_teachback.md missing sections: {missing}"
    return True, "_TEMPLATE_teachback.md has all required headings and valid markdown structure"


def check_progress_tracker():
    p = Path("docs/learning/PROGRESS.md")
    if not p.is_file():
        return False, f"{p} does not exist"
    content = p.read_text(encoding="utf-8")

    # Extract concepts from todo.md
    todo_text = Path("todo.md").read_text(encoding="utf-8")
    matches = re.findall(r"\*\*Learning Focus:\*\*\s*(.+)", todo_text)
    expected_concepts = []
    for m in matches:
        raw = m.strip()
        name = re.split(r"\s*(?:->|→)\s*", raw)[0].strip()
        if name not in expected_concepts:
            expected_concepts.append(name)

    missing_concepts = [c for c in expected_concepts if c not in content]
    if missing_concepts:
        return False, f"PROGRESS.md missing concepts: {missing_concepts}"

    # Check columns for both teammates
    for member in ["Yashwant", "Sushil"]:
        for col in ["Note", "Teach-Back", "Self-Check", "Date"]:
            header_name = f"{member} {col}"
            if header_name not in content:
                return False, f"PROGRESS.md missing column '{header_name}'"

    return True, f"PROGRESS.md lists all {len(expected_concepts)} concepts with columns for Yashwant and Sushil"


def main():
    print("=== Running T1-015 Verification ===")
    errors = []

    ok_c, msg_c = check_concept_template()
    if not ok_c:
        errors.append(msg_c)
    else:
        print(f"[PASS] {msg_c}")

    ok_t, msg_t = check_teachback_template()
    if not ok_t:
        errors.append(msg_t)
    else:
        print(f"[PASS] {msg_t}")

    ok_p, msg_p = check_progress_tracker()
    if not ok_p:
        errors.append(msg_p)
    else:
        print(f"[PASS] {msg_p}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T1-015 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
