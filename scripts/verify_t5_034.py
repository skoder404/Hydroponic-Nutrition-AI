#!/usr/bin/env python3
"""Verification script for T5-034: Phase 5 teach-back and concept check.

Acceptance Criteria:
1. Every Phase 5 concept has a note and a teach-back for both members.
2. Gaps become tracked follow-ups.
3. Both members tick the phase gate.
"""

import re
import sys
from pathlib import Path

PHASE5_CONCEPT_FILES = [
    "colour-spaces-bgrrgbhsv-and-vegetation-indices.md",
    "texture-features-glcm-contrast-and-entropy.md",
    "shape-descriptors-compactness-solidity-aspect-ratio.md",
    "instance-segmentation-for-perleaf-prediction-sam.md",
    "weak-labels-and-instancelevel-perleaf-prediction.md",
    "convolution-pooling-and-backpropagation.md",
    "data-augmentation-training-only-and-input.md",
    "optimisers-learning-rate-and-early-stopping.md",
    "transfer-learning-and-finetuning-efficientnetb0.md",
    "softmax-confidence-calibration-and-uncertainty-thresholds.md",
]


def verify_all_concept_notes():
    for cf in PHASE5_CONCEPT_FILES:
        path = Path("docs/learning") / cf
        if not path.is_file():
            return False, f"Missing concept note: docs/learning/{cf}"
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 500:
            return False, f"Concept note too short: docs/learning/{cf}"
    return (
        True,
        f"All {len(PHASE5_CONCEPT_FILES)} Phase 5 concept notes exist and are populated.",
    )


def verify_teachback_log():
    log_path = Path("docs/learning/phase5_teachback_log.md")
    if not log_path.is_file():
        return False, "Missing docs/learning/phase5_teachback_log.md"
    content = log_path.read_text(encoding="utf-8")

    if "Yashwant" not in content or "Sushil" not in content:
        return False, "phase5_teachback_log.md does not record both Yashwant and Sushil"

    if "Identified Gap / Follow-up:" not in content:
        return (
            False,
            "phase5_teachback_log.md does not record identified gaps and follow-ups",
        )

    if "PASS" not in content:
        return False, "phase5_teachback_log.md does not contain PASS verdict"

    return (
        True,
        "Teach-back session audit log contains both members, tracked gaps, and PASS status.",
    )


def verify_progress_tracker():
    progress_path = Path("docs/learning/PROGRESS.md")
    if not progress_path.is_file():
        return False, "Missing docs/learning/PROGRESS.md"
    content = progress_path.read_text(encoding="utf-8")

    for cf in PHASE5_CONCEPT_FILES:
        pattern = rf"`docs/learning/{re.escape(cf)}`\s*\|\s*\[x\]\s*\|\s*\[x\]\s*\|\s*\[x\].*\|\s*\[x\]\s*\|\s*\[x\]\s*\|\s*\[x\]"
        if not re.search(pattern, content):
            return (
                False,
                f"Concept docs/learning/{cf} not fully ticked for both members in PROGRESS.md",
            )

    if "Phase 5 Gate Sign-Off (T5-034)" not in content:
        return False, "PROGRESS.md missing Phase 5 Gate Sign-Off section"

    if "[x] Yashwant" not in content or "[x] Sushil" not in content:
        return (
            False,
            "PROGRESS.md Phase 5 Gate Sign-Off missing signed checklist from both members",
        )

    return (
        True,
        "PROGRESS.md has all Phase 5 concepts ticked for both members and Phase 5 gate signed.",
    )


def main():
    print("=== Verifying T5-034: Phase 5 Teach-Back and Concept Check ===")
    checks = [
        ("Concept Notes Check", verify_all_concept_notes),
        ("Teach-Back Audit Log Check", verify_teachback_log),
        ("Progress Tracker & Phase Gate Check", verify_progress_tracker),
    ]

    all_passed = True
    for name, func in checks:
        passed, msg = func()
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}: {msg}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll T5-034 acceptance criteria satisfied.")
        sys.exit(0)
    else:
        print("\nT5-034 verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
