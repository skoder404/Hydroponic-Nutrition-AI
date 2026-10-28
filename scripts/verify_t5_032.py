#!/usr/bin/env python3
"""Verification script for T5-032: Run-once final test evaluation for image models.

Validates:
1. Log entry is written for each evaluated model in reports/final_eval_log.jsonl.
2. Re-run without --force refuses.
3. Reports label metrics as test results with split named.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
LOG_FILE = REPORTS_DIR / "final_eval_log.jsonl"
sys.path.insert(0, str(REPO_ROOT))


def run_verification() -> bool:
    print("[1/4] Running unit tests in tests/unit/test_final_eval_image.py...")
    exit_code = pytest.main(
        [
            "-v",
            str(REPO_ROOT / "tests/unit/test_final_eval_image.py"),
        ]
    )
    if exit_code != 0:
        print(f"FAILED: pytest exited with code {exit_code}")
        return False

    print("\n[2/4] Verifying reports/final_eval_log.jsonl entries...")
    if not LOG_FILE.exists():
        print(f"FAILED: {LOG_FILE} does not exist.")
        return False

    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    image_entries = [
        e
        for e in entries
        if e.get("modality") == "image"
        or "efficientnet" in str(e.get("model_name", "")).lower()
    ]
    crops_logged = {e["crop"] for e in image_entries}
    required_crops = {"lettuce", "spinach", "tomato"}
    if not required_crops.issubset(crops_logged):
        print(f"FAILED: Expected crops {required_crops} logged, found {crops_logged}")
        return False

    for e in image_entries:
        if e.get("split_evaluated") != "test":
            print(
                f"FAILED: Entry {e.get('model_name')} split_evaluated is not 'test': {e.get('split_evaluated')}"
            )
            return False

    print("\n[3/4] Verifying generated test reports exist and name test split...")
    for crop in required_crops:
        json_path = REPORTS_DIR / crop / f"{crop}_efficientnet_b0_v1.0.0_test.json"
        md_path = REPORTS_DIR / crop / f"{crop}_efficientnet_b0_v1.0.0_test.md"
        if not json_path.exists():
            print(f"FAILED: Missing report {json_path}")
            return False
        if not md_path.exists():
            print(f"FAILED: Missing report {md_path}")
            return False

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("split_evaluated") != "test":
            print(f"FAILED: JSON report {json_path} split is not 'test'")
            return False

    print("\n[4/4] Verifying CLI refuses re-run without --force...")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/final_eval_image.py"), "lettuce"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        print(
            "FAILED: final_eval_image.py should have returned non-zero when re-run without --force"
        )
        return False
    if (
        "Re-running requires --force" not in proc.stderr
        and "Re-running requires --force" not in proc.stdout
    ):
        print(
            f"FAILED: Expected refusal message in output, got: {proc.stderr or proc.stdout}"
        )
        return False

    print("\nAll T5-032 acceptance criteria verified successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
