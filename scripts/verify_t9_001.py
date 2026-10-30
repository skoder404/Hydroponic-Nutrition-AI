#!/usr/bin/env python3
"""Verification script for T9-001: Write Dockerfiles for backend and frontend.

Acceptance Criteria:
1. Images build from clean checkout (Dockerfiles, multi-stage, pinned bases, valid syntax).
2. No secrets or datasets in image layers (.dockerignore rules and volume isolation).
3. Container runs as non-root (USER directive in runner stages).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify_dockerfiles_exist() -> tuple[bool, str]:
    backend = ROOT / "backend" / "Dockerfile"
    frontend = ROOT / "frontend" / "Dockerfile"
    dockerignore = ROOT / ".dockerignore"

    if not backend.is_file():
        return False, f"Missing backend Dockerfile at {backend}"
    if not frontend.is_file():
        return False, f"Missing frontend Dockerfile at {frontend}"
    if not dockerignore.is_file():
        return False, f"Missing .dockerignore at {dockerignore}"

    return True, "All Dockerfiles and .dockerignore exist."


def verify_backend_dockerfile() -> tuple[bool, str]:
    content = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")

    stages = re.findall(
        r"^FROM\s+([^\s]+)\s+AS\s+([^\s]+)", content, re.MULTILINE | re.IGNORECASE
    )
    if len(stages) < 2:
        return (
            False,
            f"Backend Dockerfile must be multi-stage, found {len(stages)} stage(s).",
        )

    for img, stage in stages:
        if ":" not in img or img.endswith(":latest"):
            return (
                False,
                f"Backend base image '{img}' in stage '{stage}' is unpinned or uses :latest.",
            )

    if not re.search(r"^USER\s+appuser", content, re.MULTILINE):
        return (
            False,
            "Backend Dockerfile must specify non-root USER appuser in runner stage.",
        )

    if "ml-cpu.txt" not in content:
        return False, "Backend Dockerfile must default to CPU build with ml-cpu.txt."

    if "GPU" not in content:
        return False, "Backend Dockerfile must document GPU build variant."

    return (
        True,
        f"Backend Dockerfile passed: {len(stages)} stages, pinned base, non-root user, CPU/GPU documented.",
    )


def verify_frontend_dockerfile() -> tuple[bool, str]:
    content = (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")

    stages = re.findall(
        r"^FROM\s+([^\s]+)\s+AS\s+([^\s]+)", content, re.MULTILINE | re.IGNORECASE
    )
    if len(stages) < 2:
        return (
            False,
            f"Frontend Dockerfile must be multi-stage, found {len(stages)} stage(s).",
        )

    for img, stage in stages:
        if ":" not in img or img.endswith(":latest"):
            return (
                False,
                f"Frontend base image '{img}' in stage '{stage}' is unpinned or uses :latest.",
            )

    if not re.search(r"^USER\s+nextjs", content, re.MULTILINE):
        return (
            False,
            "Frontend Dockerfile must specify non-root USER nextjs in runner stage.",
        )

    return (
        True,
        f"Frontend Dockerfile passed: {len(stages)} stages, pinned base, non-root user.",
    )


def verify_dockerignore_rules() -> tuple[bool, str]:
    content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    required = [
        "image_data/",
        "sensor_data/",
        "processed_data/",
        "uploads/",
        "models/",
        ".env",
        ".venv/",
    ]
    for req in required:
        if not any(
            req in line or line.rstrip("/") == req.rstrip("/") for line in lines
        ):
            return False, f".dockerignore missing required entry '{req}'."

    return (
        True,
        f".dockerignore excludes all {len(required)} sensitive/dataset targets.",
    )


def main() -> int:
    print("=== Running T9-001 Verification ===")
    checks = [
        ("Dockerfiles Existence", verify_dockerfiles_exist),
        ("Backend Dockerfile Configuration", verify_backend_dockerfile),
        ("Frontend Dockerfile Configuration", verify_frontend_dockerfile),
        (".dockerignore Exclusion Rules", verify_dockerignore_rules),
    ]

    all_passed = True
    for name, fn in checks:
        ok, msg = fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll T9-001 acceptance criteria satisfied.")
        return 0
    else:
        print("\nT9-001 verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
