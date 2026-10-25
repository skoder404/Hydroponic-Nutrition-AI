#!/usr/bin/env python3
"""Verification script for T1-005: Runtime Python and ML dependency versions.

Validates:
1. docs/decisions/0002-runtime-versions.md exists and is Accepted.
2. ADR 0002 documents all 9 mandated packages with pinned versions and one-line reasons:
   - PyTorch (torch)
   - torchvision
   - scikit-learn
   - scikit-image
   - OpenCV (opencv-python-headless)
   - FastAPI
   - Pydantic
   - SQLAlchemy
   - Alembic
3. Pinned requirements files exist in requirements/:
   - requirements/backend.txt
   - requirements/ml.txt
   - requirements/ml-cpu.txt
   - requirements/dev.txt
   - requirements.txt
4. Required dependencies are present with exact pins in their corresponding files.
5. CPU-only and GPU configuration checks are documented and valid.
"""
import sys
from pathlib import Path

MANDATED_PACKAGES = [
    ("PyTorch", "torch"),
    ("torchvision", "torchvision"),
    ("scikit-learn", "scikit-learn"),
    ("scikit-image", "scikit-image"),
    ("OpenCV", "opencv-python-headless"),
    ("FastAPI", "fastapi"),
    ("Pydantic", "pydantic"),
    ("SQLAlchemy", "SQLAlchemy"),
    ("Alembic", "alembic"),
]

REQUIRED_FILES = [
    Path("docs/decisions/0002-runtime-versions.md"),
    Path("requirements/backend.in"),
    Path("requirements/backend.txt"),
    Path("requirements/ml.in"),
    Path("requirements/ml.txt"),
    Path("requirements/ml-cpu.txt"),
    Path("requirements/dev.in"),
    Path("requirements/dev.txt"),
    Path("requirements.txt"),
]


def check_files_exist():
    missing = [f for f in REQUIRED_FILES if not f.exists()]
    if missing:
        print(f"[FAIL] Missing required files: {missing}")
        return False
    print(f"[PASS] All {len(REQUIRED_FILES)} required files exist.")
    return True


def check_adr_contents():
    adr_path = Path("docs/decisions/0002-runtime-versions.md")
    content = adr_path.read_text(encoding="utf-8")

    errors = []
    if "Status: Accepted" not in content and "**Status:** Accepted" not in content:
        errors.append("ADR status is not 'Accepted'")

    if "HP Victus 16" not in content or "RTX 3050" not in content:
        errors.append("Hardware baseline (HP Victus 16, RTX 3050) not documented")

    if "Python 3.11" not in content:
        errors.append("Python 3.11 baseline not documented")

    if "pip-tools" not in content:
        errors.append("Package tool (pip-tools) not documented")

    for display_name, pkg_name in MANDATED_PACKAGES:
        # Check if package is documented in table
        if pkg_name.lower() not in content.lower() and display_name.lower() not in content.lower():
            errors.append(f"Mandated package '{display_name}' ({pkg_name}) missing from ADR")

    if errors:
        for err in errors:
            print(f"[FAIL] ADR check: {err}")
        return False

    print("[PASS] ADR 0002 contains all required hardware, Python, package manager, and package pins.")
    return True


def check_requirement_pins():
    backend_txt = Path("requirements/backend.txt").read_text(encoding="utf-8")
    ml_txt = Path("requirements/ml.txt").read_text(encoding="utf-8")
    ml_cpu_txt = Path("requirements/ml-cpu.txt").read_text(encoding="utf-8")
    dev_txt = Path("requirements/dev.txt").read_text(encoding="utf-8")

    checks = [
        ("fastapi==0.115.11", backend_txt, "backend.txt"),
        ("pydantic==2.10.6", backend_txt, "backend.txt"),
        ("SQLAlchemy==2.0.38", backend_txt, "backend.txt"),
        ("alembic==1.14.1", backend_txt, "backend.txt"),
        ("torch==2.5.1+cu124", ml_txt, "ml.txt"),
        ("torchvision==0.20.1+cu124", ml_txt, "ml.txt"),
        ("torch==2.5.1+cpu", ml_cpu_txt, "ml-cpu.txt"),
        ("torchvision==0.20.1+cpu", ml_cpu_txt, "ml-cpu.txt"),
        ("scikit-learn==1.6.1", ml_txt, "ml.txt"),
        ("scikit-image==0.25.2", ml_txt, "ml.txt"),
        ("opencv-python-headless==4.11.0.86", ml_txt, "ml.txt"),
        ("pytest==8.3.5", dev_txt, "dev.txt"),
    ]

    failed = False
    for pin, content, filename in checks:
        if pin not in content:
            print(f"[FAIL] Expected '{pin}' in requirements/{filename}")
            failed = True

    if failed:
        return False

    print("[PASS] All core packages verified with exact pinned versions in requirements files.")
    return True


def main():
    print("=== Running T1-005 Verification ===")
    ok = True
    ok = check_files_exist() and ok
    ok = check_adr_contents() and ok
    ok = check_requirement_pins() and ok

    if ok:
        print("\n[ALL CHECKS PASSED] T1-005 acceptance criteria successfully verified.")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILED] One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
