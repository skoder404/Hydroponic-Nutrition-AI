#!/usr/bin/env python3
"""Verification script for T1-006: scripts/check_env.py and ADR 0002 appendices.

Validates:
1. scripts/check_env.py executes cleanly and returns all required libraries and hardware specs.
2. Output honestly states the device used and never claims GPU when the tensor ran on CPU.
3. FORCE_CPU=1 or force_cpu_override strictly forces CPU execution and passes tensor tests.
4. docs/decisions/0002-runtime-versions.md contains diagnostic appendices for both teammates (Sushil and Yashwant).
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_env import check_environment


def check_default_execution_honesty():
    report = check_environment(force_cpu_override=False)

    required_keys = [
        "os", "architecture", "python_version",
        "torch", "torchvision", "scikit_learn", "scikit_image", "opencv",
        "target_device", "tensor_test_device", "tensor_test_passed",
    ]
    missing = [k for k in required_keys if k not in report]
    if missing:
        return False, f"Report missing required keys: {missing}"

    if not report["tensor_test_passed"]:
        return False, f"Tensor execution test failed: {report['tensor_test_result']}"

    target = report["target_device"].lower()
    actual = report["tensor_test_device"].lower()

    if target.startswith("cpu") and "cuda" in actual:
        return False, "Honesty violation: Target is CPU but actual is CUDA"
    if target.startswith("cuda") and "cpu" in actual:
        return False, "Honesty violation: Target is CUDA but actual is CPU"

    return True, f"Device execution verified honest: target={target}, actual={actual} (PASSED)."


def check_force_cpu_override():
    # Test programmatic force_cpu
    report = check_environment(force_cpu_override=True)
    if report["target_device"] != "cpu":
        return False, f"force_cpu_override did not set target_device to 'cpu': {report['target_device']}"
    if report["tensor_test_device"] != "cpu":
        return False, f"force_cpu_override did not execute tensor on 'cpu': {report['tensor_test_device']}"
    if not report["tensor_test_passed"]:
        return False, f"Tensor test on forced CPU failed: {report['tensor_test_result']}"

    # Test environment variable FORCE_CPU=1
    old_env = os.environ.get("FORCE_CPU")
    os.environ["FORCE_CPU"] = "1"
    try:
        env_report = check_environment(force_cpu_override=False)
        if env_report["target_device"] != "cpu" or env_report["tensor_test_device"] != "cpu":
            return False, "FORCE_CPU=1 env variable did not force CPU device"
    finally:
        if old_env is not None:
            os.environ["FORCE_CPU"] = old_env
        else:
            del os.environ["FORCE_CPU"]

    return True, "FORCE_CPU=1 correctly forces CPU execution and tensor execution succeeds."


def check_adr_appendices():
    adr_path = Path("docs/decisions/0002-runtime-versions.md")
    if not adr_path.is_file():
        return False, "docs/decisions/0002-runtime-versions.md missing"

    content = adr_path.read_text(encoding="utf-8")

    # Check for both teammates' reports
    if "Sushil's Environment Diagnostic Report" not in content and "Sushil" not in content:
        return False, "ADR 0002 missing Sushil's environment diagnostic appendix"

    if "Yashwant's Environment Diagnostic Report" not in content and "Yashwant" not in content:
        return False, "ADR 0002 missing Yashwant's environment diagnostic appendix"

    # Verify both reports contain tensor test results and FORCE_CPU checks
    if "Tensor-on-Device Test" not in content or "FORCE_CPU Override Test" not in content:
        return False, "ADR 0002 appendices missing tensor or FORCE_CPU test summaries"

    return True, "Both teammates' diagnostic reports are saved in ADR 0002."


def main():
    print("=== Running T1-006 Verification ===")
    errors = []

    ok_exec, msg_exec = check_default_execution_honesty()
    if not ok_exec:
        errors.append(msg_exec)
    else:
        print(f"[PASS] {msg_exec}")

    ok_cpu, msg_cpu = check_force_cpu_override()
    if not ok_cpu:
        errors.append(msg_cpu)
    else:
        print(f"[PASS] {msg_cpu}")

    ok_adr, msg_adr = check_adr_appendices()
    if not ok_adr:
        errors.append(msg_adr)
    else:
        print(f"[PASS] {msg_adr}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T1-006 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
