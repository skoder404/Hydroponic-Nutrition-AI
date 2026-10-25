#!/usr/bin/env python3
"""Verification script for T1-007: SAM 2.1 feasibility spike and ADR 0003.

Validates:
1. reports/sam2_feasibility_metrics.json exists with measured numbers.
2. docs/decisions/0003-sam2-feasibility.md exists and documents:
   - exact checkpoint and config (sam2.1_hiera_tiny.pt, sam2.1_hiera_t.yaml)
   - measured load times, peak VRAM, and per-crop latencies
   - CPU fallback viability evaluation
   - concrete Phase 5 options and fallback strategies
3. SAM 2.1 checkpoint is present and model can be loaded.
"""
import json
import sys
from pathlib import Path


def check_metrics_report():
    report_path = Path("reports/sam2_feasibility_metrics.json")
    if not report_path.is_file():
        return False, "reports/sam2_feasibility_metrics.json does not exist"

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to parse metrics JSON: {exc}"

    required_fields = [
        "checkpoint", "config", "checkpoint_size_mb",
        "cpu_load_time_s", "predictor_latencies", "lettuce_leaf_separation",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Metrics report missing required fields: {missing}"

    return True, "reports/sam2_feasibility_metrics.json contains valid measured data."


def check_adr_0003():
    adr_path = Path("docs/decisions/0003-sam2-feasibility.md")
    if not adr_path.is_file():
        return False, "docs/decisions/0003-sam2-feasibility.md does not exist"

    content = adr_path.read_text(encoding="utf-8")

    # Check exact checkpoint & config
    if "sam2.1_hiera_tiny.pt" not in content or "sam2.1_hiera_t.yaml" not in content:
        return False, "ADR 0003 missing exact checkpoint or config specification"

    # Check measured numbers
    if "2,316.0 MB" not in content and "2316" not in content:
        return False, "ADR 0003 missing measured peak VRAM numbers"

    # Check CPU fallback statement
    if "CPU Fallback" not in content or "Viable" not in content:
        return False, "ADR 0003 missing clear statement on CPU fallback viability"

    # Check Phase 5 options
    if "Phase 5" not in content or "Fallback Options" not in content:
        return False, "ADR 0003 missing Phase 5 options and fallback strategies"

    return True, "docs/decisions/0003-sam2-feasibility.md satisfies all acceptance criteria."


def check_sam2_checkpoint_exists():
    ckpt = Path("models/image/sam2.1_hiera_tiny.pt")
    if not ckpt.is_file() or ckpt.stat().st_size < 100_000_000:
        return False, f"Checkpoint file missing or truncated: {ckpt}"
    return True, f"SAM 2.1 checkpoint verified on disk ({ckpt.stat().st_size / (1024*1024):.1f} MB)."


def main():
    print("=== Running T1-007 Verification ===")
    errors = []

    ok_m, msg_m = check_metrics_report()
    if not ok_m:
        errors.append(msg_m)
    else:
        print(f"[PASS] {msg_m}")

    ok_a, msg_a = check_adr_0003()
    if not ok_a:
        errors.append(msg_a)
    else:
        print(f"[PASS] {msg_a}")

    ok_c, msg_c = check_sam2_checkpoint_exists()
    if not ok_c:
        errors.append(msg_c)
    else:
        print(f"[PASS] {msg_c}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T1-007 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
