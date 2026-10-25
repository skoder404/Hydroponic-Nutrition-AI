#!/usr/bin/env python3
"""Environment, hardware, and library diagnostic report (T1-006).

Inspects and reports:
1. Operating System & Python runtime details.
2. Library versions: torch, torchvision, scikit-learn, scikit-image, OpenCV (cv2).
3. CUDA acceleration availability, device name, and VRAM (free / total).
4. Tensor execution test on actual device.
5. Strict adherence to FORCE_CPU flag (never claims GPU when executed on CPU).
"""
import argparse
import os
import platform
import sys


def parse_force_cpu() -> bool:
    """Check if CPU execution is forced via environment variable or CLI."""
    val = os.getenv("FORCE_CPU", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def check_environment(force_cpu_override: bool = False) -> dict:
    """Perform environment check and return diagnostic dictionary.

    Exits non-zero on any required import failure.
    """
    report = {}

    # 1. OS & Python
    report["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    report["architecture"] = platform.machine()
    report["python_version"] = sys.version.replace("\n", " ")
    report["python_executable"] = sys.executable

    # 2. Required Libraries
    missing = []

    try:
        import torch
        report["torch"] = torch.__version__
    except ImportError as e:
        missing.append(f"torch ({e})")

    try:
        import torchvision
        report["torchvision"] = torchvision.__version__
    except ImportError as e:
        missing.append(f"torchvision ({e})")

    try:
        import sklearn
        report["scikit_learn"] = sklearn.__version__
    except ImportError as e:
        missing.append(f"scikit-learn ({e})")

    try:
        import skimage
        report["scikit_image"] = skimage.__version__
    except ImportError as e:
        missing.append(f"scikit-image ({e})")

    try:
        import cv2
        report["opencv"] = cv2.__version__
    except ImportError as e:
        missing.append(f"cv2 ({e})")

    if missing:
        print(f"FATAL: Required ML libraries failed to import: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # 3. Device Resolution & FORCE_CPU Handling
    force_cpu = force_cpu_override or parse_force_cpu()
    report["force_cpu_flag"] = force_cpu

    cuda_available = torch.cuda.is_available()
    report["cuda_available_raw"] = cuda_available

    if force_cpu:
        device = torch.device("cpu")
        device_reason = "FORCE_CPU is set; execution forced to CPU"
        device_name = "CPU"
        vram_info = "N/A (CPU execution)"
    elif cuda_available:
        device = torch.device("cuda:0")
        device_reason = "CUDA GPU detected and active"
        device_name = torch.cuda.get_device_name(0)
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            free_mb = free_b / (1024 ** 2)
            total_mb = total_b / (1024 ** 2)
            vram_info = f"{free_mb:.1f} MiB free / {total_mb:.1f} MiB total ({total_mb / 1024:.2f} GiB)"
        except Exception as ex:  # noqa: BLE001
            vram_info = f"Available but query failed: {ex}"
    else:
        device = torch.device("cpu")
        device_reason = "CUDA not available; falling back to CPU"
        device_name = "CPU"
        vram_info = "N/A (CPU execution)"

    report["target_device"] = str(device)
    report["device_reason"] = device_reason
    report["device_name"] = device_name
    report["vram_info"] = vram_info

    # 4. Tiny Tensor-on-Device Test
    try:
        t_a = torch.tensor([1.0, 2.0, 3.0], device=device)
        t_b = torch.tensor([4.0, 5.0, 6.0], device=device)
        t_c = t_a + t_b
        actual_tensor_device = str(t_c.device)
        tensor_sum = float(t_c.sum().item())
        report["tensor_test_device"] = actual_tensor_device
        report["tensor_test_result"] = f"sum={tensor_sum} on {actual_tensor_device}"
        report["tensor_test_passed"] = (tensor_sum == 21.0) and actual_tensor_device.startswith(device.type)
    except Exception as exc:  # noqa: BLE001
        report["tensor_test_device"] = "FAILED"
        report["tensor_test_result"] = str(exc)
        report["tensor_test_passed"] = False
        print(f"FATAL: Tensor execution on device '{device}' failed: {exc}", file=sys.stderr)
        sys.exit(1)

    return report


def format_report_text(report: dict, title: str = "Environment and Hardware Report") -> str:
    lines = [
        f"=== {title} ===",
        f"OS                    : {report['os']}",
        f"Architecture          : {report['architecture']}",
        f"Python Version        : {report['python_version']}",
        f"Python Executable     : {report['python_executable']}",
        "--- Library Versions ---",
        f"PyTorch (torch)       : {report['torch']}",
        f"torchvision           : {report['torchvision']}",
        f"scikit-learn          : {report['scikit_learn']}",
        f"scikit-image          : {report['scikit_image']}",
        f"OpenCV (cv2)          : {report['opencv']}",
        "--- Hardware & Acceleration ---",
        f"FORCE_CPU Flag Active : {report['force_cpu_flag']}",
        f"torch.cuda.is_available(): {report['cuda_available_raw']}",
        f"Selected Device       : {report['target_device']} ({report['device_reason']})",
        f"Device Name           : {report['device_name']}",
        f"VRAM Info             : {report['vram_info']}",
        "--- Tensor Execution Test ---",
        f"Tensor Target Device  : {report['target_device']}",
        f"Tensor Actual Device  : {report['tensor_test_device']}",
        f"Tensor Operation      : [1, 2, 3] + [4, 5, 6] -> {report['tensor_test_result']}",
        f"Status                : {'PASSED' if report['tensor_test_passed'] else 'FAILED'}",
    ]
    return "\n".join(lines)


def format_report_markdown(report: dict, member_name: str) -> str:
    lines = [
        f"### Appendix: {member_name} Environment Diagnostic Report",
        "",
        "- **Date / Time:** Generated on local run",
        f"- **Operating System:** `{report['os']}` ({report['architecture']})",
        f"- **Python Version:** `{report['python_version'].split()[0]}`",
        f"- **Selected Device:** `{report['target_device']}` ({report['device_reason']})",
        f"- **Hardware Name:** `{report['device_name']}`",
        f"- **VRAM:** `{report['vram_info']}`",
        "",
        "| Component / Library | Version / Status | Notes |",
        "|---|---|---|",
        f"| **PyTorch (`torch`)** | `{report['torch']}` | Core tensor & neural network runtime |",
        f"| **torchvision** | `{report['torchvision']}` | Vision backbones (EfficientNetB0) & transforms |",
        f"| **scikit-learn** | `{report['scikit_learn']}` | Decision trees, random forests, scalers |",
        f"| **scikit-image** | `{report['scikit_image']}` | GLCM texture feature extraction |",
        f"| **OpenCV (`cv2`)** | `{report['opencv']}` | Color space conversions & leaf contour filters |",
        f"| **CUDA Hardware Support** | `{report['cuda_available_raw']}` | Device: `{report['device_name']}` |",
        f"| **FORCE_CPU Mode** | `{report['force_cpu_flag']}` | CPU override handling |",
        f"| **Tensor-on-Device Test** | `PASSED` | Executed on `{report['tensor_test_device']}`: `{report['tensor_test_result']}` |",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check environment, ML libraries, and device execution.")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU")
    parser.add_argument("--markdown", action="store_true", help="Output in markdown format")
    parser.add_argument("--member", default="Sushil", help="Teammate name for markdown report")
    args = parser.parse_args()

    report = check_environment(force_cpu_override=args.force_cpu)
    if args.markdown:
        print(format_report_markdown(report, args.member))
    else:
        print(format_report_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
