"""Verification script for .gitignore and .gitattributes rules (T1-003).

Validates:
1. .env is ignored while .env.example is tracked.
2. image_data/ and sensor_data/ remain tracked in the repository.
3. models/ and processed_data/ directories remain trackable.
4. .gitattributes exists and configures LF line-ending normalization and notebook output filtering.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_git_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repo root directory."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def is_ignored(path_str: str) -> bool:
    """Check if git ignores the specified path."""
    res = run_git_command(["check-ignore", path_str])
    return res.returncode == 0


def verify_env_rules() -> tuple[bool, str]:
    """Criterion 1: .env is ignored while .env.example is tracked."""
    if not is_ignored(".env"):
        return False, ".env is not ignored by .gitignore."
    if not is_ignored(".env.local"):
        return False, ".env.local is not ignored by .gitignore."
    if is_ignored(".env.example"):
        return False, ".env.example is ignored (should be trackable)."
    example_path = REPO_ROOT / ".env.example"
    if not example_path.exists():
        return False, ".env.example file does not exist."
    return True, ".env is ignored while .env.example is trackable and exists."


def verify_datasets_tracked() -> tuple[bool, str]:
    """Criterion 2: image_data/ and sensor_data/ remain tracked in the repository."""
    dataset_paths = [
        "image_data/Lettuce_Images/class_counts.csv",
        "image_data/Lettuce_Images/Healthier",
        "sensor_data/Lettuce_Sensor-Readings/hydroponic_lettuce_full.csv",
        "sensor_data/Spinach_Sensor-Readings/spinach_7class_full.csv",
        "sensor_data/Tomato_Sensor-Readings/tomato_hydroponic_full.csv",
        "sensor_data/Lettuce_Sensor-Readings/build_lettuce_dataset",
        "sensor_data/Spinach_Sensor-Readings/generate_7class_dataset.py",
        "sensor_data/Tomato_Sensor-Readings/build_tomato_dataset.py",
    ]
    for p in dataset_paths:
        if is_ignored(p):
            return False, f"Dataset path was ignored (must be tracked in repo): {p}"
    return True, "image_data/ and sensor_data/ datasets are confirmed tracked in the repository."


def verify_models_and_processed_data_trackable() -> tuple[bool, str]:
    """Criterion 3: models/ and processed_data/ directories remain trackable."""
    paths_to_test = [
        "models/image/model.pt",
        "models/sensor/model.pkl",
        "models/manifests/manifest.json",
        "processed_data/features/features.parquet",
        "processed_data/sensor/clean_readings.csv",
        "processed_data/manifests/split_manifest.json",
        "processed_data/splits/train.csv",
        "reports/evaluation_summary.md",
    ]
    for p in paths_to_test:
        if is_ignored(p):
            return False, f"Path should be trackable in repository but was ignored: {p}"
    return True, "models/, processed_data/, manifests, and reports are confirmed trackable."


def verify_gitattributes() -> tuple[bool, str]:
    """Criterion 4: Verify .gitattributes exists and configures LF normalization and notebook output filter."""
    ga_path = REPO_ROOT / ".gitattributes"
    if not ga_path.exists():
        return False, ".gitattributes does not exist."
    content = ga_path.read_text(encoding="utf-8")
    if "eol=lf" not in content:
        return False, ".gitattributes does not specify LF line endings."
    if "filter=nbstripout" not in content:
        return False, ".gitattributes does not configure notebook stripping filter."
    return True, ".gitattributes is properly configured."


def main() -> int:
    checks = [
        ("Env and env.example rules", verify_env_rules),
        ("Datasets remain tracked", verify_datasets_tracked),
        ("Models and processed_data trackability", verify_models_and_processed_data_trackable),
        (".gitattributes configuration", verify_gitattributes),
    ]

    all_passed = True
    print("================================================================================")
    print("VERIFY GIT RULES & ATTRIBUTES (T1-003)")
    print("================================================================================\n")

    for name, check_fn in checks:
        ok, msg = check_fn()
        status_str = "[PASS]" if ok else "[FAIL]"
        print(f"{status_str} {name}: {msg}")
        if not ok:
            all_passed = False

    print("\n--------------------------------------------------------------------------------")
    if all_passed:
        print("All criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
