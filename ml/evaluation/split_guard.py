"""Split manifest hash guard to protect prepared splits from tampering or leakage (T2-019).

Overview §9.2: Prepared dataset splits are immutable baselines. They must never be
overwritten, resampled, or altered during model training or evaluation.

This module computes and verifies SHA-256 digests of all prepared split files:
- Sensor split CSVs and manifests (processed_data/sensor/*/{train,val,test}.csv, manifest.json)
- Image split CSVs and manifests (processed_data/splits/image_splits_*.csv, *.json)

The reference hashes are stored in `processed_data/split_hashes.json`.
Training and evaluation entry points must call `assert_splits_intact()` at startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001, S110
        pass

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HASH_FILE = ROOT / "processed_data" / "split_hashes.json"


TRACKED_PATTERNS: list[str] = [
    "processed_data/sensor/*/*.csv",
    "processed_data/sensor/*/manifest.json",
    "processed_data/splits/image_splits_*.csv",
    "processed_data/splits/image_splits_*.json",
]


class SplitTamperError(Exception):
    """Raised when prepared split files have been altered, corrupted, or are missing."""


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_tracked_files() -> list[Path]:
    """Find all files in repository matching tracked split patterns."""
    matched: list[Path] = []
    for pattern in TRACKED_PATTERNS:
        matched.extend(ROOT.glob(pattern))
    return sorted(set(matched))


def generate_split_hashes(
    output_file: Path | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Generate SHA-256 digests for all tracked split files and save to JSON.

    Args:
        output_file: Custom output destination; defaults to processed_data/split_hashes.json.
        root: Repo root; defaults to project root.

    Returns:
        Dictionary mapping posix relative paths to SHA-256 digests.
    """
    base = root or ROOT
    out_path = output_file or DEFAULT_HASH_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracked = get_tracked_files()
    if not tracked:
        raise FileNotFoundError("No tracked split files found to hash.")

    hashes: dict[str, str] = {}
    for f in tracked:
        rel = f.relative_to(base).as_posix()
        hashes[rel] = compute_file_sha256(f)

    data = {
        "version": "1.0",
        "description": "Cryptographic SHA-256 checksums of all immutable dataset splits (Overview §9.2).",
        "total_files": len(hashes),
        "files": hashes,
    }

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return hashes


def verify_split_hashes(
    hash_file: Path | None = None,
    root: Path | None = None,
    *,
    raise_on_failure: bool = True,
) -> tuple[bool, dict[str, str]]:
    """Verify that current split files match recorded SHA-256 digests.

    Args:
        hash_file: Path to split_hashes.json.
        root: Repo root.
        raise_on_failure: If True, raises SplitTamperError on mismatch or missing file.

    Returns:
        (is_valid, errors): Tuple of boolean status and dictionary of errors per file.
    """
    base = root or ROOT
    h_file = hash_file or DEFAULT_HASH_FILE

    if not h_file.is_file():
        msg = f"Split hash guard file missing: {h_file}"
        if raise_on_failure:
            raise SplitTamperError(msg)
        return False, {"__all__": msg}

    try:
        content = json.loads(h_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in split hash guard file {h_file}: {exc}"
        if raise_on_failure:
            raise SplitTamperError(msg) from exc
        return False, {"__all__": msg}

    recorded = content.get("files", {})
    if not recorded:
        msg = f"No tracked file hashes recorded in {h_file}"
        if raise_on_failure:
            raise SplitTamperError(msg)
        return False, {"__all__": msg}

    errors: dict[str, str] = {}

    for rel_path, expected_hash in recorded.items():
        file_path = base / rel_path
        if not file_path.is_file():
            errors[rel_path] = f"MISSING: File does not exist at {file_path}"
            continue

        actual_hash = compute_file_sha256(file_path)
        if actual_hash != expected_hash:
            errors[rel_path] = (
                f"TAMPERED: SHA-256 mismatch!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )

    is_valid = len(errors) == 0

    if not is_valid and raise_on_failure:
        summary = "\n".join(f" - {f}: {err}" for f, err in errors.items())
        raise SplitTamperError(
            f"Split integrity violation! {len(errors)} split file(s) have been altered or are missing:\n{summary}\n"
            f"Prepared dataset splits are immutable (Overview §9.2). Revert changes to restore integrity."
        )

    return is_valid, errors


def assert_splits_intact(
    crop: str | None = None,
    modality: str | None = None,
    hash_file: Path | None = None,
    root: Path | None = None,
) -> None:
    """Enforce split integrity guard at model training or evaluation entry points.

    If crop or modality is specified, validates relevant subset of files while still
    verifying overall split integrity. Aborts with SplitTamperError on any violation.
    """
    base = root or ROOT
    h_file = hash_file or DEFAULT_HASH_FILE

    is_valid, errors = verify_split_hashes(h_file, base, raise_on_failure=False)
    if is_valid:
        return

    # Filter errors if crop or modality specified
    relevant_errors: dict[str, str] = {}
    for rel_path, err in errors.items():
        match_crop = crop is None or crop.lower() in rel_path.lower()
        match_mod = modality is None or modality.lower() in rel_path.lower()
        if match_crop and match_mod:
            relevant_errors[rel_path] = err

    if relevant_errors:
        summary = "\n".join(f" - {f}: {err}" for f, err in relevant_errors.items())
        raise SplitTamperError(
            f"Split integrity violation detected for target '{crop or 'any'}' / '{modality or 'any'}':\n{summary}\n"
            f"Aborting training/evaluation. Prepared dataset splits must remain untouched (Overview §9.2)."
        )

    # If general integrity broke on other files, still raise error
    summary = "\n".join(f" - {f}: {err}" for f, err in errors.items())
    raise SplitTamperError(
        f"Split integrity violation detected across workspace:\n{summary}\n"
        f"Aborting training/evaluation. Prepared dataset splits must remain untouched (Overview §9.2)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split Manifest Hash Guard (T2-019)")
    parser.add_argument("--generate", action="store_true", help="Generate/update split_hashes.json")
    parser.add_argument("--verify", action="store_true", help="Verify all tracked split files")
    parser.add_argument("--output", type=Path, default=None, help="Custom hash file location")
    args = parser.parse_args(argv)

    if args.generate:
        hashes = generate_split_hashes(output_file=args.output)
        print(f"✅ Generated SHA-256 checksums for {len(hashes)} split files in {args.output or DEFAULT_HASH_FILE}")
        return 0

    # Default to verify
    try:
        is_valid, errors = verify_split_hashes(hash_file=args.output, raise_on_failure=False)
        if is_valid:
            print("✅ All prepared split files and manifests match recorded cryptographic checksums.")
            return 0
        print(f"❌ Split integrity check FAILED! {len(errors)} error(s) found:")
        for rel_path, err in errors.items():
            print(f"   {rel_path}: {err}")
        return 1
    except SplitTamperError as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
