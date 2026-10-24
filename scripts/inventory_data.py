"""Data and scripts inventory inspection script for Hydroponic Nutrition AI.

Reproduces and verifies all file counts, formats, byte sizes, CSV row counts,
and schemas documented in docs/decisions/0001-data-inventory.md.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path


def format_bytes(size: int) -> str:
    """Format bytes into human-readable string."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB ({size:,} bytes)"
    if size >= 1024:
        return f"{size / 1024:.2f} KB ({size:,} bytes)"
    return f"{size:,} bytes"


def inspect_directory(base_dir: Path) -> dict[str, dict]:
    """Inspect all directories under base_dir recursively."""
    results: dict[str, dict] = {}
    for root, _dirs, files in os.walk(base_dir):
        root_path = Path(root)
        rel_path = str(root_path).replace("\\", "/")
        ext_counts: dict[str, int] = {}
        total_size = 0
        file_records = []

        for f in sorted(files):
            file_path = root_path / f
            size = file_path.stat().st_size
            total_size += size
            ext = file_path.suffix.lower() or "[no_ext]"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            file_records.append({"name": f, "size": size, "ext": ext})

        results[rel_path] = {
            "file_count": len(files),
            "total_size": total_size,
            "ext_counts": ext_counts,
            "files": file_records,
        }
    return results


def inspect_csv(file_path: Path) -> tuple[int, list[str]]:
    """Count data rows (excluding header) and extract column headers from CSV."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return 0, []
        row_count = sum(1 for _ in reader)
    return row_count, headers


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    print("================================================================================")
    print("HYDROPONIC NUTRITION AI - DATA INVENTORY VERIFICATION")
    print("================================================================================\n")

    for section in ["image_data", "sensor_data"]:
        section_path = repo_root / section
        if not section_path.exists():
            print(f"[WARN] Directory not found: {section}")
            continue

        print(f"--- Folder Inventory: {section}/ ---")
        dir_data = inspect_directory(section_path)
        for path_str, data in sorted(dir_data.items()):
            ext_summary = (
                ", ".join(f"{k}: {v}" for k, v in sorted(data["ext_counts"].items()))
                if data["ext_counts"]
                else "None (empty directory)"
            )
            print(
                f"  {path_str}\n"
                f"    Files: {data['file_count']} | Size: {format_bytes(data['total_size'])}\n"
                f"    Formats: [{ext_summary}]"
            )
        print()

    sensor_dir = repo_root / "sensor_data"
    if sensor_dir.exists():
        print("--- Sensor Data Files & Schema Breakdown ---")
        for root, _dirs, files in os.walk(sensor_dir):
            for file_name in sorted(files):
                fp = Path(root) / file_name
                rel_fp = fp.relative_to(repo_root).as_posix()
                size = fp.stat().st_size
                if fp.suffix.lower() == ".csv":
                    rows, headers = inspect_csv(fp)
                    print(f"  [CSV] {rel_fp}")
                    print(f"        Size: {format_bytes(size)} | Data Rows: {rows:,}")
                    print(f"        Columns ({len(headers)}): {headers}")
                else:
                    lines = sum(
                        1
                        for _ in fp.open(
                            "r", encoding="utf-8", errors="ignore"
                        )
                    )
                    ext_tag = fp.suffix or "[no_ext]"
                    print(f"  [DOC/SCRIPT] {rel_fp} (format: {ext_tag})")
                    print(f"        Size: {format_bytes(size)} | Lines: {lines:,}")
        print()


if __name__ == "__main__":
    main()
