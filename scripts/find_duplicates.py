#!/usr/bin/env python3
"""Detect exact and near-duplicate images across crop datasets (T2-014).

Methodology & Architecture:
---------------------------
1. Exact Duplicates:
   - Cryptographic SHA-256 digest computed on raw file bytes.
   - Files sharing identical SHA-256 hashes are bitwise identical.

2. Near-Duplicates (Perceptual Hashing):
   - Library: Pillow (PIL) - `PIL.Image`.
   - Algorithm: Difference Hash (dHash), producing a 64-bit gradient fingerprint.
     The image is converted to 8-bit grayscale ('L') and scaled to 9x8 pixels.
     Adjacent horizontal pixels are compared across 8 rows to yield 64 bits.
   - Distance Metric: Hamming distance (number of differing bits between two 64-bit hashes).
   - Conservative Threshold: Hamming distance <= 4 bits (out of 64).
     Rationale: A distance of <= 4 bits detects near-identical images produced by burst
     captures, slight JPEG re-compression, or minor illumination shifts, while maintaining
     strict specificity against different leaf specimens.
   - Efficiency: Multi-index hashing with 5 sub-fingerprint blocks (Pigeonhole principle
     guarantees 100% recall for distance <= 4 with O(N) indexing instead of O(N^2) pairwise comparisons).
   - Grouping: Connected components (Disjoint Set Union) to group near-duplicate pairs into clusters.

Safety Guarantee:
-----------------
This tool is strictly an audit tool. It performs read-only operations on image files.
NO FILES ARE MODIFIED OR DELETED.

Re-run command:
    python scripts/find_duplicates.py all
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DATA_DIR = PROJECT_ROOT / "image_data"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CROP_FOLDERS = {
    "lettuce": IMAGE_DATA_DIR / "Lettuce_Images",
    "spinach": IMAGE_DATA_DIR / "Spinach_Images",
    "tomato": IMAGE_DATA_DIR / "Tomato_Images",
}

DEFAULT_HAMMING_THRESHOLD = 4


def compute_sha256(path: Path) -> str:
    """Compute cryptographic SHA-256 digest of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(path: Path) -> int | None:
    """Compute 64-bit difference hash (dHash) using Pillow.

    Resizes to 9x8 grayscale and computes horizontal gradient bits.
    Returns 64-bit unsigned integer or None if decoding fails.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())

        diff = 0
        for row in range(8):
            row_offset = row * 9
            for col in range(8):
                left = pixels[row_offset + col]
                right = pixels[row_offset + col + 1]
                diff = (diff << 1) | (1 if left > right else 0)
        return diff
    except Exception:  # noqa: BLE001
        return None


def hash_single_file(file_path: Path) -> dict[str, Any]:
    """Worker task: extract SHA-256 and dHash for one image file."""
    sha = compute_sha256(file_path)
    dh = compute_dhash(file_path)
    rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    class_name = file_path.parent.name
    return {
        "path": rel_path,
        "class": class_name,
        "sha256": sha,
        "dhash": dh,
    }


def find_near_duplicate_pairs(
    records: list[dict[str, Any]],
    threshold: int,
) -> list[dict[str, Any]]:
    """Find pairs of images with dHash Hamming distance <= threshold using multi-index hashing."""
    # 5 blocks for 64 bits: 13, 13, 13, 13, 12 bits
    # By pigeonhole principle, if distance <= 4, at least one block has 0 differences.
    shifts = [0, 13, 26, 39, 52]
    masks = [0x1FFF, 0x1FFF, 0x1FFF, 0x1FFF, 0xFFF]
    block_indices: list[defaultdict[int, list[int]]] = [defaultdict(list) for _ in range(5)]

    valid_records = [(idx, r) for idx, r in enumerate(records) if r["dhash"] is not None]

    candidate_pairs: set[tuple[int, int]] = set()

    for idx, r in valid_records:
        h = r["dhash"]
        for b_idx, (shift, mask) in enumerate(zip(shifts, masks)):
            key = (h >> shift) & mask
            for other_idx in block_indices[b_idx][key]:
                pair = (min(idx, other_idx), max(idx, other_idx))
                candidate_pairs.add(pair)
            block_indices[b_idx][key].append(idx)

    pairs: list[dict[str, Any]] = []
    for i1, i2 in candidate_pairs:
        r1 = records[i1]
        r2 = records[i2]
        # Ignore pairs that are already exact duplicates
        if r1["sha256"] == r2["sha256"]:
            continue

        dist = (r1["dhash"] ^ r2["dhash"]).bit_count()
        if dist <= threshold:
            pairs.append({
                "file1": r1["path"],
                "class1": r1["class"],
                "file2": r2["path"],
                "class2": r2["class"],
                "hamming_distance": dist,
                "dhash1": f"{r1['dhash']:016x}",
                "dhash2": f"{r2['dhash']:016x}",
                "same_class": r1["class"] == r2["class"],
            })

    # Sort pairs by distance then paths
    pairs.sort(key=lambda p: (p["hamming_distance"], p["file1"], p["file2"]))
    return pairs


def cluster_components(pairs: list[dict[str, Any]], key1: str, key2: str) -> list[list[str]]:
    """Group pairwise connections into connected components (clusters)."""
    adj = defaultdict(set)
    nodes = set()
    for p in pairs:
        u = p[key1]
        v = p[key2]
        adj[u].add(v)
        adj[v].add(u)
        nodes.add(u)
        nodes.add(v)

    visited = set()
    clusters = []
    for node in sorted(nodes):
        if node not in visited:
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(sorted(component))
    return sorted(clusters, key=lambda c: (-len(c), c[0]))


def audit_crop_duplicates(
    crop: str,
    output_dir: Path,
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Run exact and near-duplicate audit for a single crop."""
    crop_folder = CROP_FOLDERS[crop]
    print(f"\n[{crop.upper()}] Scanning folder: {crop_folder}")

    if not crop_folder.exists():
        print(f"ERROR: Folder {crop_folder} does not exist.")
        return {}

    image_files = [
        p for p in sorted(crop_folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]
    total_images = len(image_files)
    print(f"[{crop.upper()}] Found {total_images} image files. Extracting SHA-256 and dHash fingerprints...")

    # Parallel hash extraction
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(hash_single_file, image_files))

    # 1. Exact duplicates (SHA-256)
    sha_map = defaultdict(list)
    for r in records:
        sha_map[r["sha256"]].append(r)

    exact_groups: list[dict[str, Any]] = []
    total_exact_duplicate_files = 0
    for sha, group_records in sorted(sha_map.items()):
        if len(group_records) > 1:
            paths = [r["path"] for r in group_records]
            classes = list({r["class"] for r in group_records})
            exact_groups.append({
                "sha256": sha,
                "file_count": len(paths),
                "files": paths,
                "classes": classes,
                "cross_class": len(classes) > 1,
            })
            total_exact_duplicate_files += len(paths)

    # 2. Near-duplicates (dHash)
    near_pairs = find_near_duplicate_pairs(records, threshold=threshold)
    near_clusters = cluster_components(near_pairs, "file1", "file2")
    cross_class_near_pairs = [p for p in near_pairs if not p["same_class"]]

    report: dict[str, Any] = {
        "crop": crop,
        "source_folder": str(crop_folder.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_images_scanned": total_images,
        "methodology": {
            "library": "Pillow (PIL)",
            "exact_hash_algorithm": "SHA-256",
            "perceptual_hash_algorithm": "dHash (64-bit difference hash)",
            "hamming_distance_threshold": threshold,
            "threshold_rationale": (
                f"Hamming distance <= {threshold} out of 64 bits identifies visually identical images, "
                "burst shots, or minor re-compression artifacts while preventing false positives across distinct leaves."
            ),
            "safety_policy": "Non-destructive audit only. No files were modified or deleted.",
        },
        "exact_duplicates": {
            "total_groups": len(exact_groups),
            "total_files_in_groups": total_exact_duplicate_files,
            "cross_class_groups_count": sum(1 for g in exact_groups if g["cross_class"]),
            "groups": exact_groups,
        },
        "near_duplicates": {
            "total_pairs": len(near_pairs),
            "total_clusters": len(near_clusters),
            "cross_class_pairs_count": len(cross_class_near_pairs),
            "clusters": [
                {
                    "cluster_id": i + 1,
                    "file_count": len(c),
                    "files": c,
                }
                for i, c in enumerate(near_clusters)
            ],
            "pairs_sample": near_pairs[:50],  # first 50 pairs for JSON brevity
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"image_duplicates_{crop}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[{crop.upper()}] Audit complete:")
    print(f"  - Exact duplicate groups: {len(exact_groups)} ({total_exact_duplicate_files} files)")
    print(f"  - Near duplicate pairs (dist <= {threshold}): {len(near_pairs)} ({len(near_clusters)} clusters)")
    print(f"  - Cross-class near pairs: {len(cross_class_near_pairs)}")
    print(f"  - Report saved to: {report_file}")

    return report


def generate_consolidated_markdown(
    all_reports: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate consolidated markdown summary across all crops."""
    md_lines: list[str] = [
        "# Image Duplicate Audit Report (T2-014)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/find_duplicates.py`",
        "**Safety Policy:** Non-destructive audit report only. No image files were modified or deleted.",
        "",
        "## 1. Methodology, Libraries & Thresholds",
        "",
        "- **Exact Duplicates:** `SHA-256` digest on raw file bytes.",
        "- **Near-Duplicates:** Pillow (`PIL.Image`) difference hash (`dHash`, 64-bit gradient).",
        "- **Perceptual Distance Threshold:** Hamming distance **≤ 4** bits (out of 64).",
        (
            "- **Threshold Rationale:** Conservative threshold chosen to capture identical/burst-shot "
            "photographs and slight compression differences without erroneously matching distinct plant specimens."
        ),
        "- **Search Optimization:** Multi-index inverted table (5 sub-fingerprint blocks) guaranteeing 100% recall.",
        "",
        "## 2. Summary of Findings by Crop",
        "",
        "| Crop | Total Images | Exact Groups | Exact Files | Near Pairs (≤4) | Near Clusters | Cross-Class Near Pairs |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for crop, rep in all_reports.items():
        total = rep.get("total_images_scanned", 0)
        ex_grp = rep.get("exact_duplicates", {}).get("total_groups", 0)
        ex_files = rep.get("exact_duplicates", {}).get("total_files_in_groups", 0)
        nr_pairs = rep.get("near_duplicates", {}).get("total_pairs", 0)
        nr_clust = rep.get("near_duplicates", {}).get("total_clusters", 0)
        cc_pairs = rep.get("near_duplicates", {}).get("cross_class_pairs_count", 0)
        md_lines.append(f"| `{crop}` | {total:,} | {ex_grp} | {ex_files} | {nr_pairs:,} | {nr_clust:,} | {cc_pairs:,} |")

    md_lines.extend([
        "",
        "## 3. Analysis & Implications for ML Splits",
        "",
        "1. **Exact Duplicates:**",
        "   - Identical byte-for-byte copies must be isolated so they never appear on both sides of a train/test split.",
        "2. **Near-Duplicates & Burst Capture Contamination:**",
        "   - Near-duplicates indicate multiple frames photographed from the same plant head or specimen.",
        "   - If near-duplicates leak across train and test sets, model accuracy is artificially inflated.",
        "   - Splits must be grouped by plant specimen/session rather than naive random sampling.",
        "3. **Cross-Class Near-Duplicates:**",
        "   - Cross-class near pairs highlight potential label noise or ambiguous border conditions between deficiency stages.",
        "",
        "---",
        "*Detailed per-crop JSON reports available at `reports/data/image_duplicates_{crop}.json`.*",
    ])

    summary_file = output_dir / "image_duplicates_summary.md"
    summary_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nConsolidated markdown report written to: {summary_file}")
    return summary_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit image datasets for exact and near duplicates (T2-014).")
    parser.add_argument(
        "crop",
        nargs="?",
        choices=["lettuce", "spinach", "tomato", "all"],
        default="all",
        help="Crop to audit, or 'all' to audit lettuce, spinach, and tomato (default: all).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_HAMMING_THRESHOLD,
        help=f"Hamming distance threshold for near-duplicates (default: {DEFAULT_HAMMING_THRESHOLD}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_OUTPUT_DIR,
        help="Directory to save JSON and markdown reports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) * 2),
        help="Number of worker threads for parallel hash extraction.",
    )
    args = parser.parse_args()

    crops_to_run = ["lettuce", "spinach", "tomato"] if args.crop == "all" else [args.crop]

    print("================================================================================")
    print("IMAGE DATASET DUPLICATE AUDIT (T2-014)")
    print(f"Perceptual Library: Pillow (PIL) | Hash: dHash (64-bit) | Threshold: <= {args.threshold}")
    print("Policy: Read-only non-destructive audit (no files will be modified or deleted)")
    print("================================================================================")

    all_reports: dict[str, dict[str, Any]] = {}
    for crop in crops_to_run:
        rep = audit_crop_duplicates(crop, args.output_dir, threshold=args.threshold, max_workers=args.workers)
        all_reports[crop] = rep

    generate_consolidated_markdown(all_reports, args.output_dir)

    print("\n--------------------------------------------------------------------------------")
    print("DUPLICATE AUDIT COMPLETED SUCCESSFULLY FOR ALL REQUESTED CROPS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
