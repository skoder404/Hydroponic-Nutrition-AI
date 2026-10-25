#!/usr/bin/env python3
"""Create stratified 70/15/15 image split manifests across crops (T2-015).

Methodology & Protocol:
-----------------------
1. Duplicate Isolation:
   - Ingests exact and near-duplicate audit reports (T2-014) from `reports/data/image_duplicates_{crop}.json`.
   - Connected duplicate clusters (exact SHA-256 and dHash <= 4) are grouped into atomic units using Disjoint Set Union (Union-Find).
   - Duplicate groups are kept strictly together in the same split partition to prevent train/test contamination.

2. Stratified Partitioning:
   - Target ratios: 70% Train, 15% Validation, 15% Test.
   - Stratified per class using largest-first greedy allocation with a fixed, documented seed (default: 42).
   - Leaf crops extracted later (Phase 5) will inherit the split of their source image.

3. Outputs:
   - JSON split manifests per crop: `processed_data/splits/image_splits_{crop}.json`
   - CSV split manifests per crop: `processed_data/splits/image_splits_{crop}.csv`
   - Consolidated summary manifest: `processed_data/splits/image_splits_manifest.json`

Re-run command:
    python scripts/make_image_splits.py --crop all --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DATA_DIR = PROJECT_ROOT / "image_data"
DUPLICATE_REPORT_DIR = PROJECT_ROOT / "reports" / "data"
SPLITS_OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "splits"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CROPS = ["lettuce", "spinach", "tomato"]
DEFAULT_SEED = 42
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class DisjointSetUnion:
    """Iterative Union-Find with path compression for duplicate clustering."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def load_duplicate_groups(crop: str) -> DisjointSetUnion:
    """Load duplicate clusters from T2-014 audit report and union them."""
    dup_file = DUPLICATE_REPORT_DIR / f"image_duplicates_{crop}.json"
    uf = DisjointSetUnion()

    if not dup_file.exists():
        print(f"WARNING: Duplicate report {dup_file} not found; treating all images as singletons.")
        return uf

    with open(dup_file, encoding="utf-8") as f:
        data = json.load(f)

    # Union exact duplicate groups
    for group in data.get("exact_duplicates", {}).get("groups", []):
        files = group.get("files", [])
        for i in range(1, len(files)):
            uf.union(files[0], files[i])

    # Union near-duplicate clusters
    for cluster in data.get("near_duplicates", {}).get("clusters", []):
        files = cluster.get("files", [])
        for i in range(1, len(files)):
            uf.union(files[0], files[i])

    return uf


def create_crop_split(
    crop: str,
    seed: int = DEFAULT_SEED,
    output_dir: Path = SPLITS_OUTPUT_DIR,
) -> dict[str, Any]:
    """Partition whole crop images into 70/15/15 train/val/test splits."""
    crop_folder = IMAGE_DATA_DIR / f"{crop.capitalize()}_Images"
    if not crop_folder.exists():
        raise FileNotFoundError(f"Crop directory not found: {crop_folder}")

    # Gather all valid images
    all_image_paths = sorted([
        f.relative_to(PROJECT_ROOT).as_posix()
        for f in crop_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith(".")
    ])
    total_images = len(all_image_paths)

    # Cluster by duplicate components
    uf = load_duplicate_groups(crop)
    components: dict[str, list[str]] = defaultdict(list)
    for p in all_image_paths:
        root = uf.find(p)
        components[root].append(p)

    total_class_counts = Counter(Path(f).parent.name for f in all_image_paths)

    targets = {
        "train": {c: round(cnt * TARGET_RATIOS["train"]) for c, cnt in total_class_counts.items()},
        "val": {c: round(cnt * TARGET_RATIOS["val"]) for c, cnt in total_class_counts.items()},
        "test": {
            c: cnt - round(cnt * TARGET_RATIOS["train"]) - round(cnt * TARGET_RATIOS["val"])
            for c, cnt in total_class_counts.items()
        },
    }

    current_counts = {
        "train": {c: 0 for c in total_class_counts},
        "val": {c: 0 for c in total_class_counts},
        "test": {c: 0 for c in total_class_counts},
    }

    rng = random.Random(seed)
    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []

    class_counts_by_split: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    # Sort groups descending by size with deterministic tie-breaking
    groups = list(components.values())
    decorated = [(len(g), rng.random(), g) for g in groups]
    decorated.sort(key=lambda item: (-item[0], item[1]))
    sorted_groups = [item[2] for item in decorated]

    for group in sorted_groups:
        g_classes = Counter(Path(f).parent.name for f in group)

        # Allocate to split with greatest deficit reduction
        best_split = "train"
        best_score = -float("inf")

        for sp in ("train", "val", "test"):
            score = sum(
                (targets[sp][c] - current_counts[sp][c]) * cnt
                for c, cnt in g_classes.items()
            )
            if score > best_score:
                best_score = score
                best_split = sp

        if best_split == "train":
            train_paths.extend(group)
        elif best_split == "val":
            val_paths.extend(group)
        else:
            test_paths.extend(group)

        for c, cnt in g_classes.items():
            current_counts[best_split][c] += cnt
            class_counts_by_split[best_split][c] += cnt

    # Invariant assertions
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    assert len(train_set & val_set) == 0, "Data leakage between train and val!"
    assert len(train_set & test_set) == 0, "Data leakage between train and test!"
    assert len(val_set & test_set) == 0, "Data leakage between val and test!"
    assert len(train_set | val_set | test_set) == total_images, "Image count mismatch!"

    # Verify duplicate groups kept together
    for root, files in components.items():
        if len(files) > 1:
            in_train = any(f in train_set for f in files)
            in_val = any(f in val_set for f in files)
            in_test = any(f in test_set for f in files)
            assert sum([in_train, in_val, in_test]) == 1, (
                f"Duplicate group spanning root {root} was split across partitions!"
            )

    # Format manifest
    manifest: dict[str, Any] = {
        "crop": crop,
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_ratios": TARGET_RATIOS,
        "total_images": total_images,
        "distinct_duplicate_groups": len(components),
        "split_counts": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
        "split_fractions": {
            "train": round(len(train_paths) / total_images, 4),
            "val": round(len(val_paths) / total_images, 4),
            "test": round(len(test_paths) / total_images, 4),
        },
        "class_counts": {
            split: dict(sorted(class_counts_by_split[split].items()))
            for split in ("train", "val", "test")
        },
        "splits": {
            "train": sorted(train_paths),
            "val": sorted(val_paths),
            "test": sorted(test_paths),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"image_splits_{crop}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also export CSV for convenient downstream ingestion
    csv_path = output_dir / f"image_splits_{crop}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "crop", "class", "split"])
        for sp_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for p in sorted(paths):
                writer.writerow([p, crop, Path(p).parent.name, sp_name])

    print(f"[{crop.upper()}] Split manifest created:")
    print(f"  - Total: {total_images} images")
    print(f"  - Train: {len(train_paths)} ({len(train_paths)/total_images:.1%})")
    print(f"  - Val:   {len(val_paths)} ({len(val_paths)/total_images:.1%})")
    print(f"  - Test:  {len(test_paths)} ({len(test_paths)/total_images:.1%})")
    print(f"  - JSON:  {json_path}")
    print(f"  - CSV:   {csv_path}\n")

    return manifest


def generate_consolidated_manifest(
    crop_manifests: dict[str, dict[str, Any]],
    output_dir: Path = SPLITS_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate consolidated master manifest across all crops."""
    summary: dict[str, Any] = {
        "title": "Hydroponic Nutrition AI — Image Split Manifests (T2-015)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target_ratios": TARGET_RATIOS,
        "crops": {},
    }

    for crop, data in crop_manifests.items():
        summary["crops"][crop] = {
            "total_images": data["total_images"],
            "distinct_duplicate_groups": data["distinct_duplicate_groups"],
            "split_counts": data["split_counts"],
            "split_fractions": data["split_fractions"],
            "class_counts": data["class_counts"],
            "manifest_file": f"image_splits_{crop}.json",
            "csv_file": f"image_splits_{crop}.csv",
        }

    summary_path = output_dir / "image_splits_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Master split manifest written to: {summary_path}")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create stratified 70/15/15 image split manifests (T2-015).")
    parser.add_argument(
        "--crop",
        choices=["lettuce", "spinach", "tomato", "all"],
        default="all",
        help="Target crop, or 'all' to split all crops (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic random seed for stratified group allocation (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLITS_OUTPUT_DIR,
        help="Directory to save split manifests (default: processed_data/splits).",
    )
    args = parser.parse_args()

    crops_to_process = CROPS if args.crop == "all" else [args.crop]

    print("================================================================================")
    print("CREATING STRATIFIED 70/15/15 IMAGE SPLIT MANIFESTS (T2-015)")
    print(f"Seed: {args.seed} | Target: 70% Train, 15% Val, 15% Test")
    print("Duplicate Group Preservation: Active (groups kept strictly in the same split)")
    print("================================================================================\n")

    manifests: dict[str, dict[str, Any]] = {}
    for crop in crops_to_process:
        manifest = create_crop_split(crop, seed=args.seed, output_dir=args.output_dir)
        manifests[crop] = manifest

    if len(manifests) > 1:
        generate_consolidated_manifest(manifests, output_dir=args.output_dir, seed=args.seed)

    print("--------------------------------------------------------------------------------")
    print("ALL IMAGE SPLIT MANIFESTS GENERATED AND VERIFIED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
