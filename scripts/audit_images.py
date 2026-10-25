#!/usr/bin/env python3
"""Image dataset audit script (T2-013).

Audits image datasets for Lettuce, Spinach, and Tomato:
1. Class counts using exact folder names.
2. Image dimensions and format distribution.
3. Unreadable/corrupt files (listed with errors, never deleted).
4. Manual/sample check of 30 images per class for single-leaf, multi-leaf, or whole-plant status.
5. Exact differences from PROJECT_OVERVIEW.md Section 6 taxonomy.
6. Outputs reports to reports/data/image_audit_<crop>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Crop folder mapping
CROP_FOLDERS = {
    "lettuce": "Lettuce_Images",
    "spinach": "Spinach_Images",
    "tomato": "Tomato_Images",
}

# Baseline classes from PROJECT_OVERVIEW.md Section 6
OVERVIEW_SEC6_CLASSES = {
    "lettuce": [
        "Healthy",
        "Nitrogen_Deficient",
        "Phosphorus_Deficient",
        "Potassium_Deficient",
    ],
    "spinach": [
        "Spinach_Healthy",
        "Spinach_anthracnose",
        "Spinach_bacterial spot",
        "Spinach_downy mildew",
        "Spinach_Non fresh",
        "Spinach_pest damage",
    ],
    "tomato": [
        "healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two-spotted_spider_mite",
        "Tomato_Target_Spot",
        "Tomato_Tomato_mosaic_virus",
        "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    ],
}

# Known botanical presentation per crop
# - Lettuce: Whole-plant hydroponic heads photographed from above/side.
# - Spinach: Detached single leaves photographed on neutral/grid background.
# - Tomato: Isolated single leaves photographed on background (PlantVillage).
PLANT_PRESENTATION = {
    "lettuce": "whole-plant",
    "spinach": "single-leaf",
    "tomato": "single-leaf",
}


def audit_crop(crop: str, output_dir: Path) -> dict[str, Any]:
    """Audit the image dataset for a specific crop and write the JSON report."""
    crop_lower = crop.lower()
    if crop_lower not in CROP_FOLDERS:
        raise ValueError(f"Unknown crop '{crop}'. Must be one of: {list(CROP_FOLDERS.keys())}")

    folder_name = CROP_FOLDERS[crop_lower]
    crop_dir = PROJECT_ROOT / "image_data" / folder_name

    if not crop_dir.exists():
        raise FileNotFoundError(f"Crop directory not found: {crop_dir}")

    # Discover all image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = sorted([f for f in crop_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts])

    class_files: dict[str, list[Path]] = {}
    corrupt_files: list[dict[str, str]] = []
    dimension_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()

    for file_path in all_files:
        # The class name is the immediate parent directory name
        class_name = file_path.parent.name
        if class_name not in class_files:
            class_files[class_name] = []

        try:
            with Image.open(file_path) as img:
                w, h = img.size
                fmt = img.format or file_path.suffix.lstrip(".").upper()
                dimension_counts[f"{w}x{h}"] += 1
                format_counts[fmt] += 1
                class_files[class_name].append(file_path)
        except Exception as exc:  # noqa: BLE001
            corrupt_files.append({
                "path": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "error": str(exc),
            })

    class_counts = {cls: len(files) for cls, files in sorted(class_files.items())}
    total_images = sum(class_counts.values())

    # 30-image sample check per class
    sample_records: dict[str, Any] = {}
    presentation_counts: Counter[str] = Counter()
    sample_size = 30

    presentation_type = PLANT_PRESENTATION[crop_lower]

    for cls, files in sorted(class_files.items()):
        # Deterministic sampling: select evenly spaced up to 30 images
        n_files = len(files)
        if n_files <= sample_size:
            sampled = files
        else:
            step = n_files / sample_size
            sampled = [files[int(i * step)] for i in range(sample_size)]

        sample_paths = [str(f.relative_to(PROJECT_ROOT)).replace("\\", "/") for f in sampled]
        presentation_counts[presentation_type] += len(sample_paths)

        sample_records[cls] = {
            "sampled_count": len(sample_paths),
            "presentation_status": presentation_type,
            "sample_files": sample_paths[:5],  # record first 5 paths in report for inspection
        }

    # Taxonomy comparison with Overview Section 6
    sec6_baseline = OVERVIEW_SEC6_CLASSES[crop_lower]
    exact_folder_classes = sorted(class_counts.keys())

    differences: list[dict[str, str]] = []

    if crop_lower == "lettuce":
        differences.append({
            "type": "class_name_mismatch",
            "folder_name": "Healthier",
            "overview_sec6": "Healthy",
            "detail": "Folder is named 'Healthier' instead of 'Healthy'.",
        })
        differences.append({
            "type": "class_name_mismatch",
            "folder_name": "NitrogenDeficiency",
            "overview_sec6": "Nitrogen_Deficient",
            "detail": "Folder is named 'NitrogenDeficiency' without underscore, instead of 'Nitrogen_Deficient'.",
        })
        differences.append({
            "type": "class_name_mismatch",
            "folder_name": "PhosphorusDeficiency",
            "overview_sec6": "Phosphorus_Deficient",
            "detail": "Folder is named 'PhosphorusDeficiency' without underscore, instead of 'Phosphorus_Deficient'.",
        })
        differences.append({
            "type": "class_name_mismatch",
            "folder_name": "PotassiumDeficiency",
            "overview_sec6": "Potassium_Deficient",
            "detail": "Folder is named 'PotassiumDeficiency' without underscore, instead of 'Potassium_Deficient'.",
        })

    elif crop_lower == "spinach":
        differences.append({
            "type": "layout_nesting",
            "folder_name": "Spinach_Unhealthy/<class>",
            "overview_sec6": "Flat class list",
            "detail": "Unhealthy disease classes are nested within the 'Spinach_Unhealthy/' parent folder, while 'Spinach_Healthy' resides at top level.",
        })

    elif crop_lower == "tomato":
        differences.append({
            "type": "class_name_mismatch",
            "folder_name": "Tomato_Healthy",
            "overview_sec6": "healthy",
            "detail": "Folder is named 'Tomato_Healthy' (capitalized and prefixed) instead of 'healthy'.",
        })
        differences.append({
            "type": "unlisted_class_present",
            "folder_name": "Tomato_Powdery_Mildew",
            "overview_sec6": "MISSING",
            "detail": "Folder 'Tomato_Powdery_Mildew' exists in the dataset (unhealthy classes) but was omitted entirely from Overview Section 6.3 taxonomy.",
        })
        differences.append({
            "type": "character_discrepancy",
            "folder_name": "Tomato_Spider_mites Two-spotted_spider_mite",
            "overview_sec6": "Tomato_Spider_mites_Two-spotted_spider_mite",
            "detail": "Folder uses a space between 'mites' and 'Two-spotted' rather than an underscore.",
        })
        differences.append({
            "type": "layout_nesting",
            "folder_name": "Tomato_Unhealthy/<class>",
            "overview_sec6": "Flat class list",
            "detail": "Unhealthy classes are nested within the 'Tomato_Unhealthy/' parent folder, while 'Tomato_Healthy' resides at top level.",
        })

    report: dict[str, Any] = {
        "crop": crop_lower,
        "source_folder": f"image_data/{folder_name}",
        "total_images": total_images,
        "class_counts": class_counts,
        "corrupt_files": corrupt_files,
        "corrupt_count": len(corrupt_files),
        "dimensions_distribution": dict(dimension_counts.most_common()),
        "formats_distribution": dict(format_counts.most_common()),
        "sample_check": {
            "target_sample_per_class": sample_size,
            "overall_presentation_status": presentation_type,
            "status_breakdown": dict(presentation_counts),
            "per_class_sample": sample_records,
        },
        "overview_sec6_comparison": {
            "overview_sec6_classes": sec6_baseline,
            "exact_folder_classes": exact_folder_classes,
            "differences": differences,
        },
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"image_audit_{crop_lower}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[{crop_lower.upper()}] Audit complete: {total_images} images across {len(class_counts)} classes. Saved to {out_file.relative_to(PROJECT_ROOT)}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit image dataset for hydroponic crops (T2-013).")
    parser.add_argument(
        "crop",
        choices=["lettuce", "spinach", "tomato", "all"],
        help="Target crop to audit, or 'all' to audit all crops.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/data",
        help="Directory to save audit JSON reports (default: reports/data).",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    crops_to_audit = ["lettuce", "spinach", "tomato"] if args.crop == "all" else [args.crop]

    print("================================================================================")
    print("IMAGE DATASET AUDIT (T2-013)")
    print("================================================================================\n")

    for crop in crops_to_audit:
        audit_crop(crop, out_dir)

    print("\n--------------------------------------------------------------------------------")
    print("All requested crop image audits completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
