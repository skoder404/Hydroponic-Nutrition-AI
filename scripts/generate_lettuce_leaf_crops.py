#!/usr/bin/env python3
"""Generate lettuce leaf crops from whole-plant images with source mapping (T5-012).

Workflow:
1. Read split manifest: `processed_data/splits/image_splits_lettuce.csv`.
2. Segment whole-plant images into individual leaves using SAM 2.1 via `extract_leaves()`.
3. Save masked leaf crops to `processed_data/lettuce_leaf_crops/crops/<crop_id>.png`.
4. Inherit source image split and class label (weak label).
5. Output metadata to `processed_data/lettuce_leaf_crops/source_map.csv`:
   - crop_id
   - source_image_id
   - crop_path
   - class
   - split
   - mask_verdict
6. Record and report zero-crop images.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.segmentation.leaf_extraction import LeafExtractionConfig, extract_leaves
from ml.segmentation.mask_validation import MaskVerdict
from ml.segmentation.sam2_segmenter import SAM2Segmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("generate_lettuce_leaf_crops")


def generate_crops(
    manifest_csv: Path = PROJECT_ROOT / "processed_data/splits/image_splits_lettuce.csv",
    output_dir: Path = PROJECT_ROOT / "processed_data/lettuce_leaf_crops",
    max_images: int | None = None,
    overwrite: bool = False,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Process all lettuce whole-plant images in the split manifest into leaf crops."""
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    source_map_path = output_dir / "source_map.csv"
    summary_path = output_dir / "summary.json"

    if source_map_path.exists() and not overwrite:
        LOGGER.info(f"Source map already exists at '{source_map_path}'. Use --overwrite to regenerate.")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    df_manifest = pd.read_csv(manifest_csv)
    if max_images is not None:
        df_manifest = df_manifest.head(max_images)

    total_images = len(df_manifest)
    LOGGER.info(f"Loaded {total_images} images from {manifest_csv}")

    LOGGER.info("Initializing SAM 2.1 segmenter...")
    segmenter = SAM2Segmenter(force_cpu=force_cpu)
    segmenter.load_model()
    LOGGER.info(f"SAM 2.1 loaded on {segmenter.loaded_device}")

    config = LeafExtractionConfig(
        single_leaf_mode=False,
        max_leaves=8,
        iou_overlap_threshold=0.50,
        background_fill=(0, 0, 0),
    )

    records: list[dict[str, Any]] = []
    zero_crop_images: list[dict[str, str]] = []
    start_time = time.time()

    for idx, row in tqdm(df_manifest.iterrows(), total=total_images, desc="Extracting lettuce leaf crops"):
        rel_img_path = row["image_path"]
        img_class = row["class"]
        img_split = row["split"]
        full_img_path = PROJECT_ROOT / rel_img_path

        source_img_id = Path(rel_img_path).stem

        if not full_img_path.exists():
            LOGGER.warning(f"File not found: {full_img_path}")
            zero_crop_images.append({
                "source_image_id": source_img_id,
                "image_path": rel_img_path,
                "reason": "File not found",
            })
            continue

        bgr_img = cv2.imread(str(full_img_path))
        if bgr_img is None:
            LOGGER.warning(f"Failed to read image: {full_img_path}")
            zero_crop_images.append({
                "source_image_id": source_img_id,
                "image_path": rel_img_path,
                "reason": "Corrupted or unreadable image",
            })
            continue

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        try:
            extraction_result = extract_leaves(rgb_img, segmenter=segmenter, config=config)
        except Exception as e:
            LOGGER.error(f"Error extracting leaves from {source_img_id}: {e}")
            zero_crop_images.append({
                "source_image_id": source_img_id,
                "image_path": rel_img_path,
                "reason": f"Extraction error: {e}",
            })
            continue

        valid_leaves = extraction_result.leaves

        if not valid_leaves:
            zero_crop_images.append({
                "source_image_id": source_img_id,
                "image_path": rel_img_path,
                "reason": "All candidate proposals rejected by mask validator",
            })
            continue

        for l_idx, leaf in enumerate(valid_leaves):
            crop_id = f"{source_img_id}_crop_{l_idx:02d}"
            crop_filename = f"{crop_id}.png"
            crop_dest_path = crops_dir / crop_filename

            # Save RGB crop (with masked background)
            crop_bgr = cv2.cvtColor(leaf.masked_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_dest_path), crop_bgr)

            records.append({
                "crop_id": crop_id,
                "source_image_id": source_img_id,
                "crop_path": f"processed_data/lettuce_leaf_crops/crops/{crop_filename}",
                "class": img_class,
                "split": img_split,
                "mask_verdict": MaskVerdict.OK.value,
                "confidence": round(float(leaf.confidence), 4),
                "area_pixels": int(leaf.metrics.mask_pixels),
                "area_ratio": round(float(leaf.metrics.area_ratio), 4),
            })

    elapsed = time.time() - start_time
    LOGGER.info(f"Extraction complete in {elapsed:.1f}s. Extracted {len(records)} leaf crops.")

    df_out = pd.DataFrame(records)
    df_out.to_csv(source_map_path, index=False)
    LOGGER.info(f"Saved source map with {len(df_out)} crops to {source_map_path}")

    summary = {
        "total_source_images": total_images,
        "total_valid_crops": len(records),
        "crops_per_image_mean": round(len(records) / max(1, total_images), 2),
        "zero_crop_count": len(zero_crop_images),
        "zero_crop_images": zero_crop_images,
        "crops_per_class": df_out["class"].value_counts().to_dict() if not df_out.empty else {},
        "crops_per_split": df_out["split"].value_counts().to_dict() if not df_out.empty else {},
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lettuce leaf crops from whole plant images")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed (for quick testing)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops and source map")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    summary = generate_crops(max_images=args.max_images, overwrite=args.overwrite, force_cpu=args.cpu)
    print(json.dumps(summary, indent=2))

