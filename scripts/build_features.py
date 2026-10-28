#!/usr/bin/env python3
"""Build feature datasets from split manifests (T5-013).

Workflow:
1. `scripts/build_features.py <crop> [--overwrite] [--cpu]`:
   - Validates supported crops: `lettuce`, `spinach`, `tomato`, `all`.
   - Re-running without `--overwrite` refuses to overwrite existing files.
2. For each crop:
   - Lettuce: Reads `processed_data/lettuce_leaf_crops/source_map.csv`.
     Extracts 7 features per Overview §16.1:
     `exg_mean`, `exr_mean`, `yellow_pixel_ratio`, `brown_pixel_ratio`, `mean_hue`, `mean_saturation`, `contour_compactness`.
     Crops inherit their source image's split and class.
   - Spinach: Reads `processed_data/splits/image_splits_spinach.csv`.
     Extracts 6 features per Overview §16.2:
     `exg_mean`, `exr_mean`, `yellow_pixel_ratio`, `brown_pixel_ratio`, `relative_leaf_area`, `glcm_contrast`.
   - Tomato: Reads `processed_data/splits/image_splits_tomato.csv`.
     Extracts 4 features per ADR 0007:
     `glcm_contrast`, `glcm_entropy`, `bounding_aspect_ratio`, `solidity`.
3. Validates that no NaN/Inf rows are written (`assert_finite_features`).
4. Writes per-split CSVs:
   - `processed_data/features/<crop>/train.csv`
   - `processed_data/features/<crop>/val.csv`
   - `processed_data/features/<crop>/test.csv`
   - `processed_data/features/<crop>/metadata.json`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.common import (  # noqa: E402
    DEFAULT_FEATURE_SCHEMA_VERSION,
    FeatureExtractionError,
)
from ml.features.lettuce import (  # noqa: E402
    DEFAULT_LETTUCE_CONFIG,
    LETTUCE_FEATURE_NAMES,
    extract_lettuce_features,
)
from ml.features.spinach import (  # noqa: E402
    DEFAULT_SPINACH_CONFIG,
    SPINACH_FEATURE_NAMES,
    extract_spinach_features,
)
from ml.features.tomato import (  # noqa: E402
    DEFAULT_TOMATO_CONFIG,
    TOMATO_FEATURE_NAMES,
    extract_tomato_features,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("build_features")

CROP_FEATURE_NAMES = {
    "lettuce": LETTUCE_FEATURE_NAMES,
    "spinach": SPINACH_FEATURE_NAMES,
    "tomato": TOMATO_FEATURE_NAMES,
}


def build_lettuce_features(
    output_dir: Path,
    overwrite: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Extract features for lettuce using pre-segmented leaf crops from T5-012."""
    source_map_path = PROJECT_ROOT / "processed_data/lettuce_leaf_crops/source_map.csv"
    if not source_map_path.exists():
        raise FileNotFoundError(
            f"Lettuce source map not found at {source_map_path}. Run T5-012 crop extraction first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    metadata_path = output_dir / "metadata.json"

    if (
        train_path.exists() or val_path.exists() or test_path.exists()
    ) and not overwrite:
        LOGGER.info(
            f"Lettuce features already exist in {output_dir}. Refusing to overwrite without --overwrite."
        )
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "skipped", "reason": "existing files without overwrite"}

    df_source = pd.read_csv(source_map_path)
    if max_samples is not None:
        df_source = df_source.head(max_samples)

    LOGGER.info(f"Building lettuce features for {len(df_source)} leaf crops...")
    records: list[dict[str, Any]] = []
    failed_crops: list[dict[str, str]] = []

    for idx, row in tqdm(
        df_source.iterrows(), total=len(df_source), desc="Extracting lettuce features"
    ):
        crop_id = row["crop_id"]
        crop_rel_path = row["crop_path"]
        crop_full_path = PROJECT_ROOT / crop_rel_path
        crop_class = row["class"]
        crop_split = row["split"]
        source_img_id = row["source_image_id"]

        if not crop_full_path.exists():
            failed_crops.append({"crop_id": crop_id, "reason": "Crop file missing"})
            continue

        bgr = cv2.imread(str(crop_full_path))
        if bgr is None:
            failed_crops.append({"crop_id": crop_id, "reason": "Unreadable image file"})
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Foreground mask: non-zero pixels (since background_fill is (0,0,0))
        mask = (rgb > 0).any(axis=2)

        try:
            fv = extract_lettuce_features(rgb, mask, config=DEFAULT_LETTUCE_CONFIG)
            row_dict = {
                "image_id": crop_id,
                "source_image_id": source_img_id,
                "class": crop_class,
                "split": crop_split,
            }
            feat_map = fv.to_dict()
            # Add features in strict canonical order
            for fname in LETTUCE_FEATURE_NAMES:
                row_dict[fname] = feat_map[fname]

            records.append(row_dict)
        except FeatureExtractionError as fee:
            failed_crops.append({"crop_id": crop_id, "reason": str(fee)})
        except Exception as e:
            failed_crops.append(
                {"crop_id": crop_id, "reason": f"Unexpected error: {e}"}
            )

    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise RuntimeError("No features extracted for lettuce!")

    # Verify no NaN or Inf in feature columns
    for fname in LETTUCE_FEATURE_NAMES:
        assert not df_all[fname].isna().any(), f"NaN detected in feature {fname}"
        assert not np.isinf(df_all[fname]).any(), f"Inf detected in feature {fname}"

    # Partition and save CSVs
    feature_cols = (
        ["image_id", "source_image_id", "class"]
        + list(LETTUCE_FEATURE_NAMES)
        + ["split"]
    )
    counts_per_split = {}
    for split_name in ("train", "val", "test"):
        sub_df = df_all[df_all["split"] == split_name][feature_cols]
        sub_path = output_dir / f"{split_name}.csv"
        sub_df.to_csv(sub_path, index=False)
        counts_per_split[split_name] = len(sub_df)
        LOGGER.info(f"Saved {split_name}.csv with {len(sub_df)} rows")

    metadata = {
        "crop": "lettuce",
        "schema_version": DEFAULT_FEATURE_SCHEMA_VERSION,
        "feature_names": list(LETTUCE_FEATURE_NAMES),
        "total_crops": len(df_all),
        "counts_per_split": counts_per_split,
        "counts_per_class": df_all["class"].value_counts().to_dict(),
        "failed_crops_count": len(failed_crops),
        "failed_crops": failed_crops,
        "extraction_config": DEFAULT_LETTUCE_CONFIG.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def build_spinach_features(
    output_dir: Path,
    overwrite: bool = False,
    max_samples: int | None = None,
    num_workers: int = 8,
) -> dict[str, Any]:
    """Extract features for spinach leaf images using multi-threaded batch processing."""
    from concurrent.futures import ThreadPoolExecutor

    split_manifest = PROJECT_ROOT / "processed_data/splits/image_splits_spinach.csv"
    if not split_manifest.exists():
        raise FileNotFoundError(f"Spinach split manifest not found at {split_manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    metadata_path = output_dir / "metadata.json"

    if (
        train_path.exists() or val_path.exists() or test_path.exists()
    ) and not overwrite:
        LOGGER.info(
            f"Spinach features already exist in {output_dir}. Refusing to overwrite without --overwrite."
        )
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "skipped", "reason": "existing files without overwrite"}

    df_manifest = pd.read_csv(split_manifest)
    if max_samples is not None:
        df_manifest = df_manifest.head(max_samples)

    LOGGER.info(
        f"Building spinach features for {len(df_manifest)} images using {num_workers} threads..."
    )

    def process_spinach_image(
        row_tuple: tuple[int, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        _, row = row_tuple
        rel_img_path = row["image_path"]
        full_path = PROJECT_ROOT / rel_img_path
        img_id = Path(rel_img_path).stem
        img_class = row["class"]
        img_split = row["split"]

        if not full_path.exists():
            return None, {"image_id": img_id, "reason": "File missing"}

        bgr = cv2.imread(str(full_path))
        if bgr is None:
            return None, {"image_id": img_id, "reason": "Unreadable image file"}

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            rgb = cv2.resize(
                rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        h, w = rgb.shape[:2]
        r, g, b = (
            rgb[:, :, 0].astype(float),
            rgb[:, :, 1].astype(float),
            rgb[:, :, 2].astype(float),
        )
        exg = 2.0 * g - r - b
        mask = (exg > 5.0) & ((r + g + b) > 30.0)
        if not mask.any():
            return None, {
                "image_id": img_id,
                "reason": "Masking failed: no vegetation pixels detected",
            }

        try:
            fv = extract_spinach_features(rgb, mask, config=DEFAULT_SPINACH_CONFIG)
            row_dict = {
                "image_id": img_id,
                "class": img_class,
                "split": img_split,
            }
            feat_map = fv.to_dict()
            for fname in SPINACH_FEATURE_NAMES:
                row_dict[fname] = feat_map[fname]
            return row_dict, None
        except Exception as e:
            return None, {"image_id": img_id, "reason": str(e)}

    records: list[dict[str, Any]] = []
    failed_images: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_spinach_image, df_manifest.iterrows()),
                total=len(df_manifest),
                desc="Extracting spinach features",
            )
        )

    for rec, failed in results:
        if rec is not None:
            records.append(rec)
        if failed is not None:
            failed_images.append(failed)

    df_all = pd.DataFrame(records)
    for fname in SPINACH_FEATURE_NAMES:
        assert not df_all[fname].isna().any(), f"NaN in feature {fname}"
        assert not np.isinf(df_all[fname]).any(), f"Inf in feature {fname}"

    feature_cols = ["image_id", "class"] + list(SPINACH_FEATURE_NAMES) + ["split"]
    counts_per_split = {}
    for split_name in ("train", "val", "test"):
        sub_df = df_all[df_all["split"] == split_name][feature_cols]
        sub_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        counts_per_split[split_name] = len(sub_df)

    metadata = {
        "crop": "spinach",
        "schema_version": DEFAULT_FEATURE_SCHEMA_VERSION,
        "feature_names": list(SPINACH_FEATURE_NAMES),
        "total_images": len(df_all),
        "counts_per_split": counts_per_split,
        "counts_per_class": df_all["class"].value_counts().to_dict(),
        "failed_images_count": len(failed_images),
        "failed_images": failed_images,
        "extraction_config": DEFAULT_SPINACH_CONFIG.to_dict(),
        "mask_config": {
            "algorithm": "excess_green_thresholding",
            "exg_threshold": 5.0,
            "min_intensity": 30.0,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def build_tomato_features(
    output_dir: Path,
    overwrite: bool = False,
    max_samples: int | None = None,
    num_workers: int = 8,
) -> dict[str, Any]:
    """Extract features for tomato leaf images using multi-threaded batch processing."""
    from concurrent.futures import ThreadPoolExecutor

    split_manifest = PROJECT_ROOT / "processed_data/splits/image_splits_tomato.csv"
    if not split_manifest.exists():
        raise FileNotFoundError(f"Tomato split manifest not found at {split_manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    metadata_path = output_dir / "metadata.json"

    if (
        train_path.exists() or val_path.exists() or test_path.exists()
    ) and not overwrite:
        LOGGER.info(
            f"Tomato features already exist in {output_dir}. Refusing to overwrite without --overwrite."
        )
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "skipped", "reason": "existing files without overwrite"}

    df_manifest = pd.read_csv(split_manifest)
    if max_samples is not None:
        df_manifest = df_manifest.head(max_samples)

    LOGGER.info(
        f"Building tomato features for {len(df_manifest)} images using {num_workers} threads..."
    )

    def process_tomato_image(
        row_tuple: tuple[int, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        _, row = row_tuple
        rel_img_path = row["image_path"]
        full_path = PROJECT_ROOT / rel_img_path
        img_id = Path(rel_img_path).stem
        img_class = row["class"]
        img_split = row["split"]

        if not full_path.exists():
            return None, {"image_id": img_id, "reason": "File missing"}

        bgr = cv2.imread(str(full_path))
        if bgr is None:
            return None, {"image_id": img_id, "reason": "Unreadable image file"}

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            rgb = cv2.resize(
                rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        h, w = rgb.shape[:2]
        r, g, b = (
            rgb[:, :, 0].astype(float),
            rgb[:, :, 1].astype(float),
            rgb[:, :, 2].astype(float),
        )
        exg = 2.0 * g - r - b
        mask = (exg > 0.0) & ((r + g + b) > 20.0)
        if not mask.any():
            mask = np.zeros((h, w), dtype=bool)
            mask[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)] = True

        try:
            fv = extract_tomato_features(rgb, mask, config=DEFAULT_TOMATO_CONFIG)
            row_dict = {
                "image_id": img_id,
                "class": img_class,
                "split": img_split,
            }
            feat_map = fv.to_dict()
            for fname in TOMATO_FEATURE_NAMES:
                row_dict[fname] = feat_map[fname]
            return row_dict, None
        except Exception as e:
            return None, {"image_id": img_id, "reason": str(e)}

    records: list[dict[str, Any]] = []
    failed_images: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_tomato_image, df_manifest.iterrows()),
                total=len(df_manifest),
                desc="Extracting tomato features",
            )
        )

    for rec, fail in results:
        if rec is not None:
            records.append(rec)
        if fail is not None:
            failed_images.append(fail)

    df_all = pd.DataFrame(records)
    for fname in TOMATO_FEATURE_NAMES:
        assert not df_all[fname].isna().any(), f"NaN in feature {fname}"
        assert not np.isinf(df_all[fname]).any(), f"Inf in feature {fname}"

    feature_cols = ["image_id", "class"] + list(TOMATO_FEATURE_NAMES) + ["split"]
    counts_per_split = {}
    for split_name in ("train", "val", "test"):
        sub_df = df_all[df_all["split"] == split_name][feature_cols]
        sub_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        counts_per_split[split_name] = len(sub_df)

    metadata = {
        "crop": "tomato",
        "schema_version": DEFAULT_FEATURE_SCHEMA_VERSION,
        "feature_names": list(TOMATO_FEATURE_NAMES),
        "total_images": len(df_all),
        "counts_per_split": counts_per_split,
        "counts_per_class": df_all["class"].value_counts().to_dict(),
        "failed_images_count": len(failed_images),
        "failed_images": failed_images,
        "extraction_config": DEFAULT_TOMATO_CONFIG.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build feature datasets from split manifests (T5-013)"
    )
    parser.add_argument(
        "crop",
        choices=["lettuce", "spinach", "tomato", "all"],
        help="Crop dataset to extract features for",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite existing feature datasets",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for rapid debugging",
    )
    args = parser.parse_args()

    crops = ["lettuce", "spinach", "tomato"] if args.crop == "all" else [args.crop]

    for c in crops:
        out_dir = PROJECT_ROOT / f"processed_data/features/{c}"
        LOGGER.info(f"Processing features for crop: {c}")
        if c == "lettuce":
            summary = build_lettuce_features(
                out_dir, overwrite=args.overwrite, max_samples=args.max_samples
            )
        elif c == "spinach":
            summary = build_spinach_features(
                out_dir, overwrite=args.overwrite, max_samples=args.max_samples
            )
        elif c == "tomato":
            summary = build_tomato_features(
                out_dir, overwrite=args.overwrite, max_samples=args.max_samples
            )
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
