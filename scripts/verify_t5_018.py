#!/usr/bin/env python3
"""Verification script for T5-018: Implement PyTorch datasets and training-only augmentation.

Acceptance Criteria:
1. Validation/test transforms contain no random operations
2. Normalisation constants are saved in metadata
3. Dataset refuses to mix leaf sources from different crops

Usage:
    python scripts/verify_t5_018.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from ml.image_models.data import (  # noqa: E402
    CropLeafDataset,
    build_transforms,
    concat_crop_datasets,
    get_normalisation_metadata,
    has_random_transforms,
)


def verify_criterion_1_deterministic_eval_transforms() -> tuple[bool, str]:
    """Criterion 1: Validation/test transforms contain no random operations."""
    train_tf = build_transforms(split="train")
    val_tf = build_transforms(split="val")
    test_tf = build_transforms(split="test")

    if not has_random_transforms(train_tf):
        return (
            False,
            "FAIL: Training transform unexpectedly lacks stochastic augmentations.",
        )

    if has_random_transforms(val_tf):
        return (
            False,
            "FAIL: Validation transform contains stochastic/random operations.",
        )

    if has_random_transforms(test_tf):
        return False, "FAIL: Test transform contains stochastic/random operations."

    return (
        True,
        "PASS: Validation and test transforms contain no random operations; augmentation is training-only.",
    )


def verify_criterion_2_normalisation_in_metadata() -> tuple[bool, str]:
    """Criterion 2: Normalisation constants are saved in metadata."""
    custom_size = (224, 224)
    custom_mean = (0.485, 0.456, 0.406)
    custom_std = (0.229, 0.224, 0.225)

    meta = get_normalisation_metadata(custom_size, custom_mean, custom_std)
    if "mean" not in meta or "std" not in meta or "image_size" not in meta:
        return False, f"FAIL: Normalisation metadata incomplete: {meta}"

    dataset = CropLeafDataset(
        crop="lettuce",
        split="train",
        image_size=custom_size,
        mean=custom_mean,
        std=custom_std,
    )

    ds_meta = dataset.metadata
    if ds_meta.get("mean") != [0.485, 0.456, 0.406] or ds_meta.get("std") != [
        0.229,
        0.224,
        0.225,
    ]:
        return (
            False,
            f"FAIL: Dataset metadata missing or mismatching normalisation constants: {ds_meta}",
        )

    return (
        True,
        f"PASS: Normalisation constants (mean={ds_meta['mean']}, std={ds_meta['std']}, image_size={ds_meta['image_size']}) are saved in metadata.",
    )


def verify_criterion_3_refuses_to_mix_crops() -> tuple[bool, str]:
    """Criterion 3: Dataset refuses to mix leaf sources from different crops."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Manifest containing mixed crops
        mixed_csv = tmp_path / "mixed.csv"
        df_mixed = pd.DataFrame(
            {
                "image_path": ["image_data/spinach/1.jpg", "image_data/lettuce/2.jpg"],
                "crop": ["spinach", "lettuce"],
                "class": ["Spinach_Healthy", "Healthier"],
                "split": ["train", "train"],
            }
        )
        df_mixed.to_csv(mixed_csv, index=False)

        try:
            CropLeafDataset(crop="spinach", manifest_path=mixed_csv)
            return False, "FAIL: CropLeafDataset failed to reject mixed crop manifest."
        except ValueError as exc:
            if "Refusing to mix leaf sources from different crops" not in str(exc):
                return False, f"FAIL: Unexpected error message: {exc}"

        # 2. Concat datasets of different crops
        ds_lettuce = CropLeafDataset(crop="lettuce", split="train")
        ds_tomato = CropLeafDataset(crop="tomato", split="train")

        try:
            concat_crop_datasets(ds_lettuce, ds_tomato)
            return (
                False,
                "FAIL: concat_crop_datasets failed to reject combining different crops.",
            )
        except ValueError as exc:
            if "Refusing to mix leaf sources from different crops" not in str(exc):
                return False, f"FAIL: Unexpected error message: {exc}"

    return (
        True,
        "PASS: Dataset strictly refuses to mix leaf sources from different crops in manifests and concats.",
    )


def main() -> int:
    """Run all verification checks for T5-018."""
    print("=" * 70)
    print("Verifying T5-018: Implement PyTorch datasets and training-only augmentation")
    print("=" * 70)

    checks = [
        (
            "Criterion 1: Validation/test transforms contain no random operations",
            verify_criterion_1_deterministic_eval_transforms,
        ),
        (
            "Criterion 2: Normalisation constants are saved in metadata",
            verify_criterion_2_normalisation_in_metadata,
        ),
        (
            "Criterion 3: Dataset refuses to mix leaf sources from different crops",
            verify_criterion_3_refuses_to_mix_crops,
        ),
    ]

    all_passed = True
    for title, check_func in checks:
        print(f"\nChecking {title}...")
        passed, msg = check_func()
        print(f"  {msg}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY (T5-018).")
        print("=" * 70)
        return 0
    else:
        print("VERIFICATION FAILED FOR ONE OR MORE CRITERIA.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
