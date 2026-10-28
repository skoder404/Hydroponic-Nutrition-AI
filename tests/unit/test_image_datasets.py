"""Unit tests for PyTorch datasets and training-only augmentation (T5-018).

Acceptance Criteria:
- [x] Validation/test transforms contain no random operations
- [x] Normalisation constants are saved in metadata
- [x] Dataset refuses to mix leaf sources from different crops
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from ml.image_models.data import (
    CropLeafDataset,
    build_transforms,
    concat_crop_datasets,
    create_dataloaders,
    get_normalisation_metadata,
    has_random_transforms,
)


def test_criterion_1_val_test_transforms_contain_no_random_operations():
    """Acceptance Criterion 1: Validation and test transforms contain no random operations."""
    train_transform = build_transforms(split="train")
    val_transform = build_transforms(split="val")
    test_transform = build_transforms(split="test")

    # Training transform MUST contain random augmentation operations
    assert has_random_transforms(train_transform) is True, (
        "Expected training transform to include data augmentations"
    )

    # Validation and test transforms MUST be strictly deterministic (no random operations)
    assert has_random_transforms(val_transform) is False, (
        "Validation transform must contain no random operations"
    )
    assert has_random_transforms(test_transform) is False, (
        "Test transform must contain no random operations"
    )

    # Inspect transform class names directly to confirm
    val_classes = [type(t).__name__ for t in val_transform.transforms]
    test_classes = [type(t).__name__ for t in test_transform.transforms]

    for cls_name in val_classes:
        assert not cls_name.startswith("Random"), (
            f"Random operation '{cls_name}' found in validation pipeline"
        )
        assert cls_name not in ("ColorJitter", "AutoAugment", "RandAugment"), (
            f"Augmentation '{cls_name}' in val pipeline"
        )

    for cls_name in test_classes:
        assert not cls_name.startswith("Random"), (
            f"Random operation '{cls_name}' found in test pipeline"
        )
        assert cls_name not in ("ColorJitter", "AutoAugment", "RandAugment"), (
            f"Augmentation '{cls_name}' in test pipeline"
        )


def test_criterion_2_normalisation_constants_saved_in_metadata():
    """Acceptance Criterion 2: Normalisation constants are saved in metadata."""
    custom_mean = (0.5, 0.5, 0.5)
    custom_std = (0.25, 0.25, 0.25)
    custom_size = (128, 128)

    meta = get_normalisation_metadata(
        image_size=custom_size, mean=custom_mean, std=custom_std
    )
    assert meta["mean"] == [0.5, 0.5, 0.5]
    assert meta["std"] == [0.25, 0.25, 0.25]
    assert meta["image_size"] == [128, 128]

    # Check on dataset instance
    dataset = CropLeafDataset(
        crop="lettuce",
        split="train",
        image_size=custom_size,
        mean=custom_mean,
        std=custom_std,
    )
    assert "mean" in dataset.metadata
    assert "std" in dataset.metadata
    assert "image_size" in dataset.metadata
    assert dataset.metadata["mean"] == [0.5, 0.5, 0.5]
    assert dataset.metadata["std"] == [0.25, 0.25, 0.25]
    assert dataset.metadata["image_size"] == [128, 128]
    assert dataset.metadata["crop"] == "lettuce"
    assert dataset.metadata["split"] == "train"


def test_criterion_3_refuses_to_mix_leaf_sources_from_different_crops(tmp_path: Path):
    """Acceptance Criterion 3: Dataset refuses to mix leaf sources from different crops."""
    # 1. Reject manifest containing multiple crops
    mixed_csv = tmp_path / "mixed_manifest.csv"
    df_mixed = pd.DataFrame(
        {
            "image_path": ["image_data/spinach/1.jpg", "image_data/tomato/2.jpg"],
            "crop": ["spinach", "tomato"],
            "class": ["Spinach_Healthy", "Tomato_Healthy"],
            "split": ["train", "train"],
        }
    )
    df_mixed.to_csv(mixed_csv, index=False)

    with pytest.raises(
        ValueError, match="Refusing to mix leaf sources from different crops"
    ):
        CropLeafDataset(crop="spinach", manifest_path=mixed_csv)

    # 2. Reject manifest with mismatched single crop
    tomato_csv = tmp_path / "tomato_manifest.csv"
    df_tomato = pd.DataFrame(
        {
            "image_path": ["image_data/tomato/1.jpg"],
            "crop": ["tomato"],
            "class": ["Tomato_Healthy"],
            "split": ["train"],
        }
    )
    df_tomato.to_csv(tomato_csv, index=False)

    with pytest.raises(
        ValueError, match="Refusing to mix leaf sources from different crops"
    ):
        CropLeafDataset(crop="lettuce", manifest_path=tomato_csv)

    # 3. Reject concat_crop_datasets with different crops
    ds_lettuce = CropLeafDataset(crop="lettuce", split="train")
    ds_spinach = CropLeafDataset(crop="spinach", split="train")

    with pytest.raises(
        ValueError, match="Refusing to mix leaf sources from different crops"
    ):
        concat_crop_datasets(ds_lettuce, ds_spinach)


def test_dataset_item_retrieval_and_shapes():
    """Verify __getitem__ returns torch.Tensor with correct (3, H, W) shape and integer label."""
    dataset = CropLeafDataset(crop="lettuce", split="val", image_size=(160, 160))
    assert len(dataset) > 0

    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 160, 160)
    assert isinstance(label, int)
    assert 0 <= label < len(dataset.classes)


def test_dataloaders_creation_and_reproducibility():
    """Verify create_dataloaders generates functional DataLoaders with configured batch_size and seed."""
    train_loader, val_loader, test_loader = create_dataloaders(
        crop="spinach",
        batch_size=8,
        num_workers=0,
        seed=123,
        image_size=(64, 64),
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    # Fetch one batch from train_loader
    images, labels = next(iter(train_loader))
    assert images.shape == (8, 3, 64, 64)
    assert labels.shape == (8,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
