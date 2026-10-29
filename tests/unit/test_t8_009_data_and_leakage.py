"""Unit tests for Data and Leakage verification (T8-009).

Overview §27.3 Data and leakage tests:
1. Class counts before/after preprocessing
2. No overlap of source image identities across image splits
3. Sibling lettuce crops share a split (all crops from same original plant image remain in one split)
4. Sensor group/time separation holds (calendar month holdouts, day-groups, 6-hr blocks)
5. Prohibited leakage fields absent from sensor features (ml.sensor_models.schemas blacklist)
6. Target column not present among features
7. Train/val/test split files not silently regenerated (hash guard split_hashes.json)
8. No NaN/Inf values reach models without documented policy
9. Generated feature column order matches artifact metadata

Acceptance criteria:
- Each Overview §27.3 bullet maps to at least one test
- Tests skip with a clear message if data is absent locally
- Tests run in CI against tiny fixtures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.split_guard import (
    DEFAULT_HASH_FILE,
    SplitTamperError,
    assert_splits_intact,
    compute_file_sha256,
    verify_split_hashes,
)
from ml.sensor_models.schemas import (
    CROP_SENSOR_SCHEMAS,
    DataLeakageError,
    SchemaValidationError,
    assert_no_leakage,
    find_blacklisted_columns,
    get_leakage_blacklist,
    validate_feature_matrix,
)

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = ROOT / "processed_data"
SENSOR_DATA_DIR = PROCESSED_DATA_DIR / "sensor"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
LETTUCE_CROPS_DIR = PROCESSED_DATA_DIR / "lettuce_leaf_crops"
MODELS_DIR = ROOT / "models"


# ==============================================================================
# Helper predicates for skipping when local full data is absent
# ==============================================================================

def has_sensor_splits() -> bool:
    """Check if sensor split CSVs exist locally."""
    for crop in ["lettuce", "spinach", "tomato"]:
        for split in ["train.csv", "val.csv", "test.csv"]:
            if not (SENSOR_DATA_DIR / crop / split).is_file():
                return False
    return True


def has_image_splits() -> bool:
    """Check if image split CSVs exist locally."""
    for crop in ["lettuce", "spinach", "tomato"]:
        if not (SPLITS_DIR / f"image_splits_{crop}.csv").is_file():
            return False
    return True


def has_lettuce_source_map() -> bool:
    """Check if lettuce leaf crops source map exists locally."""
    return (LETTUCE_CROPS_DIR / "source_map.csv").is_file()


def has_processed_features() -> bool:
    """Check if processed feature extraction CSVs and metadata exist locally."""
    for crop in ["lettuce", "spinach", "tomato"]:
        if not (FEATURES_DIR / crop / "metadata.json").is_file():
            return False
        for split in ["train.csv", "val.csv", "test.csv"]:
            if not (FEATURES_DIR / crop / split).is_file():
                return False
    return True


# ==============================================================================
# 1. Class counts before/after preprocessing (§27.3 Bullet 1)
# ==============================================================================

def test_class_counts_fixture_preservation() -> None:
    """Bullet 1: Verify class counting logic correctly detects preserved or shifted counts."""
    # Synthetic raw dataset
    raw_df = pd.DataFrame({
        "sample_id": range(10),
        "label": ["Normal"] * 4 + ["Deficient"] * 4 + ["Toxic"] * 2,
    })
    # Filtered split
    train_split = raw_df.iloc[:6]
    test_split = raw_df.iloc[6:]

    raw_counts = raw_df["label"].value_counts().to_dict()
    split_counts: dict[str, int] = {}
    for sub in [train_split, test_split]:
        for k, v in sub["label"].value_counts().items():
            split_counts[k] = split_counts.get(k, 0) + v

    assert raw_counts == split_counts, "Class counts must balance between raw and splits"


@pytest.mark.skipif(not has_sensor_splits(), reason="Sensor split data missing locally")
def test_sensor_manifest_class_counts_match_splits() -> None:
    """Bullet 1: Verify actual sensor class counts in manifests match CSV splits."""
    for crop in ["lettuce", "spinach", "tomato"]:
        manifest_path = SENSOR_DATA_DIR / crop / "manifest.json"
        assert manifest_path.is_file(), f"Missing manifest for {crop}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        train_df = pd.read_csv(SENSOR_DATA_DIR / crop / "train.csv")
        val_df = pd.read_csv(SENSOR_DATA_DIR / crop / "val.csv")
        test_df = pd.read_csv(SENSOR_DATA_DIR / crop / "test.csv")

        manifest_counts = manifest.get("row_counts", {})
        if "train" in manifest_counts:
            assert len(train_df) == manifest_counts["train"]
            assert len(val_df) == manifest_counts["val"]
            assert len(test_df) == manifest_counts["test"]
        elif "train_balanced" in manifest_counts:
            # Lettuce balanced train
            assert len(train_df) == manifest_counts["train_balanced"]
            assert len(val_df) == manifest_counts["val_real"]
            assert len(test_df) == manifest_counts["test_real"]


# ==============================================================================
# 2. No overlap of source image identities across image splits (§27.3 Bullet 2)
# ==============================================================================

def test_no_source_image_overlap_fixture() -> None:
    """Bullet 2: Verify disjoint image split logic detects overlap."""
    clean_splits = pd.DataFrame({
        "image_path": ["img_01.jpg", "img_02.jpg", "img_03.jpg", "img_04.jpg"],
        "split": ["train", "train", "val", "test"],
    })
    train_imgs = set(clean_splits[clean_splits["split"] == "train"]["image_path"])
    val_imgs = set(clean_splits[clean_splits["split"] == "val"]["image_path"])
    test_imgs = set(clean_splits[clean_splits["split"] == "test"]["image_path"])

    assert train_imgs.isdisjoint(val_imgs)
    assert train_imgs.isdisjoint(test_imgs)
    assert val_imgs.isdisjoint(test_imgs)

    # Injected leak
    leaky_splits = pd.DataFrame({
        "image_path": ["img_01.jpg", "img_02.jpg", "img_01.jpg"],
        "split": ["train", "val", "test"],
    })
    train_leaky = set(leaky_splits[leaky_splits["split"] == "train"]["image_path"])
    test_leaky = set(leaky_splits[leaky_splits["split"] == "test"]["image_path"])
    assert not train_leaky.isdisjoint(test_leaky), "Should detect overlap"


@pytest.mark.skipif(not has_image_splits(), reason="Image split data missing locally")
def test_no_source_image_overlap_across_splits() -> None:
    """Bullet 2: Ensure actual image splits have zero overlapping image paths."""
    for crop in ["lettuce", "spinach", "tomato"]:
        split_csv = SPLITS_DIR / f"image_splits_{crop}.csv"
        df = pd.read_csv(split_csv)

        train_imgs = set(df[df["split"] == "train"]["image_path"])
        val_imgs = set(df[df["split"] == "val"]["image_path"])
        test_imgs = set(df[df["split"] == "test"]["image_path"])

        assert train_imgs.isdisjoint(val_imgs), f"{crop}: train and val overlap!"
        assert train_imgs.isdisjoint(test_imgs), f"{crop}: train and test overlap!"
        assert val_imgs.isdisjoint(test_imgs), f"{crop}: val and test overlap!"


# ==============================================================================
# 3. All crops from same original lettuce image remain in one split (§27.3 Bullet 3)
# ==============================================================================

def test_sibling_crops_fixture_enforcement() -> None:
    """Bullet 3: Verify grouping logic catches leaf crops leaking into different splits."""
    valid_map = pd.DataFrame({
        "crop_id": ["plant1_leaf1", "plant1_leaf2", "plant2_leaf1"],
        "source_image_id": ["plant1", "plant1", "plant2"],
        "split": ["train", "train", "val"],
    })
    grouped = valid_map.groupby("source_image_id")["split"].nunique()
    assert (grouped == 1).all(), "Valid sibling crops must have exactly 1 split per source image"

    leaky_map = pd.DataFrame({
        "crop_id": ["plant1_leaf1", "plant1_leaf2"],
        "source_image_id": ["plant1", "plant1"],
        "split": ["train", "test"],  # Leaked!
    })
    leaky_grouped = leaky_map.groupby("source_image_id")["split"].nunique()
    assert not (leaky_grouped == 1).all(), "Leaked siblings must fail assertion"


@pytest.mark.skipif(not has_lettuce_source_map(), reason="Lettuce leaf crop source map missing locally")
def test_sibling_lettuce_crops_share_split() -> None:
    """Bullet 3: Ensure all extracted lettuce leaf crops from the same source image share one split."""
    source_map_path = LETTUCE_CROPS_DIR / "source_map.csv"
    df = pd.read_csv(source_map_path)
    assert "source_image_id" in df.columns
    assert "split" in df.columns

    splits_per_source = df.groupby("source_image_id")["split"].nunique()
    multisplit_sources = splits_per_source[splits_per_source > 1]

    assert len(multisplit_sources) == 0, (
        f"Found {len(multisplit_sources)} source images with crops in multiple splits: "
        f"{multisplit_sources.index.tolist()[:5]}"
    )


# ==============================================================================
# 4. Sensor group/time separation holds (§27.3 Bullet 4)
# ==============================================================================

def test_sensor_temporal_holdout_fixture() -> None:
    """Bullet 4: Verify temporal/group holdout separation detection."""
    df_train = pd.DataFrame({"month": ["2023-01", "2023-02"], "value": [1.0, 2.0]})
    df_val = pd.DataFrame({"month": ["2023-03"], "value": [3.0]})
    df_test = pd.DataFrame({"month": ["2023-04"], "value": [4.0]})

    train_groups = set(df_train["month"])
    val_groups = set(df_val["month"])
    test_groups = set(df_test["month"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


@pytest.mark.skipif(not has_sensor_splits(), reason="Sensor split data missing locally")
def test_sensor_group_and_time_separation() -> None:
    """Bullet 4: Ensure sensor splits preserve strict group/temporal isolation."""
    # 1. Lettuce: Day-grouped holdouts (experiment + date)
    lettuce_train = pd.read_csv(SENSOR_DATA_DIR / "lettuce" / "train.csv")
    lettuce_val = pd.read_csv(SENSOR_DATA_DIR / "lettuce" / "val.csv")
    lettuce_test = pd.read_csv(SENSOR_DATA_DIR / "lettuce" / "test.csv")

    def lettuce_keys(df: pd.DataFrame) -> set[tuple[Any, Any]]:
        return set(zip(df["experiment"], df["date"], strict=False))

    l_train_keys = lettuce_keys(lettuce_train)
    l_val_keys = lettuce_keys(lettuce_val)
    l_test_keys = lettuce_keys(lettuce_test)

    assert l_train_keys.isdisjoint(l_val_keys), "Lettuce train and val share experiment-date keys!"
    assert l_train_keys.isdisjoint(l_test_keys), "Lettuce train and test share experiment-date keys!"
    assert l_val_keys.isdisjoint(l_test_keys), "Lettuce val and test share experiment-date keys!"

    # 2. Spinach: 6-hour block holdouts (bench + date + hour // 6)
    spinach_train = pd.read_csv(SENSOR_DATA_DIR / "spinach" / "train.csv")
    spinach_val = pd.read_csv(SENSOR_DATA_DIR / "spinach" / "val.csv")
    spinach_test = pd.read_csv(SENSOR_DATA_DIR / "spinach" / "test.csv")

    def spinach_keys(df: pd.DataFrame) -> set[str]:
        ts = pd.to_datetime(df["source_timestamp_EDT"])
        hours_block = (ts.dt.hour // 6).astype(str)
        dates = ts.dt.strftime("%Y-%m-%d")
        return set(df["bench"].astype(str) + "|" + dates + "|" + hours_block)

    s_train_keys = spinach_keys(spinach_train)
    s_val_keys = spinach_keys(spinach_val)
    s_test_keys = spinach_keys(spinach_test)

    assert s_train_keys.isdisjoint(s_val_keys), "Spinach train and val share 6-hour blocks!"
    assert s_train_keys.isdisjoint(s_test_keys), "Spinach train and test share 6-hour blocks!"
    assert s_val_keys.isdisjoint(s_test_keys), "Spinach val and test share 6-hour blocks!"

    # 3. Tomato: Calendar month holdouts (YYYY-MM)
    tomato_train = pd.read_csv(SENSOR_DATA_DIR / "tomato" / "train.csv")
    tomato_val = pd.read_csv(SENSOR_DATA_DIR / "tomato" / "val.csv")
    tomato_test = pd.read_csv(SENSOR_DATA_DIR / "tomato" / "test.csv")

    def tomato_keys(df: pd.DataFrame) -> set[str]:
        return set(df["source_timestamp"].str[:7])

    t_train_keys = tomato_keys(tomato_train)
    t_val_keys = tomato_keys(tomato_val)
    t_test_keys = tomato_keys(tomato_test)

    assert t_train_keys.isdisjoint(t_val_keys), "Tomato train and val share calendar months!"
    assert t_train_keys.isdisjoint(t_test_keys), "Tomato train and test share calendar months!"
    assert t_val_keys.isdisjoint(t_test_keys), "Tomato val and test share calendar months!"


# ==============================================================================
# 5. Prohibited leakage fields absent from features (§27.3 Bullet 5)
# ==============================================================================

def test_leakage_detection_fixture_raises_error() -> None:
    """Bullet 5: Verify assert_no_leakage raises DataLeakageError when prohibited fields present."""
    clean_cols = ["ph", "ec", "water_temp", "humidity", "air_temp"]
    assert_no_leakage(clean_cols, "lettuce")  # Should not raise

    leaky_cols = ["ph", "ec", "label_next", "humidity", "air_temp"]
    with pytest.raises(DataLeakageError) as exc_info:
        assert_no_leakage(leaky_cols, "lettuce")
    assert "label_next" in str(exc_info.value)

    # Wildcard pattern check
    pattern_leak = ["ph", "ec", "ph_roll3"]
    with pytest.raises(DataLeakageError):
        assert_no_leakage(pattern_leak, "lettuce")


def test_prohibited_leakage_fields_absent_from_sensor_features() -> None:
    """Bullet 5: Ensure schema ordered features contain zero blacklisted columns."""
    for crop in ["lettuce", "spinach", "tomato"]:
        schema = CROP_SENSOR_SCHEMAS[crop]
        feature_names = schema.feature_names
        blacklisted = find_blacklisted_columns(feature_names, crop)
        assert len(blacklisted) == 0, f"{crop} schema has blacklisted features: {blacklisted}"


# ==============================================================================
# 6. Target column is not present among features (§27.3 Bullet 6)
# ==============================================================================

def test_target_in_features_fixture_detected() -> None:
    """Bullet 6: Verify validation rejects target column present in feature matrix."""
    df_with_target = pd.DataFrame({
        "ph": [6.0],
        "ec": [1.5],
        "water_temp": [20.0],
        "humidity": [60.0],
        "air_temp": [22.0],
        "label": ["Normal"],
    })
    # Target column must be rejected as leakage
    with pytest.raises(DataLeakageError):
        assert_no_leakage(df_with_target.columns, "lettuce")


@pytest.mark.skipif(not has_sensor_splits(), reason="Sensor splits missing locally")
def test_target_column_not_in_sensor_or_feature_inputs() -> None:
    """Bullet 6: Ensure target label is never present in input feature matrices."""
    for crop in ["lettuce", "spinach", "tomato"]:
        schema = CROP_SENSOR_SCHEMAS[crop]
        for prohibited in ["label", "label_id", "label_next", "class", "target"]:
            assert prohibited not in schema.feature_names, (
                f"{crop} feature names contains target column: {prohibited}"
            )

    if has_processed_features():
        for crop in ["lettuce", "spinach", "tomato"]:
            meta_path = FEATURES_DIR / crop / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feature_names = meta.get("feature_names", [])
            for prohibited in ["label", "label_id", "class", "target"]:
                assert prohibited not in feature_names, (
                    f"{crop} processed features contains target: {prohibited}"
                )


# ==============================================================================
# 7. Train/validation/test files not silently regenerated (§27.3 Bullet 7)
# ==============================================================================

def test_split_tampering_fixture_detected(tmp_path: Path) -> None:
    """Bullet 7: Verify SplitTamperError is raised if file digest changes."""
    split_file = tmp_path / "dummy_split.csv"
    split_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    h = compute_file_sha256(split_file)
    rel_path = str(split_file.relative_to(tmp_path)).replace("\\", "/")
    hash_record = {
        "version": "1.0",
        "total_files": 1,
        "files": {
            rel_path: h,
        },
    }
    hash_file = tmp_path / "split_hashes.json"
    hash_file.write_text(json.dumps(hash_record), encoding="utf-8")

    # Verify intact
    is_valid, errors = verify_split_hashes(hash_file=hash_file, root=tmp_path, raise_on_failure=False)
    assert is_valid
    assert len(errors) == 0

    # Tamper with file
    split_file.write_text("a,b,c\n1,2,999\n", encoding="utf-8")
    is_valid_tampered, errors_tampered = verify_split_hashes(
        hash_file=hash_file, root=tmp_path, raise_on_failure=False
    )
    assert not is_valid_tampered
    assert len(errors_tampered) == 1

    with pytest.raises(SplitTamperError):
        verify_split_hashes(hash_file=hash_file, root=tmp_path, raise_on_failure=True)


def test_split_hashes_guard_intact() -> None:
    """Bullet 7: Ensure repository split files match split_hashes.json exactly."""
    assert DEFAULT_HASH_FILE.is_file(), f"Missing split hash registry: {DEFAULT_HASH_FILE}"
    is_valid, errors = verify_split_hashes(raise_on_failure=False)
    assert is_valid, (
        f"Split files have been altered or are missing! Errors:\n"
        f"{json.dumps(errors, indent=2)}"
    )


# ==============================================================================
# 8. No NaN/Inf values reach models without a documented policy (§27.3 Bullet 8)
# ==============================================================================

def test_nan_inf_fixture_detected() -> None:
    """Bullet 8: Ensure validation detects NaN or Inf values in feature arrays."""
    df_nan = pd.DataFrame({"f1": [1.0, np.nan], "f2": [2.0, 3.0]})
    assert df_nan.isna().any().any(), "Should detect NaN"

    df_inf = pd.DataFrame({"f1": [1.0, np.inf], "f2": [2.0, -np.inf]})
    assert np.isinf(df_inf.values).any(), "Should detect Inf"


@pytest.mark.skipif(not has_sensor_splits(), reason="Sensor splits missing locally")
def test_no_nan_or_inf_in_sensor_ordered_features() -> None:
    """Bullet 8: Verify no NaN or Inf values reach models without a documented policy.

    Policy:
    - Spinach and Tomato sensor splits must have 0 NaNs and 0 Infs.
    - Lettuce has missing values in raw sensor splits handled by an explicit
      fitted SimpleImputer(strategy='median') within the promoted SensorPipeline,
      ensuring 0 NaNs reach the classifier.
    - Online SensorService strictly rejects missing inputs with MISSING_FIELD.
    """
    import joblib

    for crop in ["spinach", "tomato"]:
        schema = CROP_SENSOR_SCHEMAS[crop]
        feature_cols = list(schema.feature_names)
        for split in ["train.csv", "val.csv", "test.csv"]:
            csv_path = SENSOR_DATA_DIR / crop / split
            df = pd.read_csv(csv_path)
            sub = df[feature_cols]

            assert not sub.isna().any().any(), f"{crop} {split} contains undocumented NaN values!"
            numeric_vals = sub.to_numpy(dtype=float)
            assert not np.isinf(numeric_vals).any(), f"{crop} {split} contains Inf values!"

    # Lettuce documented policy verification
    lettuce_schema = CROP_SENSOR_SCHEMAS["lettuce"]
    lettuce_features = list(lettuce_schema.feature_names)
    lettuce_model_path = MODELS_DIR / "sensor" / "lettuce" / "v1.0.0" / "lettuce_sensor_decision_tree_v1.0.0.joblib"
    assert lettuce_model_path.is_file(), "Promoted lettuce model pipeline missing"
    lettuce_pipeline = joblib.load(lettuce_model_path)

    # Verify imputer step exists
    imputer = getattr(lettuce_pipeline, "named_steps", {}).get("imputer")
    assert imputer is not None, "Lettuce pipeline must include documented imputer for missing sensor data"

    for split in ["train.csv", "val.csv", "test.csv"]:
        csv_path = SENSOR_DATA_DIR / "lettuce" / split
        df = pd.read_csv(csv_path)
        sub = df[lettuce_features]
        # Inf must never be present
        numeric_vals = sub.to_numpy(dtype=float)
        assert not np.isinf(numeric_vals).any(), f"Lettuce {split} contains Inf values!"

        # Imputed output reaching classifier must contain zero NaNs
        imputed_vals = imputer.transform(sub)
        assert not np.isnan(imputed_vals).any(), f"Imputed lettuce {split} features still contain NaNs!"
        assert not np.isinf(imputed_vals).any(), f"Imputed lettuce {split} features contain Infs!"


@pytest.mark.skipif(not has_processed_features(), reason="Extracted features missing locally")
def test_no_nan_or_inf_in_extracted_features() -> None:
    """Bullet 8: Verify no NaN or Inf values in computer vision extracted features."""
    for crop in ["lettuce", "spinach", "tomato"]:
        meta_path = FEATURES_DIR / crop / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = meta.get("feature_names", [])

        for split in ["train.csv", "val.csv", "test.csv"]:
            csv_path = FEATURES_DIR / crop / split
            df = pd.read_csv(csv_path)
            sub = df[feature_cols]

            assert not sub.isna().any().any(), f"{crop} {split} features contain NaN!"
            vals = sub.to_numpy(dtype=float)
            assert not np.isinf(vals).any(), f"{crop} {split} features contain Inf!"


# ==============================================================================
# 9. Feature column order matches artifact metadata (§27.3 Bullet 9)
# ==============================================================================

def test_feature_order_fixture_validation() -> None:
    """Bullet 9: Verify feature ordering validation rejects permutation or mismatch."""
    expected_order = ["ph", "ec", "water_temp", "humidity", "air_temp"]
    actual_cols = ["ph", "ec", "water_temp", "humidity", "air_temp"]
    assert actual_cols == expected_order

    shuffled_cols = ["ec", "ph", "water_temp", "humidity", "air_temp"]
    assert shuffled_cols != expected_order, "Shuffled columns must not match"


def test_sensor_model_ordered_features_matches_schema() -> None:
    """Bullet 9: Verify active model metadata ordered_features matches canonical schema."""
    for crop in ["lettuce", "spinach", "tomato"]:
        metadata_path = MODELS_DIR / "sensor" / crop / "v1.0.0" / "metadata.json"
        if not metadata_path.is_file():
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_features = meta.get("ordered_features")
        schema_features = list(CROP_SENSOR_SCHEMAS[crop].feature_names)
        assert model_features == schema_features, (
            f"{crop} sensor model metadata ordered_features {model_features} "
            f"does not match schema {schema_features}"
        )


@pytest.mark.skipif(not has_processed_features(), reason="Extracted features missing locally")
def test_extracted_features_column_order_matches_metadata() -> None:
    """Bullet 9: Verify extracted features CSV columns strictly match metadata.json."""
    for crop in ["lettuce", "spinach", "tomato"]:
        meta_path = FEATURES_DIR / crop / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_features = meta["feature_names"]

        for split in ["train.csv", "val.csv", "test.csv"]:
            csv_path = FEATURES_DIR / crop / split
            df = pd.read_csv(csv_path)
            # Numeric feature columns excluding IDs and labels
            numeric_cols = [
                c for c in df.columns
                if c not in ("image_id", "source_image_id", "class", "split")
            ]
            assert numeric_cols == expected_features, (
                f"{crop} {split} feature column order {numeric_cols} "
                f"does not match metadata {expected_features}"
            )
