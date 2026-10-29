#!/usr/bin/env python3
"""Verification script for T8-009: Data and leakage tests.

Overview §27.3 Data and leakage tests:
1. Class counts before/after preprocessing.
2. No overlap of source image identities across image splits.
3. Sibling lettuce crops share a split.
4. Sensor group/time separation holds.
5. Prohibited leakage fields absent from sensor features.
6. Target column not present among features.
7. Split hashes unchanged (split_hashes.json).
8. No NaN/Inf values reach models without documented policy.
9. Generated feature column order matches artifact metadata.

Acceptance criteria:
1. Each Overview §27.3 bullet maps to at least one test.
2. Tests skip with a clear message if data is absent locally.
3. Tests run in CI against tiny fixtures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.split_guard import DEFAULT_HASH_FILE, verify_split_hashes
from ml.sensor_models.schemas import CROP_SENSOR_SCHEMAS, find_blacklisted_columns


def run_verification() -> bool:
    print("================================================================================")
    print("VERIFICATION FOR T8-009: DATA AND LEAKAGE TESTS")
    print("================================================================================\n")

    all_passed = True
    crops = ["lettuce", "spinach", "tomato"]

    # 1. Class counts consistency
    print("[1/9] Checking Class Counts Consistency (Manifests vs Splits)...")
    try:
        for crop in crops:
            manifest_path = PROJECT_ROOT / "processed_data" / "sensor" / crop / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                counts = manifest.get("row_counts", {})
                train_df = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / crop / "train.csv")
                val_df = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / crop / "val.csv")
                test_df = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / crop / "test.csv")

                if "train" in counts:
                    assert len(train_df) == counts["train"]
                    assert len(val_df) == counts["val"]
                    assert len(test_df) == counts["test"]
                elif "train_balanced" in counts:
                    assert len(train_df) == counts["train_balanced"]
                    assert len(val_df) == counts["val_real"]
                    assert len(test_df) == counts["test_real"]
                print(f"  OK: {crop} sensor class row counts match manifest.")
    except Exception as e:
        print(f"  FAIL: Class counts check failed: {e}")
        all_passed = False

    # 2. No source image overlap
    print("\n[2/9] Checking No Source Image Overlap Across Image Splits...")
    try:
        for crop in crops:
            split_csv = PROJECT_ROOT / "processed_data" / "splits" / f"image_splits_{crop}.csv"
            if split_csv.is_file():
                df = pd.read_csv(split_csv)
                train_imgs = set(df[df["split"] == "train"]["image_path"])
                val_imgs = set(df[df["split"] == "val"]["image_path"])
                test_imgs = set(df[df["split"] == "test"]["image_path"])

                assert train_imgs.isdisjoint(val_imgs)
                assert train_imgs.isdisjoint(test_imgs)
                assert val_imgs.isdisjoint(test_imgs)
                print(f"  OK: {crop} image splits are mutually disjoint across train/val/test.")
    except Exception as e:
        print(f"  FAIL: Source image overlap check failed: {e}")
        all_passed = False

    # 3. Sibling lettuce crops share split
    print("\n[3/9] Checking Sibling Lettuce Crops Share Split...")
    try:
        source_map = PROJECT_ROOT / "processed_data" / "lettuce_leaf_crops" / "source_map.csv"
        if source_map.is_file():
            df = pd.read_csv(source_map)
            splits_per_source = df.groupby("source_image_id")["split"].nunique()
            leaky = splits_per_source[splits_per_source > 1]
            assert len(leaky) == 0, f"Found {len(leaky)} source images in multiple splits"
            print("  OK: All sibling lettuce leaf crops share exactly one split per source image.")
    except Exception as e:
        print(f"  FAIL: Sibling crops check failed: {e}")
        all_passed = False

    # 4. Sensor group/time separation
    print("\n[4/9] Checking Sensor Group/Time Separation...")
    try:
        # Lettuce day-group
        l_tr = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "lettuce" / "train.csv")
        l_va = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "lettuce" / "val.csv")
        l_te = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "lettuce" / "test.csv")
        k_tr = set(zip(l_tr["experiment"], l_tr["date"], strict=False))
        k_va = set(zip(l_va["experiment"], l_va["date"], strict=False))
        k_te = set(zip(l_te["experiment"], l_te["date"], strict=False))
        assert k_tr.isdisjoint(k_va) and k_tr.isdisjoint(k_te) and k_va.isdisjoint(k_te)
        print("  OK: Lettuce day-grouped holdouts strictly separated.")

        # Spinach 6-hour block
        s_tr = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "spinach" / "train.csv")
        s_va = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "spinach" / "val.csv")
        s_te = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "spinach" / "test.csv")

        def sp_blocks(df: pd.DataFrame) -> set[str]:
            ts = pd.to_datetime(df["source_timestamp_EDT"])
            return set(df["bench"].astype(str) + "|" + ts.dt.strftime("%Y-%m-%d") + "|" + (ts.dt.hour // 6).astype(str))

        b_tr, b_va, b_te = sp_blocks(s_tr), sp_blocks(s_va), sp_blocks(s_te)
        assert b_tr.isdisjoint(b_va) and b_tr.isdisjoint(b_te) and b_va.isdisjoint(b_te)
        print("  OK: Spinach 6-hour block holdouts strictly separated.")

        # Tomato calendar month
        t_tr = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "tomato" / "train.csv")
        t_va = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "tomato" / "val.csv")
        t_te = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "tomato" / "test.csv")
        m_tr = set(t_tr["source_timestamp"].str[:7])
        m_va = set(t_va["source_timestamp"].str[:7])
        m_te = set(t_te["source_timestamp"].str[:7])
        assert m_tr.isdisjoint(m_va) and m_tr.isdisjoint(m_te) and m_va.isdisjoint(m_te)
        print("  OK: Tomato calendar month holdouts strictly separated.")
    except Exception as e:
        print(f"  FAIL: Sensor separation check failed: {e}")
        all_passed = False

    # 5. Prohibited leakage fields absent
    print("\n[5/9] Checking Prohibited Leakage Fields Absent from Sensor Features...")
    try:
        for crop in crops:
            schema = CROP_SENSOR_SCHEMAS[crop]
            violating = find_blacklisted_columns(schema.feature_names, crop)
            assert len(violating) == 0, f"{crop} has blacklisted features: {violating}"
            print(f"  OK: {crop} schema features contain zero blacklisted leakage fields.")
    except Exception as e:
        print(f"  FAIL: Leakage blacklist check failed: {e}")
        all_passed = False

    # 6. Target column not in features
    print("\n[6/9] Checking Target Column Absent from Features...")
    try:
        for crop in crops:
            schema = CROP_SENSOR_SCHEMAS[crop]
            for target_col in ["label", "label_id", "label_next", "class", "target"]:
                assert target_col not in schema.feature_names
            print(f"  OK: {crop} target columns strictly absent from input schema.")
    except Exception as e:
        print(f"  FAIL: Target in features check failed: {e}")
        all_passed = False

    # 7. Split hashes intact
    print("\n[7/9] Checking Split Hash Manifest Integrity (split_hashes.json)...")
    try:
        is_valid, errors = verify_split_hashes(raise_on_failure=False)
        assert is_valid, f"Split hashes tampered or missing: {errors}"
        print("  OK: All tracked split files match split_hashes.json digests.")
    except Exception as e:
        print(f"  FAIL: Split hash integrity check failed: {e}")
        all_passed = False

    # 8. No NaN/Inf reaching models without documented policy
    print("\n[8/9] Checking No NaN/Inf Reaching Models Without Documented Policy...")
    try:
        # Check spinach & tomato sensor features
        for crop in ["spinach", "tomato"]:
            schema = CROP_SENSOR_SCHEMAS[crop]
            cols = list(schema.feature_names)
            for split in ["train.csv", "val.csv", "test.csv"]:
                df = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / crop / split)
                assert not df[cols].isna().any().any(), f"{crop} {split} contains NaNs"
                assert not np.isinf(df[cols].to_numpy(dtype=float)).any(), f"{crop} {split} contains Infs"

        # Check lettuce imputer policy
        import joblib
        lettuce_model_path = PROJECT_ROOT / "models" / "sensor" / "lettuce" / "v1.0.0" / "lettuce_sensor_decision_tree_v1.0.0.joblib"
        lettuce_pipe = joblib.load(lettuce_model_path)
        imputer = lettuce_pipe.named_steps["imputer"]
        assert imputer is not None
        for split in ["train.csv", "val.csv", "test.csv"]:
            df = pd.read_csv(PROJECT_ROOT / "processed_data" / "sensor" / "lettuce" / split)
            imputed = imputer.transform(df[list(CROP_SENSOR_SCHEMAS["lettuce"].feature_names)])
            assert not np.isnan(imputed).any()
            assert not np.isinf(imputed).any()

        # Check CV extracted features
        for crop in crops:
            meta = json.loads((PROJECT_ROOT / "processed_data" / "features" / crop / "metadata.json").read_text(encoding="utf-8"))
            cols = meta["feature_names"]
            for split in ["train.csv", "val.csv", "test.csv"]:
                df = pd.read_csv(PROJECT_ROOT / "processed_data" / "features" / crop / split)
                assert not df[cols].isna().any().any()
                assert not np.isinf(df[cols].to_numpy(dtype=float)).any()
        print("  OK: Zero undocumented NaN/Inf values reach models.")
    except Exception as e:
        print(f"  FAIL: NaN/Inf check failed: {e}")
        all_passed = False

    # 9. Feature column order matches metadata
    print("\n[9/9] Checking Feature Column Order Matches Artifact Metadata...")
    try:
        for crop in crops:
            # Sensor metadata
            meta_path = PROJECT_ROOT / "models" / "sensor" / crop / "v1.0.0" / "metadata.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                assert meta["ordered_features"] == list(CROP_SENSOR_SCHEMAS[crop].feature_names)

            # Features metadata
            feat_meta_path = PROJECT_ROOT / "processed_data" / "features" / crop / "metadata.json"
            if feat_meta_path.is_file():
                f_meta = json.loads(feat_meta_path.read_text(encoding="utf-8"))
                expected_cols = f_meta["feature_names"]
                df = pd.read_csv(PROJECT_ROOT / "processed_data" / "features" / crop / "train.csv")
                numeric_cols = [c for c in df.columns if c not in ("image_id", "source_image_id", "class", "split")]
                assert numeric_cols == expected_cols
        print("  OK: Feature column ordering exactly matches metadata.")
    except Exception as e:
        print(f"  FAIL: Feature column order check failed: {e}")
        all_passed = False

    # Run PyTest suite for T8-009
    print("\n" + "=" * 80)
    print("RUNNING PYTEST SUITE: tests/unit/test_t8_009_data_and_leakage.py")
    print("=" * 80)
    cmd = [sys.executable, "-m", "pytest", "tests/unit/test_t8_009_data_and_leakage.py", "-v"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("STATUS: VERIFICATION SUCCESSFUL - ALL ACCEPTANCE CRITERIA MET")
    else:
        print("STATUS: VERIFICATION FAILED")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)

