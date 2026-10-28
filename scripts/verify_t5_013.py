#!/usr/bin/env python3
"""Verification script for T5-013: Build feature datasets from split manifests.

Acceptance Criteria:
1. Row counts per split match the manifest (or the difference is explained)
2. No NaN/Inf rows are written
3. Re-running without the flag does not overwrite
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


def run_verification() -> bool:
    print("[1/3] Verifying feature dataset splits and row count reconciliation...")
    features_dir = PROJECT_ROOT / "processed_data/features/lettuce"
    source_map_path = PROJECT_ROOT / "processed_data/lettuce_leaf_crops/source_map.csv"
    metadata_path = features_dir / "metadata.json"

    if not features_dir.exists() or not metadata_path.exists():
        print(f"FAILED: Feature dataset or metadata not found at {features_dir}")
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df_source = pd.read_csv(source_map_path)
    expected_split_counts = df_source["split"].value_counts().to_dict()

    train_df = pd.read_csv(features_dir / "train.csv")
    val_df = pd.read_csv(features_dir / "val.csv")
    test_df = pd.read_csv(features_dir / "test.csv")

    assert len(train_df) == expected_split_counts.get("train", 0)
    assert len(val_df) == expected_split_counts.get("val", 0)
    assert len(test_df) == expected_split_counts.get("test", 0)
    assert len(train_df) + len(val_df) + len(test_df) == len(df_source)

    print(f"  Split counts strictly reconciled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}.")
    print("  AC 1 PASS: Row counts match leaf crops manifest across all splits.")

    print("[2/3] Verifying no NaN / Inf values across feature columns...")
    canonical_features = meta["feature_names"]
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for fname in canonical_features:
            assert fname in df.columns, f"Missing feature {fname} in {split_name}.csv"
            vals = df[fname].to_numpy()
            assert not np.isnan(vals).any(), f"NaN detected in {split_name}.csv column {fname}"
            assert not np.isinf(vals).any(), f"Inf detected in {split_name}.csv column {fname}"

    print("  AC 2 PASS: All feature columns are strictly finite numbers (zero NaN / Inf).")

    print("[3/3] Verifying re-running without --overwrite does not overwrite...")
    mod_time_before = (features_dir / "train.csv").stat().st_mtime
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/build_features.py"), "lettuce"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Refusing to overwrite without --overwrite" in proc.stdout or "Refusing to overwrite" in proc.stderr
    mod_time_after = (features_dir / "train.csv").stat().st_mtime
    assert mod_time_before == mod_time_after

    print("  AC 3 PASS: Script refuses to overwrite existing feature files without --overwrite flag.")

    print("\nAll T5-013 acceptance criteria successfully verified!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
