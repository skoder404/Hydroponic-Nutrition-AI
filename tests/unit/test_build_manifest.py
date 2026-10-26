"""Unit tests for dataset manifest generator (T2-020).

Validates Acceptance Criteria:
1. Manifest exists for each sensor and image dataset.
2. Count changes between stages are explained or the script fails loudly.
3. Output is deterministic.
4. All Overview §9.1 fields are present and conform to repository contracts.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_manifest import (
    APPROVED_PROVENANCE_TAGS,
    DATASET_CONFIGS,
    DATASET_KEYS,
    UnexplainedCountChangeError,
    build_all_manifests,
    build_manifest_for_dataset,
    save_manifest,
)

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = ROOT / "processed_data" / "manifests"


def test_manifest_exists_for_each_dataset() -> None:
    """AC 1: Manifest exists for each sensor and image dataset."""
    assert len(DATASET_KEYS) == 6
    saved = build_all_manifests(repo_root=ROOT, output_dir=MANIFEST_DIR)

    for d_key in DATASET_KEYS:
        canon_file = MANIFEST_DIR / f"{d_key}.json"
        alias_file = MANIFEST_DIR / f"{d_key}_manifest.json"

        assert canon_file.is_file(), f"Missing canonical manifest: {canon_file}"
        assert alias_file.is_file(), f"Missing alias manifest: {alias_file}"
        assert d_key in saved


@pytest.mark.parametrize("d_key", DATASET_KEYS)
def test_manifest_overview_section_9_fields(d_key: str) -> None:
    """Verify each manifest contains all Overview §9.1 mandatory fields."""
    manifest = build_manifest_for_dataset(d_key, repo_root=ROOT)

    # 1. Top-level metadata
    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["dataset_key"] == d_key
    assert manifest["crop"] in ("lettuce", "spinach", "tomato")
    assert manifest["modality"] in ("sensor", "image")
    assert "git_commit" in manifest
    assert "generated_at" in manifest

    # 2. Provenance §9.1
    prov = manifest["provenance"]
    assert "source_name" in prov and len(prov["source_name"]) > 0
    assert "source_url" in prov and prov["source_url"].startswith("http")
    assert "source_version" in prov
    assert "license" in prov
    assert "redistribution_status" in prov
    assert "original_filenames" in prov and len(prov["original_filenames"]) > 0

    # 3. Counts §9.1
    counts = manifest["counts"]
    assert counts["original_raw_count"] > 0
    assert counts["prepared_count"] > 0
    assert counts["unexplained_count_change"] == 0
    assert "splits" in counts
    assert "train" in counts["splits"]
    assert "val" in counts["splits"]
    assert "test" in counts["splits"]

    # 4. Classes §9.1
    classes = manifest["classes"]
    assert len(classes["retained_classes"]) > 0
    assert "class_counts_per_split" in classes
    for split in ("train", "val", "test"):
        assert split in classes["class_counts_per_split"]
        assert len(classes["class_counts_per_split"][split]) > 0

    # 5. Split Protocol §9.1
    proto = manifest["split_protocol"]
    assert "protocol_name" in proto
    assert "leakage_prevention" in proto
    assert "group_keys" in proto

    # 6. Preprocessing Script & Git Commit §9.1
    pre = manifest["preprocessing"]
    assert "script" in pre
    assert "git_commit" in pre
    assert "cleaning_steps" in pre

    # 7. Derived / Synthetic Flags §9.1
    flags = manifest["derived_synthetic_flags"]
    assert "columns" in flags
    assert "has_synthetic_data" in flags
    for col, tag in flags["columns"].items():
        assert tag in APPROVED_PROVENANCE_TAGS, f"Invalid tag {tag} in {col}"

    # 8. Known Limitations §9.1
    assert "known_limitations" in manifest
    assert len(manifest["known_limitations"]) > 0


def test_unexplained_count_change_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC 2: Count changes between stages are explained or the script fails loudly."""
    # Tamper with lettuce_sensor config so unexplained discrepancy is non-zero
    tampered_cfg = copy.deepcopy(DATASET_CONFIGS["lettuce_sensor"])
    tampered_cfg["counts_specification"]["original_raw_count"] = 999  # was 726

    monkeypatch.setitem(DATASET_CONFIGS, "lettuce_sensor", tampered_cfg)

    with pytest.raises(UnexplainedCountChangeError) as exc_info:
        build_manifest_for_dataset("lettuce_sensor", repo_root=ROOT, fail_on_unexplained=True)

    assert "Unexplained count change" in str(exc_info.value)
    assert "discrepancy=273" in str(exc_info.value)


def test_output_is_deterministic(tmp_path: Path) -> None:
    """AC 3: Manifest output is deterministic."""
    manifest1 = build_manifest_for_dataset("spinach_image", repo_root=ROOT)
    manifest2 = build_manifest_for_dataset("spinach_image", repo_root=ROOT)

    p1 = save_manifest(manifest1, tmp_path / "spinach1.json")
    p2 = save_manifest(manifest2, tmp_path / "spinach2.json")

    content1 = p1.read_text(encoding="utf-8")
    content2 = p2.read_text(encoding="utf-8")

    assert content1 == content2, "Manifest serialization must be deterministic byte-for-byte."


def test_cli_execution() -> None:
    """Test CLI execution for individual dataset and invalid dataset handling."""
    # Individual dataset
    res = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "build_manifest.py"), "tomato_image"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0
    assert "Generated manifest for 'tomato_image'" in res.stdout

    # Invalid dataset fails with non-zero exit code
    res_err = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "build_manifest.py"), "invalid_banana_dataset"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res_err.returncode != 0
    assert "Error building manifest" in res_err.stderr
