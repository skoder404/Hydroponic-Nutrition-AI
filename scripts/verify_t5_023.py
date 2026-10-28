"""Verification script for T5-023: Train EfficientNetB0 on lettuce leaf crops.

Acceptance Criteria:
1. Both stages' settings are logged (Stage 1 frozen-backbone, Stage 2 fine-tuning).
2. Same split manifest hash as the CNN run (SHA256: 5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52).
3. Validation report produced (reports/lettuce/lettuce_efficientnet_b0_v1.0.0_validation.{md,json}).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_MANIFEST_HASH = (
    "5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52"
)
LETTUCE_CLASSES = [
    "Healthier",
    "NitrogenDeficiency",
    "PhosphorusDeficiency",
    "PotassiumDeficiency",
]


def verify_t5_023() -> bool:
    print("=== Verifying T5-023: Train EfficientNetB0 on Lettuce Leaf Crops ===")

    model_dir = PROJECT_ROOT / "models/image/lettuce/efficientnet_b0/v1.0.0"
    checkpoint_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    report_md_path = (
        PROJECT_ROOT / "reports/lettuce/lettuce_efficientnet_b0_v1.0.0_validation.md"
    )
    report_json_path = (
        PROJECT_ROOT / "reports/lettuce/lettuce_efficientnet_b0_v1.0.0_validation.json"
    )

    # 1. Artifacts existence
    print("\n[1/4] Checking model artifacts and reports existence...")
    assert checkpoint_path.exists(), f"Model checkpoint missing at {checkpoint_path}"
    assert checkpoint_path.stat().st_size > 1_000_000, (
        f"Checkpoint file too small ({checkpoint_path.stat().st_size} bytes)"
    )
    assert metadata_path.exists(), f"Metadata JSON missing at {metadata_path}"
    assert report_md_path.exists(), f"Report Markdown missing at {report_md_path}"
    assert report_json_path.exists(), f"Report JSON missing at {report_json_path}"
    print(
        f"[PASS] Model checkpoint ({checkpoint_path.stat().st_size:,} bytes), metadata, and validation reports exist."
    )

    # 2. Acceptance Criterion 1: Both stages' settings are logged
    print(
        "\n[2/4] Verifying Acceptance Criterion 1: Both stages' settings are logged..."
    )
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["crop"] == "lettuce", (
        f"Expected crop 'lettuce', got '{meta.get('crop')}'"
    )
    assert meta["family"] == "EfficientNetB0"
    assert meta["architecture"] == "efficientnet_b0"
    assert "device_actually_used" in meta and meta["device_actually_used"] in (
        "cpu",
        "cuda:0",
    )

    assert "stages" in meta, "Missing 'stages' configuration in metadata"
    stages = meta["stages"]
    assert "stage_1_frozen_backbone" in stages, (
        "Missing stage_1_frozen_backbone settings"
    )
    assert "stage_2_finetuning" in stages, "Missing stage_2_finetuning settings"

    s1 = stages["stage_1_frozen_backbone"]
    s2 = stages["stage_2_finetuning"]
    assert s1["lr"] > 0, f"Stage 1 LR must be positive, got {s1['lr']}"
    assert s2["lr"] > 0, f"Stage 2 LR must be positive, got {s2['lr']}"
    assert s1["trainable_parameters"] < s2["trainable_parameters"], (
        f"Stage 1 trainable ({s1['trainable_parameters']}) must be less than Stage 2 ({s2['trainable_parameters']})"
    )
    assert s1["frozen_parameters"] > 0, "Stage 1 must have frozen backbone parameters"
    assert s2["unfreeze_epoch"] >= 2, (
        f"Stage 2 unfreeze epoch must be >= 2, got {s2['unfreeze_epoch']}"
    )
    assert "weights_identifier" in meta, "Weights identifier must be logged in metadata"
    print(
        f"[PASS] Both stages logged: Stage 1 (Frozen Backbone, trainable={s1['trainable_parameters']}, "
        f"lr={s1['lr']}), Stage 2 (Fine-tuning, trainable={s2['trainable_parameters']}, lr={s2['lr']})."
    )

    # 3. Acceptance Criterion 2: Same split manifest hash as the CNN run
    print(
        "\n[3/4] Verifying Acceptance Criterion 2: Same split manifest hash as CNN run..."
    )
    manifest_csv = PROJECT_ROOT / "processed_data/lettuce_leaf_crops/source_map.csv"
    assert manifest_csv.exists(), f"Source map manifest missing at {manifest_csv}"
    actual_manifest_hash = hashlib.sha256(manifest_csv.read_bytes()).hexdigest()
    assert actual_manifest_hash == EXPECTED_MANIFEST_HASH, (
        f"Manifest file hash mismatch: expected {EXPECTED_MANIFEST_HASH}, got {actual_manifest_hash}"
    )

    meta_manifest_hash = meta.get("split_manifest_hash")
    assert meta_manifest_hash == EXPECTED_MANIFEST_HASH, (
        f"Metadata manifest hash mismatch: expected {EXPECTED_MANIFEST_HASH}, got {meta_manifest_hash}"
    )

    split_counts = meta.get("split_counts", {})
    assert split_counts.get("train") == 703, (
        f"Expected 703 train samples, got {split_counts.get('train')}"
    )
    assert split_counts.get("val") == 128, (
        f"Expected 128 val samples, got {split_counts.get('val')}"
    )
    print(
        f"[PASS] Split manifest SHA256 matches CNN run: {actual_manifest_hash} (703 train, 128 val)."
    )

    # 4. Acceptance Criterion 3: Validation report produced
    print("\n[4/4] Verifying Acceptance Criterion 3: Validation report produced...")
    with open(report_json_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["crop"] == "lettuce"
    assert rep["model_name"] == "lettuce_efficientnet_b0_v1.0.0"
    assert rep["split_evaluated"] == "validation", (
        "Report must evaluate validation split exclusively"
    )
    assert rep["split_counts"]["test"] == 0, "Test partition must not be evaluated"

    metrics = rep["metrics"]
    assert "macro_f1" in metrics and metrics["macro_f1"] > 0.0
    assert "accuracy" in metrics and metrics["accuracy"] > 0.0
    assert "per_class" in metrics

    per_class = metrics["per_class"]
    for cls_name in LETTUCE_CLASSES:
        assert cls_name in per_class, (
            f"Missing per-class metrics for class '{cls_name}'"
        )
        assert "f1" in per_class[cls_name], f"Missing F1 for class '{cls_name}'"
        assert "support" in per_class[cls_name], (
            f"Missing support for class '{cls_name}'"
        )

    report_md = report_md_path.read_text(encoding="utf-8")
    assert (
        "# Model Evaluation Report: Lettuce — lettuce_efficientnet_b0_v1.0.0"
        in report_md
    )
    assert "Evaluated Split:** `validation`" in report_md
    assert "Macro-F1" in report_md
    assert "Confusion Matrix" in report_md
    assert EXPECTED_MANIFEST_HASH in report_md or EXPECTED_MANIFEST_HASH in str(
        rep.get("limitations", [])
    )
    print(
        f"[PASS] Validation report verified: Macro-F1={metrics['macro_f1']:.4f}, "
        f"Accuracy={metrics['accuracy']:.4f}, 4 lettuce classes evaluated."
    )

    print("\n>>> All T5-023 acceptance criteria successfully verified! <<<")
    return True


if __name__ == "__main__":
    if verify_t5_023():
        sys.exit(0)
    sys.exit(1)
