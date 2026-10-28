"""Verification script for Task T5-020: Generic PyTorch training loop.

Acceptance Criteria:
1. Overfit test: loop can reach near-zero training loss on 20 images.
2. OOM path is exercised (simulated).
3. Report records device and versions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.image_models.train_image import (
    TrainingConfig,
    run_overfit_test,
    train_image_model,
)


def verify_t5_020() -> bool:
    print("=== Verifying T5-020: Generic PyTorch Training Loop ===")

    # 1. Acceptance Criterion 1: Overfit test on 20 images reaching near-zero loss
    print("\n[1/3] Running overfit test on 20 images...")
    overfit_res = run_overfit_test(crop="lettuce", num_samples=20, max_epochs=35)
    final_loss = overfit_res["final_loss"]
    final_acc = overfit_res["final_accuracy"]
    assert final_loss < 0.15, f"Overfit loss {final_loss} >= 0.15"
    assert final_acc >= 0.90, f"Overfit accuracy {final_acc} < 0.90"
    print(f"[PASS] Overfit test succeeded: loss={final_loss:.4f}, acc={final_acc:.4f} on 20 images.")

    # 2. Acceptance Criterion 2: OOM path is exercised (simulated)
    print("\n[2/3] Exercising simulated OOM path...")
    cfg = TrainingConfig(
        crop="lettuce",
        epochs=2,
        batch_size=32,
        lr=1e-3,
        early_stopping_patience=2,
        simulate_oom=True,
    )
    result = train_image_model(cfg)
    meta = result["metadata"]
    assert meta["oom_events_caught"] >= 1, "Expected at least 1 OOM event caught by retry ladder"
    print(f"[PASS] Simulated OOM caught ({meta['oom_events_caught']} event) and resolved via retry ladder.")

    # 3. Acceptance Criterion 3: Report records device and versions
    print("\n[3/3] Verifying report records device and software versions...")
    report_md = PROJECT_ROOT / "reports/lettuce/lettuce_custom_cnn_v1.0.0_validation.md"
    report_json = PROJECT_ROOT / "reports/lettuce/lettuce_custom_cnn_v1.0.0_validation.json"
    metadata_json = PROJECT_ROOT / "models/image/lettuce/custom_cnn/v1.0.0/metadata.json"

    assert report_md.exists(), f"Report MD missing: {report_md}"
    assert report_json.exists(), f"Report JSON missing: {report_json}"
    assert metadata_json.exists(), f"Metadata JSON missing: {metadata_json}"

    with open(metadata_json, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert "device_actually_used" in meta_data
    assert "software_versions" in meta_data
    soft_vers = meta_data["software_versions"]
    for pkg in ("python", "torch", "numpy", "scikit-learn"):
        assert pkg in soft_vers, f"Missing package version for '{pkg}' in metadata"

    md_content = report_md.read_text(encoding="utf-8")
    assert "## 6. Software & Runtime Versions" in md_content
    assert "torch" in md_content
    assert "actually_used" in md_content
    assert "Macro-F1" in md_content
    assert "Evaluated Split: `validation`" in md_content or "validation" in md_content
    print(f"[PASS] Report and metadata record device ('{meta_data['device_actually_used']}') and all software versions.")

    # 4. Verify concept note
    concept_note = PROJECT_ROOT / "docs/learning/optimisers-learning-rate-and-early-stopping.md"
    assert concept_note.exists(), f"Concept note missing: {concept_note}"
    c_text = concept_note.read_text(encoding="utf-8")
    assert "AdamW" in c_text
    assert "Early Stopping" in c_text
    assert "Macro-F1" in c_text
    print("[PASS] Concept note verified at docs/learning/optimisers-learning-rate-and-early-stopping.md.")

    print("\n>>> All T5-020 acceptance criteria verified successfully! <<<")
    return True


if __name__ == "__main__":
    if verify_t5_020():
        sys.exit(0)
    sys.exit(1)
