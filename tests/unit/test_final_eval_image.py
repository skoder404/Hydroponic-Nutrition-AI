"""Unit tests verifying run-once final test evaluation for image models (T5-032).

Acceptance Criteria:
- Log entry is written for each evaluated model
- Re-run without --force refuses
- Reports label metrics as test results with split named
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.final_eval_image import check_already_evaluated, run_final_test_evaluation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
LOG_FILE = REPORTS_DIR / "final_eval_log.jsonl"


def test_final_eval_log_contains_all_image_models():
    """Verify log entries are written for lettuce, spinach, and tomato image models."""
    assert LOG_FILE.exists(), f"Missing final evaluation log: {LOG_FILE}"

    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    image_entries = [
        e
        for e in entries
        if e.get("modality") == "image"
        or "efficientnet" in str(e.get("model_name", "")).lower()
    ]

    crops_logged = {e["crop"] for e in image_entries}
    assert {"lettuce", "spinach", "tomato"}.issubset(crops_logged), (
        f"Expected all 3 crops logged, found: {crops_logged}"
    )

    for entry in image_entries:
        assert entry["split_evaluated"] == "test", "Split must be labeled 'test'"
        assert "test_metrics" in entry
        metrics = entry["test_metrics"]
        assert "macro_f1" in metrics
        assert "accuracy" in metrics
        assert metrics["macro_f1"] > 0.0
        assert metrics["accuracy"] > 0.0
        assert entry["test_sample_count"] > 0
        assert Path(entry["report_json"]).exists()
        assert Path(entry["report_md"]).exists()


def test_rerun_without_force_refuses():
    """Verify re-running final test evaluation without --force refuses with RuntimeError."""
    assert check_already_evaluated("lettuce", LOG_FILE)
    assert check_already_evaluated("spinach", LOG_FILE)
    assert check_already_evaluated("tomato", LOG_FILE)

    with pytest.raises(RuntimeError, match="Re-running requires --force"):
        run_final_test_evaluation(
            crop="lettuce",
            log_file=LOG_FILE,
            force=False,
        )


def test_rerun_with_force_requires_reason():
    """Verify re-running with --force but without a non-empty reason refuses with ValueError."""
    with pytest.raises(ValueError, match="requires a non-empty written --reason"):
        run_final_test_evaluation(
            crop="lettuce",
            log_file=LOG_FILE,
            force=True,
            reason="",
        )


def test_test_reports_label_metrics_as_test_results():
    """Verify generated reports explicitly label metrics as test results with split named."""
    for crop in ["lettuce", "spinach", "tomato"]:
        json_report = REPORTS_DIR / crop / f"{crop}_efficientnet_b0_v1.0.0_test.json"
        md_report = REPORTS_DIR / crop / f"{crop}_efficientnet_b0_v1.0.0_test.md"

        assert json_report.exists(), f"Missing JSON report for {crop}"
        assert md_report.exists(), f"Missing MD report for {crop}"

        with open(json_report, encoding="utf-8") as f:
            data = json.load(f)

        assert data["split_evaluated"] == "test"
        assert "metrics" in data
        assert "macro_f1" in data["metrics"]
        assert "accuracy" in data["metrics"]
        assert data["split_counts"]["test"] > 0

        md_content = md_report.read_text(encoding="utf-8")
        assert "Evaluated Split:** `test`" in md_content or "`test`" in md_content
        assert "Final Test Evaluation" in md_content


def test_model_metadata_contains_test_metrics():
    """Verify model metadata files are updated with test metrics."""
    for crop in ["lettuce", "spinach", "tomato"]:
        meta_path = (
            MODELS_DIR / "image" / crop / "efficientnet_b0" / "v1.0.0" / "metadata.json"
        )
        assert meta_path.exists(), f"Missing metadata for {crop}"

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        assert "test_metrics" in meta, f"Metadata missing test_metrics for {crop}"
        assert "macro_f1" in meta["test_metrics"]
        assert "accuracy" in meta["test_metrics"]
        assert "test_evaluation_date" in meta
        assert meta["test_sample_count"] > 0
