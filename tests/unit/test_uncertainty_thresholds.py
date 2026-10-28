"""Unit tests for T5-031: Derive per-leaf uncertainty thresholds from validation data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CROPS = ["lettuce", "spinach", "tomato"]


def test_calibration_artifacts_and_plots_exist() -> None:
    """Verify that calibration reports and reliability diagrams exist on disk."""
    cal_dir = PROJECT_ROOT / "reports/calibration"
    report_json = cal_dir / "uncertainty_thresholds.json"
    report_md = cal_dir / "uncertainty_thresholds.md"
    plot_combined = cal_dir / "reliability_plots_combined.png"

    assert report_json.exists(), f"Missing calibration json at {report_json}"
    assert report_md.exists(), f"Missing calibration markdown at {report_md}"
    assert plot_combined.exists(), f"Missing reliability plot at {plot_combined}"
    assert plot_combined.stat().st_size > 10_000


def test_thresholds_computed_from_validation_only() -> None:
    """Verify that thresholds and scaling are strictly marked as validation-only."""
    report_json = PROJECT_ROOT / "reports/calibration/uncertainty_thresholds.json"
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_id"] == "T5-031"
    assert data["split_evaluated"] == "validation"

    for crop in CROPS:
        crop_data = data["crops"][crop]
        th = crop_data["thresholds"]
        assert th["split_evaluated"] == "validation"
        assert 0.0 < th["uncertain"] <= th["moderate"] <= th["high"] <= 1.0
        # Post-scaling ECE must be lower than or equal to pre-scaling
        assert crop_data["post_scaling_ece"] <= crop_data["pre_scaling_ece"] + 1e-4


def test_metadata_contains_score_semantics_and_thresholds() -> None:
    """Verify that each promoted model metadata records score semantics and uncertainty thresholds."""
    for crop in CROPS:
        meta_path = PROJECT_ROOT / f"models/image/{crop}/efficientnet_b0/v1.0.0/metadata.json"
        assert meta_path.exists(), f"Missing metadata at {meta_path}"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "score_semantics" in meta
        semantics = meta["score_semantics"].lower()
        assert "uncalibrated" in semantics or "raw softmax" in semantics
        assert "inherited" in semantics or "plant" in semantics

        assert "uncertainty_thresholds" in meta
        th = meta["uncertainty_thresholds"]
        assert th["split_evaluated"] == "validation"
        assert 0.0 < th["uncertain"] <= th["moderate"] <= th["high"] <= 1.0

        assert "temperature_scaling" in meta
        ts = meta["temperature_scaling"]
        assert ts["optimization_split"] == "validation"
        assert ts["fitted_temperature"] > 0.0

