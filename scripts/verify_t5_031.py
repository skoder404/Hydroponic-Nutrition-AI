"""Verification script for T5-031: Derive per-leaf uncertainty thresholds from validation data.

Acceptance Criteria:
1. Thresholds are computed from validation only.
2. Metadata states score semantics for each promoted model.
3. Reliability plot saved.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CROPS = ["lettuce", "spinach", "tomato"]


def verify_t5_031() -> bool:
    print("=== Verifying T5-031: Derive Per-Leaf Uncertainty Thresholds from Validation Data ===")

    cal_dir = PROJECT_ROOT / "reports/calibration"
    report_json = cal_dir / "uncertainty_thresholds.json"
    report_md = cal_dir / "uncertainty_thresholds.md"
    plot_combined = cal_dir / "reliability_plots_combined.png"

    # 1. Check reports and reliability plot existence
    print("\n[1/3] Checking report artifacts and reliability plot existence...")
    assert report_json.exists(), f"Missing calibration json at {report_json}"
    assert report_md.exists(), f"Missing calibration markdown at {report_md}"
    assert plot_combined.exists(), f"Missing reliability plot at {plot_combined}"
    assert plot_combined.stat().st_size > 10_000, f"Plot file too small: {plot_combined.stat().st_size} bytes"
    print(f"[PASS] Calibration reports and reliability plot exist ({plot_combined.name}, {plot_combined.stat().st_size:,} bytes).")

    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Acceptance Criterion 1: Thresholds computed from validation only
    print("\n[2/3] Verifying Acceptance Criterion 1: Thresholds computed from validation only...")
    assert data["split_evaluated"] == "validation", (
        f"Expected split_evaluated to be 'validation', got '{data.get('split_evaluated')}'"
    )

    for crop in CROPS:
        crop_data = data["crops"][crop]
        assert crop_data["thresholds"]["split_evaluated"] == "validation"
        assert crop_data["sample_count"] > 0
        th = crop_data["thresholds"]
        assert 0.0 < th["uncertain"] <= th["moderate"] <= th["high"] <= 1.0, (
            f"Invalid threshold ordering for {crop}: {th}"
        )
        assert crop_data["post_scaling_ece"] <= crop_data["pre_scaling_ece"] + 1e-4, (
            f"Temperature scaling did not improve ECE for {crop}: {crop_data}"
        )
        print(
            f"[PASS] {crop.capitalize()}: N_val={crop_data['sample_count']}, "
            f"High>={th['high']}, Mod>={th['moderate']}, Uncertain<{th['uncertain']}, "
            f"Pre-ECE={crop_data['pre_scaling_ece']:.4f} -> Post-ECE={crop_data['post_scaling_ece']:.4f} (T={crop_data['fitted_temperature']})"
        )

    # Acceptance Criterion 2: Metadata states score semantics
    print("\n[3/3] Verifying Acceptance Criterion 2: Metadata states score semantics...")
    expected_keyword = "uncalibrated"
    assert expected_keyword in data["score_semantics"].lower()

    for crop in CROPS:
        meta_path = PROJECT_ROOT / f"models/image/{crop}/efficientnet_b0/v1.0.0/metadata.json"
        assert meta_path.exists(), f"Metadata missing at {meta_path}"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "score_semantics" in meta, f"Metadata for {crop} missing score_semantics"
        assert "uncertainty_thresholds" in meta, f"Metadata for {crop} missing uncertainty_thresholds"
        assert "temperature_scaling" in meta, f"Metadata for {crop} missing temperature_scaling"

        semantics = meta["score_semantics"].lower()
        assert "raw softmax" in semantics or "uncalibrated" in semantics, (
            f"Metadata score_semantics for {crop} does not explain score nature: {meta['score_semantics']}"
        )
        assert "inherited" in semantics or "plant" in semantics, (
            f"Metadata score_semantics for {crop} does not explain foliar label inheritance: {meta['score_semantics']}"
        )

        assert meta["uncertainty_thresholds"]["split_evaluated"] == "validation"
        assert meta["temperature_scaling"]["optimization_split"] == "validation"
        print(f"[PASS] Metadata verified for {crop} EfficientNetB0 (score semantics & validation thresholds present).")

    print("\n>>> All T5-031 acceptance criteria successfully verified! <<<")
    return True


if __name__ == "__main__":
    verify_t5_031()

