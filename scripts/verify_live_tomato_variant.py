#!/usr/bin/env python3
"""Verification script for live-compatible tomato sensor variant (T4-020).

Verifies Acceptance Criteria:
1. Feature availability table (server field vs model feature, units compared) is in the report.
2. Variant has its own metadata, report and class list.
3. Live adapter maps only to features the variant uses.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from backend.app.schemas.sensors import (
    TomatoLiveSensorRequest,
    validate_sensor_request,
)
from ml.adapters.live_tomato_adapter import (
    FEATURE_AVAILABILITY_TABLE,
    format_feature_availability_markdown,
    map_campus_to_live_tomato_features,
)
from ml.labels import (
    SENSOR_CLASSES,
    TOMATO_LIVE_SENSOR_CLASSES,
    TOMATO_SENSOR_CLASSES,
)
from ml.registry.metadata_schema import ModelMetadata
from ml.sensor_models.schemas import (
    TOMATO_LIVE_ORDERED_FEATURE_NAMES,
    TOMATO_ORDERED_FEATURE_NAMES,
)


def verify_feature_availability_table() -> None:
    print("[1/5] Verifying Feature Availability Table...")
    assert len(FEATURE_AVAILABILITY_TABLE) == 8, (
        f"Expected 8 features, got {len(FEATURE_AVAILABILITY_TABLE)}"
    )
    table_features = [row["model_feature"] for row in FEATURE_AVAILABILITY_TABLE]
    assert table_features == list(TOMATO_ORDERED_FEATURE_NAMES), (
        f"Table features {table_features} do not match {list(TOMATO_ORDERED_FEATURE_NAMES)}"
    )

    npk = [
        row
        for row in FEATURE_AVAILABILITY_TABLE
        if row["model_feature"]
        in ("nitrogen_mg_L", "phosphorus_mg_L", "potassium_mg_L")
    ]
    assert len(npk) == 3
    for row in npk:
        assert "Missing" in row["status"] or "Disabled" in row["status"]
        assert "_deleted_keys" in row["notes"] or "deleted" in row["notes"].lower()

    md_table = format_feature_availability_markdown()
    assert "| Model Feature | Server Field |" in md_table
    assert "`air_temperature_C`" in md_table
    assert "**Available**" in md_table
    print(
        "  -> Feature availability table covers 8 features, compares units, and notes N/P/K hardware deletion."
    )


def verify_class_list() -> None:
    print("[2/5] Verifying Variant Class List...")
    assert len(TOMATO_LIVE_SENSOR_CLASSES) == 7, (
        f"Expected 7 live classes, got {len(TOMATO_LIVE_SENSOR_CLASSES)}"
    )
    assert len(TOMATO_SENSOR_CLASSES) == 10, (
        f"Expected 10 full classes, got {len(TOMATO_SENSOR_CLASSES)}"
    )

    excluded = {"N_Deficiency", "P_Deficiency", "K_Deficiency"}
    for c in excluded:
        assert c not in TOMATO_LIVE_SENSOR_CLASSES, (
            f"{c} should be excluded from live variant!"
        )
        assert c in TOMATO_SENSOR_CLASSES, f"{c} must remain in full 10-class model!"

    assert "tomato_live" in SENSOR_CLASSES
    assert SENSOR_CLASSES["tomato_live"] == TOMATO_LIVE_SENSOR_CLASSES
    print(
        "  -> Variant class list verified: 7 classes (N, P, K deficiencies strictly excluded)."
    )


def verify_live_adapter() -> None:
    print("[3/5] Verifying Live Adapter Mapping...")
    sample_payload = {
        "temp_avg": 40.0,
        "temperature1": 41.2,
        "humidity_avg": 36.8,
        "humidity1": 35.1,
        "ph": 7.67,
        "ec": 36.8,  # in µS/cm
        "tds": 18.3,
        "salinity": 0.02,
        "nitrogen_mg_L": 120.0,
        "phosphorus": 45.0,
        "potassium": 200.0,
    }

    mapped = map_campus_to_live_tomato_features(sample_payload)
    assert list(mapped.keys()) == list(TOMATO_LIVE_ORDERED_FEATURE_NAMES), (
        f"Mapped keys {list(mapped.keys())} do not match {list(TOMATO_LIVE_ORDERED_FEATURE_NAMES)}"
    )
    assert len(mapped) == 5

    # Check EC scaled: 36.8 µS/cm -> 0.0368 mS/cm
    assert abs(mapped["ec_mS_cm"] - 0.0368) < 1e-4

    # Check thermal proxy: 8.0 + 0.55 * 40.0 = 30.0
    assert abs(mapped["water_temperature_C"] - 30.0) < 1e-2

    # Check N/P/K strictly omitted
    for bad in (
        "nitrogen_mg_L",
        "phosphorus_mg_L",
        "potassium_mg_L",
        "nitrogen",
        "phosphorus",
        "potassium",
    ):
        assert bad not in mapped, (
            f"Prohibited key '{bad}' found in live adapter output!"
        )
    print(
        "  -> Live adapter verified: maps strictly to 5 live features, scales EC, computes thermal proxy, drops N/P/K."
    )


def verify_request_schema() -> None:
    print("[4/5] Verifying TomatoLiveSensorRequest Schema...")
    valid_payload = {
        "air_temperature_C": 25.0,
        "relative_humidity_percent": 65.0,
        "water_temperature_C": 21.0,
        "ph": 6.2,
        "ec_mS_cm": 2.5,
    }
    req = validate_sensor_request("tomato_live", valid_payload)
    assert isinstance(req, TomatoLiveSensorRequest)
    feat_dict = req.to_feature_dict()
    assert len(feat_dict) == 5
    assert feat_dict["ph"] == 6.2

    # Verify extra fields forbidden
    try:
        validate_sensor_request(
            "tomato_live", {**valid_payload, "nitrogen_mg_L": 100.0}
        )
        raise AssertionError("Failed to reject extra field nitrogen_mg_L!")
    except (ValidationError, ValueError):
        pass
    print(
        "  -> TomatoLiveSensorRequest verified: strictly accepts 5 features, extra='forbid' enforced."
    )


def verify_reports_and_metadata() -> None:
    print("[5/5] Verifying Reports and Model Artifacts...")
    reports_dir = ROOT / "reports" / "tomato_live"
    models_dir = ROOT / "models" / "sensor" / "tomato_live" / "v1.0.0"

    assert reports_dir.is_dir(), f"Reports directory {reports_dir} does not exist!"
    assert models_dir.is_dir(), f"Models directory {models_dir} does not exist!"

    exp_json = reports_dir / "tomato_live_sensor_experiments.json"
    exp_md = reports_dir / "tomato_live_sensor_experiments.md"
    assert exp_json.is_file(), f"Missing {exp_json}"
    assert exp_md.is_file(), f"Missing {exp_md}"

    json_data = json.loads(exp_json.read_text(encoding="utf-8"))
    assert json_data["variant"] == "live_compatible"
    assert json_data["feature_set"] == "live_compatible"
    assert len(json_data["features"]) == 5
    assert len(json_data["classes"]) == 7
    assert json_data["excluded_classes"] == [
        "N_Deficiency",
        "P_Deficiency",
        "K_Deficiency",
    ]
    assert json_data["experimental"] is True
    assert len(json_data["model_summaries"]) == 5

    md_content = exp_md.read_text(encoding="utf-8")
    assert "Feature Availability Table" in md_content
    assert "| `air_temperature_C` | `temp_avg` |" in md_content
    assert "| `nitrogen_mg_L` | `nitrogen` |" in md_content
    assert "Excluded Deficiencies" in md_content

    # Verify metadata.json
    meta_path = models_dir / "metadata.json"
    assert meta_path.is_file(), f"Missing {meta_path}"
    meta = ModelMetadata.load(meta_path)
    assert meta.feature_set == "live_compatible"
    assert meta.experimental is True
    assert meta.ordered_features is not None
    assert len(meta.ordered_features) == 5
    assert len(meta.classes) == 7
    print(
        "  -> Reports and metadata verified: feature availability table present, feature_set='live_compatible', experimental=True."
    )


def main() -> int:
    print("=" * 60)
    print("T4-020: Live-Compatible Tomato Sensor Variant Verification")
    print("=" * 60)
    verify_feature_availability_table()
    verify_class_list()
    verify_live_adapter()
    verify_request_schema()
    verify_reports_and_metadata()
    print("=" * 60)
    print("ALL T4-020 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
