"""Unit tests for Tomato Kolar Provenance & Severity Scales (T2-008)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROVENANCE_DOC = ROOT / "docs" / "datasets" / "tomato_sensor_kolar.md"
CONTRACT_DOC = ROOT / "docs" / "datasets" / "tomato_sensor_contract.md"
MANIFEST_FILE = ROOT / "processed_data" / "sensor" / "tomato" / "manifest.json"

EXPECTED_SHA256 = "c4032fdc026cc4cb971119fdc27d709bbb4d9b6e24d55df87eebd20464a9d515"
EXPECTED_SEVERITY_SCALES = {
    "Water_Temp_High": 6.58,
    "EC_High_Toxicity": 0.20,
    "EC_Low_Deficiency": 1.30,
    "pH_Imbalance": 1.00,
    "Humidity_High_Mildew_Risk": 5.00,
    "Humidity_Low_Stress": 5.00,
    "N_Deficiency": 50.00,
    "P_Deficiency": 20.00,
    "K_Deficiency": 54.00,
}


def test_provenance_sheet_records() -> None:
    """Verify download date, version, SHA-256, and row/column counts in provenance doc."""
    assert PROVENANCE_DOC.is_file()
    text = PROVENANCE_DOC.read_text(encoding="utf-8")

    assert EXPECTED_SHA256 in text
    assert "2026-10-25" in text
    assert "Version 1" in text
    assert "200,000 rows" in text
    assert "29 columns" in text
    assert "build_tomato_dataset.py" in text
    assert "--mode primary" in text


def test_manifest_severity_scales_and_archive() -> None:
    """Verify severity scales and raw archive metadata in tomato manifest.json."""
    assert MANIFEST_FILE.is_file()
    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    raw = data.get("raw_archive", {})
    assert raw.get("sha256") == EXPECTED_SHA256
    assert raw.get("rows") == 200000
    assert raw.get("columns") == 29
    assert raw.get("download_date") == "2026-10-25"

    scales = data.get("severity_scales", {})
    assert len(scales) == 9
    for rule, exp_val in EXPECTED_SEVERITY_SCALES.items():
        assert abs(float(scales[rule]) - exp_val) < 0.001

    scales_meta = data.get("severity_scales_metadata", {})
    assert scales_meta.get("script_version") == "1.0"


def test_contract_severity_scales_table() -> None:
    """Verify technical contract contains severity scale constants and script version."""
    assert CONTRACT_DOC.is_file()
    text = CONTRACT_DOC.read_text(encoding="utf-8")

    assert "Nine Severity Scale Constants" in text
    assert "build_tomato_dataset.py" in text
    assert "v1.0" in text

    for rule, exp_val in EXPECTED_SEVERITY_SCALES.items():
        assert rule in text
        assert f"{exp_val:.2f}" in text
