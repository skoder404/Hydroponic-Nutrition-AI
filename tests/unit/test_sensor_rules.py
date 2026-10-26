"""Unit tests for sensor rule-label reference oracles (T4-016)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.sensor_models.rules import (
    LETTUCE_ACTION_BANDS,
    TOMATO_THRESHOLDS,
    compute_tomato_deviations,
    evaluate_crop_rules_df,
    evaluate_lettuce_rules_df,
    evaluate_spinach_rules_df,
    evaluate_tomato_rules_df,
    explain_rule_prediction,
    get_lettuce_broken_rules,
    get_spinach_broken_rules,
    get_tomato_broken_rules,
    predict_crop_rule_label,
    predict_lettuce_rule_label,
    predict_spinach_rule_label,
    predict_tomato_rule_label,
)


class TestLettuceRules:
    """Test suite for lettuce sensor rules."""

    @pytest.fixture
    def healthy_lettuce(self) -> dict[str, float]:
        return {
            "ph": 6.2,
            "ec": 1.8,
            "air_temp": 24.0,
            "humidity": 65.0,
            "water_temp": 22.0,
        }

    def test_healthy_row_is_normal(self, healthy_lettuce: dict[str, float]) -> None:
        assert predict_lettuce_rule_label(healthy_lettuce) == "Normal"
        assert get_lettuce_broken_rules(healthy_lettuce) == []

    @pytest.mark.parametrize(
        ("feature", "value", "expected_label"),
        [
            ("ph", 5.4, "pH_Low"),
            ("ph", 7.1, "pH_High"),
            ("ec", 1.1, "EC_Low"),
            ("ec", 2.6, "EC_High"),
            ("air_temp", 17.5, "Temp_Low"),
            ("air_temp", 30.5, "Temp_High"),
            ("humidity", 49.0, "Humidity_Low"),
            ("humidity", 81.0, "Humidity_High"),
            ("water_temp", 17.5, "WaterTemp_Low"),
            ("water_temp", 26.5, "WaterTemp_High"),
        ],
    )
    def test_single_variable_out_assigns_class(
        self,
        healthy_lettuce: dict[str, float],
        feature: str,
        value: float,
        expected_label: str,
    ) -> None:
        sample = {**healthy_lettuce, feature: value}
        assert predict_lettuce_rule_label(sample) == expected_label
        assert get_lettuce_broken_rules(sample) == [expected_label]

    def test_multiple_variables_out_gives_multiple_stress(
        self,
        healthy_lettuce: dict[str, float],
    ) -> None:
        sample = {**healthy_lettuce, "ph": 5.0, "ec": 3.0}
        assert predict_lettuce_rule_label(sample) == "Multiple_Stress"
        assert len(get_lettuce_broken_rules(sample)) == 2

    def test_boundary_values_are_in_band(
        self,
        healthy_lettuce: dict[str, float],
    ) -> None:
        # Boundaries: ph=(5.5, 7.0), ec=(1.2, 2.5), air_temp=(18.0, 30.0), hum=(50.0, 80.0), wt=(18.0, 26.0)
        for var, (lo, hi) in LETTUCE_ACTION_BANDS.items():
            sample_lo = {**healthy_lettuce, var: lo}
            assert predict_lettuce_rule_label(sample_lo) == "Normal", f"{var}={lo} should be Normal"

            sample_hi = {**healthy_lettuce, var: hi}
            assert predict_lettuce_rule_label(sample_hi) == "Normal", f"{var}={hi} should be Normal"

    def test_empty_or_all_nan_returns_unknown(self) -> None:
        assert predict_lettuce_rule_label({}) == "Unknown"
        assert predict_lettuce_rule_label({"ph": float("nan")}) == "Unknown"

    def test_vectorized_df_matches_row_by_row(
        self,
        healthy_lettuce: dict[str, float],
    ) -> None:
        rows = [
            healthy_lettuce,
            {**healthy_lettuce, "ph": 5.0},
            {**healthy_lettuce, "ec": 3.0},
            {**healthy_lettuce, "ph": 5.0, "ec": 3.0},
        ]
        df = pd.DataFrame(rows)
        batch = evaluate_lettuce_rules_df(df)
        single = pd.Series([predict_lettuce_rule_label(r) for r in rows])
        pd.testing.assert_series_equal(batch, single, check_names=False, check_dtype=False)


class TestSpinachRules:
    """Test suite for spinach sensor rules."""

    @pytest.fixture
    def healthy_spinach(self) -> dict[str, float]:
        return {
            "air_temperature_C": 22.0,
            "relative_humidity_percent": 60.0,
            "water_temperature_C": 19.5,
            "ph": 6.5,
            "ec_mS_cm": 2.0,
        }

    def test_healthy_spinach_is_normal(self, healthy_spinach: dict[str, float]) -> None:
        assert predict_spinach_rule_label(healthy_spinach) == "Normal"
        assert get_spinach_broken_rules(healthy_spinach) == []

    @pytest.mark.parametrize(
        ("feature", "value", "expected_label"),
        [
            ("water_temperature_C", 21.5, "Water_Temp_High"),
            ("ec_mS_cm", 2.35, "EC_High_Toxicity"),
            ("ec_mS_cm", 1.75, "EC_Low_Deficiency"),
            ("ph", 5.8, "pH_Imbalance"),
            ("ph", 7.2, "pH_Imbalance"),
            ("relative_humidity_percent", 76.0, "Humidity_High_Mildew_Risk"),
            ("relative_humidity_percent", 44.0, "Humidity_Low_Stress"),
        ],
    )
    def test_spinach_single_rules(
        self,
        healthy_spinach: dict[str, float],
        feature: str,
        value: float,
        expected_label: str,
    ) -> None:
        sample = {**healthy_spinach, feature: value}
        assert predict_spinach_rule_label(sample) == expected_label
        assert get_spinach_broken_rules(sample) == [expected_label]

    def test_spinach_boundary_values(self, healthy_spinach: dict[str, float]) -> None:
        # Boundaries: wt=21.0, ec=2.3, ec=1.8, ph=6.0, ph=7.0, hum=75.0, hum=45.0
        assert predict_spinach_rule_label({**healthy_spinach, "water_temperature_C": 21.0}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "ec_mS_cm": 2.3}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "ec_mS_cm": 1.8}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "ph": 6.0}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "ph": 7.0}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "relative_humidity_percent": 75.0}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "relative_humidity_percent": 45.0}) == "Normal"

    def test_spinach_multiple_broken(self, healthy_spinach: dict[str, float]) -> None:
        sample = {**healthy_spinach, "water_temperature_C": 22.0, "ec_mS_cm": 2.5}
        assert predict_spinach_rule_label(sample) == "Multiple"

    def test_spinach_df_matches_row_by_row(self, healthy_spinach: dict[str, float]) -> None:
        rows = [
            healthy_spinach,
            {**healthy_spinach, "water_temperature_C": 22.0},
            {**healthy_spinach, "ph": 5.5},
            {**healthy_spinach, "ec_mS_cm": 2.5, "ph": 5.5},
        ]
        df = pd.DataFrame(rows)
        batch = evaluate_spinach_rules_df(df)
        single = pd.Series([predict_spinach_rule_label(r) for r in rows])
        pd.testing.assert_series_equal(batch, single, check_names=False, check_dtype=False)


class TestTomatoRules:
    """Test suite for tomato sensor rules."""

    @pytest.fixture
    def healthy_tomato(self) -> dict[str, float]:
        return {
            "air_temperature_C": 24.0,
            "relative_humidity_percent": 65.0,
            "water_temperature_C": 23.0,
            "ph": 6.2,
            "ec_mS_cm": 2.8,
            "nitrogen_mg_L": 140.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 220.0,
        }

    def test_healthy_tomato_is_normal(self, healthy_tomato: dict[str, float]) -> None:
        assert predict_tomato_rule_label(healthy_tomato) == "Normal"
        assert get_tomato_broken_rules(healthy_tomato) == []

    @pytest.mark.parametrize(
        ("feature", "value", "expected_label"),
        [
            ("water_temperature_C", 28.0, "Water_Temp_High"),
            ("ec_mS_cm", 4.15, "EC_High_Toxicity"),
            ("ec_mS_cm", 1.5, "EC_Low_Deficiency"),
            ("ph", 5.8, "pH_Imbalance"),
            ("ph", 6.8, "pH_Imbalance"),
            ("relative_humidity_percent", 88.0, "Humidity_High_Mildew_Risk"),
            ("relative_humidity_percent", 45.0, "Humidity_Low_Stress"),
            ("nitrogen_mg_L", 70.0, "N_Deficiency"),
            ("phosphorus_mg_L", 25.0, "P_Deficiency"),
            ("potassium_mg_L", 110.0, "K_Deficiency"),
        ],
    )
    def test_single_tomato_rule_broken(
        self,
        healthy_tomato: dict[str, float],
        feature: str,
        value: float,
        expected_label: str,
    ) -> None:
        sample = {**healthy_tomato, feature: value}
        assert predict_tomato_rule_label(sample) == expected_label
        assert get_tomato_broken_rules(sample) == [expected_label]

    def test_tomato_boundary_values_are_normal(self, healthy_tomato: dict[str, float]) -> None:
        t = TOMATO_THRESHOLDS
        boundary_sample = {
            **healthy_tomato,
            "water_temperature_C": t["water_high"],
            "ec_mS_cm": t["ec_high"],
            "ph": t["ph_low"],
            "relative_humidity_percent": t["hum_high"],
            "nitrogen_mg_L": t["n_low"],
            "phosphorus_mg_L": t["p_low"],
            "potassium_mg_L": t["k_low"],
        }
        assert predict_tomato_rule_label(boundary_sample) == "Normal"
        devs = compute_tomato_deviations(boundary_sample)
        assert all(v <= 0.0 for v in devs.values())

    def test_severity_based_selection(self, healthy_tomato: dict[str, float]) -> None:
        # Water temp dev = 3.29 -> sev = 3.29 / 6.58 = 0.50
        # N dev = 35 -> sev = 35 / 50.0 = 0.70
        # P dev = 4 -> sev = 4 / 20.0 = 0.20
        # N should win as highest severity
        sample = {
            **healthy_tomato,
            "water_temperature_C": 28.29,
            "nitrogen_mg_L": 55.0,
            "phosphorus_mg_L": 31.0,
        }
        assert predict_tomato_rule_label(sample, mode="primary") == "N_Deficiency"
        assert predict_tomato_rule_label(sample, mode="single") == "Multiple_Faults"

    def test_tomato_df_matches_row_by_row(self, healthy_tomato: dict[str, float]) -> None:
        rows = [
            healthy_tomato,
            {**healthy_tomato, "water_temperature_C": 28.0},
            {**healthy_tomato, "nitrogen_mg_L": 50.0},
        ]
        df = pd.DataFrame(rows)
        batch = evaluate_tomato_rules_df(df)
        single = pd.Series([predict_tomato_rule_label(r) for r in rows])
        pd.testing.assert_series_equal(batch, single, check_names=False, check_dtype=False)


class TestUnifiedInterface:
    """Test suite for unified multi-crop interface."""

    def test_predict_crop_rule_label_dispatch(self) -> None:
        lettuce = {"ph": 6.0, "ec": 1.5, "water_temp": 20.0, "humidity": 60.0, "air_temp": 22.0}
        assert predict_crop_rule_label("lettuce", lettuce) == "Normal"
        assert predict_crop_rule_label("LETTUCE", lettuce) == "Normal"

        spinach = {"water_temperature_C": 20.0, "ec_mS_cm": 2.0, "ph": 6.5, "relative_humidity_percent": 60.0}
        assert predict_crop_rule_label("spinach", spinach) == "Normal"

        tomato = {
            "water_temperature_C": 24.0, "ec_mS_cm": 2.5, "ph": 6.2, "relative_humidity_percent": 60.0,
            "nitrogen_mg_L": 120.0, "phosphorus_mg_L": 50.0, "potassium_mg_L": 200.0,
        }
        assert predict_crop_rule_label("tomato", tomato) == "Normal"

    def test_unknown_crop_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown crop 'kale'"):
            predict_crop_rule_label("kale", {})

        with pytest.raises(ValueError, match="Unknown crop 'kale'"):
            evaluate_crop_rules_df(pd.DataFrame(), "kale")

        with pytest.raises(ValueError, match="Unknown crop 'kale'"):
            explain_rule_prediction("kale", {})

    def test_explain_rule_prediction_structure(self) -> None:
        sample = {
            "water_temperature_C": 28.0,
            "ec_mS_cm": 2.5,
            "ph": 6.2,
            "relative_humidity_percent": 60.0,
            "nitrogen_mg_L": 120.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 200.0,
        }
        exp = explain_rule_prediction("tomato", sample)
        assert exp["crop"] == "tomato"
        assert exp["predicted_label"] == "Water_Temp_High"
        assert exp["n_rules_broken"] == 1
        assert exp["broken_rules"] == ["Water_Temp_High"]
        assert "severities" in exp["details"]
        assert exp["details"]["severities"]["Water_Temp_High"] > 0.0

    def test_explain_lettuce_and_spinach(self) -> None:
        lettuce_exp = explain_rule_prediction("lettuce", {"ph": 5.0, "ec": 1.5})
        assert lettuce_exp["crop"] == "lettuce"
        assert lettuce_exp["predicted_label"] == "pH_Low"

        spinach_exp = explain_rule_prediction(
            "spinach",
            {"water_temperature_C": 23.0, "ec_mS_cm": 2.0, "ph": 6.5, "relative_humidity_percent": 60.0},
        )
        assert spinach_exp["crop"] == "spinach"
        assert spinach_exp["predicted_label"] == "Water_Temp_High"
