"""Rule-based reference oracles for hydroponic sensor datasets.

This module provides pure reference functions that implement the domain rules
and labelling heuristics for lettuce, spinach, and tomato crops as defined
in their respective dataset preparation scripts:
- Lettuce: 5 action bands (ph, ec, air_temp, humidity, water_temp).
  A single out-of-band parameter assigns that parameter's stress label.
  Two or more out-of-band parameters assign 'Multiple_Stress'.
  All parameters within band assign 'Normal'.
- Spinach: 6 threshold rules (water_temperature_C, ec_mS_cm, ph, relative_humidity_percent).
  Every non-Normal sample in the prepared dataset breaks exactly one rule.
- Tomato: 9 threshold rules across environmental and nutrient parameters.
  Label is assigned to the most severe fault, where severity is the deviation
  divided by the maximum deviation per rule recorded over the mapped Kolar file.
  The severity scales are sourced from `processed_data/sensor/tomato/manifest.json`.

NOTE: These functions serve strictly as test oracles, baseline comparisons,
and diagnostic explainers; they are NOT a replacement for ML sensor models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# Lettuce Rule Specifications
# Sourced from sensor_data/Lettuce_Sensor-Readings/build_lettuce_dataset
# =============================================================================

LETTUCE_ACTION_BANDS: dict[str, tuple[float, float]] = {
    "ph": (5.5, 7.0),
    "ec": (1.2, 2.5),
    "air_temp": (18.0, 30.0),
    "humidity": (50.0, 80.0),
    "water_temp": (18.0, 26.0),
}

LETTUCE_CLASS_MAP: dict[tuple[str, str], str] = {
    ("ph", "low"): "pH_Low",
    ("ph", "high"): "pH_High",
    ("ec", "low"): "EC_Low",
    ("ec", "high"): "EC_High",
    ("air_temp", "low"): "Temp_Low",
    ("air_temp", "high"): "Temp_High",
    ("humidity", "low"): "Humidity_Low",
    ("humidity", "high"): "Humidity_High",
    ("water_temp", "low"): "WaterTemp_Low",
    ("water_temp", "high"): "WaterTemp_High",
}

# =============================================================================
# Spinach Rule Specifications
# Sourced from sensor_data/Spinach_Sensor-Readings/generate_7class_dataset.py
# =============================================================================

SPINACH_THRESHOLDS: dict[str, Any] = {
    "water_temp_high": 21.0,
    "ec_high": 2.3,
    "ec_low": 1.8,
    "ph_low": 6.0,
    "ph_high": 7.0,
    "hum_high": 75.0,
    "hum_low": 45.0,
}

SPINACH_RULES_ORDER: list[str] = [
    "Water_Temp_High",
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "pH_Imbalance",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
]

# =============================================================================
# Tomato Rule Specifications
# Sourced from sensor_data/Tomato_Sensor-Readings/build_tomato_dataset.py
# and processed_data/sensor/tomato/manifest.json
# =============================================================================

TOMATO_THRESHOLDS: dict[str, float] = {
    "water_high": 25.0,
    "ec_low": 2.0,
    "ec_high": 4.0,
    "ph_low": 6.0,
    "ph_high": 6.5,
    "hum_low": 50.0,
    "hum_high": 85.0,
    "n_low": 90.0,
    "p_low": 35.0,
    "k_low": 144.0,
}

TOMATO_SEVERITY_SCALES: dict[str, float] = {
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

TOMATO_RULES_ORDER: list[str] = [
    "Water_Temp_High",
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "pH_Imbalance",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "N_Deficiency",
    "P_Deficiency",
    "K_Deficiency",
]


# =============================================================================
# Lettuce Reference Oracle
# =============================================================================

def get_lettuce_broken_rules(sample: Mapping[str, Any]) -> list[str]:
    """Return list of stress class names for parameters outside action bands.

    Parameters on boundary values (lo <= value <= hi) are considered in-band.
    NaN or None values are skipped.
    """
    problems: list[str] = []
    for var, (lo, hi) in LETTUCE_ACTION_BANDS.items():
        val = sample.get(var)
        if val is None or pd.isna(val):
            continue
        v = float(val)
        if v < lo:
            problems.append(LETTUCE_CLASS_MAP[(var, "low")])
        elif v > hi:
            problems.append(LETTUCE_CLASS_MAP[(var, "high")])
    return problems


def predict_lettuce_rule_label(sample: Mapping[str, Any]) -> str:
    """Predict lettuce label using the 5 action bands.

    Returns:
        - 'Normal' if no variables are out of bounds.
        - The specific stress class if exactly one variable is out.
        - 'Multiple_Stress' if two or more variables are out.
        - 'Unknown' if no valid features are present.
    """
    valid_count = sum(
        1 for var in LETTUCE_ACTION_BANDS
        if sample.get(var) is not None and not pd.isna(sample.get(var))
    )
    if valid_count == 0:
        return "Unknown"

    problems = get_lettuce_broken_rules(sample)
    if not problems:
        return "Normal"
    if len(problems) == 1:
        return problems[0]
    return "Multiple_Stress"


def evaluate_lettuce_rules_df(df: pd.DataFrame) -> pd.Series:
    """Vectorized / batch rule evaluation for lettuce DataFrame."""
    conds: dict[str, pd.Series] = {}
    for var, (lo, hi) in LETTUCE_ACTION_BANDS.items():
        if var in df.columns:
            s = pd.to_numeric(df[var], errors="coerce")
            conds[LETTUCE_CLASS_MAP[(var, "low")]] = s < lo
            conds[LETTUCE_CLASS_MAP[(var, "high")]] = s > hi

    cond_df = pd.DataFrame(conds, index=df.index).fillna(False)
    n_problems = cond_df.sum(axis=1)

    labels = pd.Series("Normal", index=df.index, dtype=object)
    labels = labels.where(n_problems != 0, "Normal")
    single_mask = n_problems == 1
    if single_mask.any():
        labels.loc[single_mask] = cond_df.loc[single_mask].idxmax(axis=1)
    labels.loc[n_problems > 1] = "Multiple_Stress"
    return labels


# =============================================================================
# Spinach Reference Oracle
# =============================================================================

def get_spinach_broken_rules(sample: Mapping[str, Any]) -> list[str]:
    """Return list of violated rule names for spinach sensor reading.

    Strict inequalities match `sensor_data/Spinach_Sensor-Readings/generate_7class_dataset.py`.
    Boundary values (e.g. wt == 21.0, ec == 2.3, ph == 6.0) do not violate rules.
    """
    wt = sample.get("water_temperature_C")
    ec = sample.get("ec_mS_cm")
    ph = sample.get("ph")
    rh = sample.get("relative_humidity_percent")

    problems: list[str] = []
    if wt is not None and not pd.isna(wt) and float(wt) > SPINACH_THRESHOLDS["water_temp_high"]:
        problems.append("Water_Temp_High")
    if ec is not None and not pd.isna(ec) and float(ec) > SPINACH_THRESHOLDS["ec_high"]:
        problems.append("EC_High_Toxicity")
    if ec is not None and not pd.isna(ec) and float(ec) < SPINACH_THRESHOLDS["ec_low"]:
        problems.append("EC_Low_Deficiency")
    if ph is not None and not pd.isna(ph):
        ph_f = float(ph)
        if ph_f < SPINACH_THRESHOLDS["ph_low"] or ph_f > SPINACH_THRESHOLDS["ph_high"]:
            problems.append("pH_Imbalance")
    if rh is not None and not pd.isna(rh):
        rh_f = float(rh)
        if rh_f > SPINACH_THRESHOLDS["hum_high"]:
            problems.append("Humidity_High_Mildew_Risk")
        elif rh_f < SPINACH_THRESHOLDS["hum_low"]:
            problems.append("Humidity_Low_Stress")

    return problems


def predict_spinach_rule_label(sample: Mapping[str, Any]) -> str:
    """Predict spinach label using the 6 threshold rules.

    Returns:
        - 'Normal' if no rules are violated.
        - Rule class name if exactly one rule is violated.
        - 'Multiple' if more than one rule is violated.
    """
    problems = get_spinach_broken_rules(sample)
    if not problems:
        return "Normal"
    if len(problems) == 1:
        return problems[0]
    return "Multiple"


def evaluate_spinach_rules_df(df: pd.DataFrame) -> pd.Series:
    """Batch rule evaluation for spinach DataFrame matching generate_7class_dataset."""
    v = pd.DataFrame(
        {
            "Water_Temp_High": df["water_temperature_C"] > SPINACH_THRESHOLDS["water_temp_high"],
            "EC_High_Toxicity": df["ec_mS_cm"] > SPINACH_THRESHOLDS["ec_high"],
            "EC_Low_Deficiency": df["ec_mS_cm"] < SPINACH_THRESHOLDS["ec_low"],
            "pH_Imbalance": (df["ph"] < SPINACH_THRESHOLDS["ph_low"])
            | (df["ph"] > SPINACH_THRESHOLDS["ph_high"]),
            "Humidity_High_Mildew_Risk": (
                df["relative_humidity_percent"] > SPINACH_THRESHOLDS["hum_high"]
            ),
            "Humidity_Low_Stress": (
                df["relative_humidity_percent"] < SPINACH_THRESHOLDS["hum_low"]
            ),
        },
        index=df.index,
    )
    n = v.sum(axis=1)
    res = np.where(n == 0, "Normal", np.where(n == 1, v.idxmax(axis=1), "Multiple"))
    return pd.Series(res, index=df.index)


# =============================================================================
# Tomato Reference Oracle
# =============================================================================

def compute_tomato_deviations(sample: Mapping[str, Any]) -> dict[str, float]:
    """Calculate positive deviations past healthy limits for tomato.

    Values exactly at or within thresholds yield zero or negative deviation.
    """
    wt = float(sample.get("water_temperature_C", 0.0))
    ec = float(sample.get("ec_mS_cm", 0.0))
    ph = float(sample.get("ph", 0.0))
    rh = float(sample.get("relative_humidity_percent", 0.0))
    n = float(sample.get("nitrogen_mg_L", 0.0))
    p = float(sample.get("phosphorus_mg_L", 0.0))
    k = float(sample.get("potassium_mg_L", 0.0))

    t = TOMATO_THRESHOLDS
    return {
        "Water_Temp_High": wt - t["water_high"],
        "EC_High_Toxicity": ec - t["ec_high"],
        "EC_Low_Deficiency": t["ec_low"] - ec,
        "pH_Imbalance": max(t["ph_low"] - ph, ph - t["ph_high"]),
        "Humidity_High_Mildew_Risk": rh - t["hum_high"],
        "Humidity_Low_Stress": t["hum_low"] - rh,
        "N_Deficiency": t["n_low"] - n,
        "P_Deficiency": t["p_low"] - p,
        "K_Deficiency": t["k_low"] - k,
    }


def compute_tomato_severities(sample: Mapping[str, Any]) -> dict[str, float]:
    """Calculate normalized severity scores (0 to 1+) for all 9 tomato rules.

    Severity = deviation.clip(lower=0) / severity_scale.
    """
    devs = compute_tomato_deviations(sample)
    return {
        rule: max(0.0, devs[rule]) / TOMATO_SEVERITY_SCALES[rule]
        for rule in TOMATO_RULES_ORDER
    }


def get_tomato_broken_rules(sample: Mapping[str, Any]) -> list[str]:
    """Return list of rules broken with positive deviation (> 0.0)."""
    devs = compute_tomato_deviations(sample)
    return [rule for rule in TOMATO_RULES_ORDER if devs[rule] > 0.0]


def predict_tomato_rule_label(
    sample: Mapping[str, Any],
    *,
    mode: str = "primary",
) -> str:
    """Predict tomato label from sensor values.

    Args:
        sample: Mapping containing tomato sensor features.
        mode:
            - 'primary' (default): Assign label to rule with highest normalized severity
              (deviation / scale). Zero rules broken yields 'Normal'.
            - 'single': Only rows breaking exactly 1 rule retain that class;
              rows breaking 2+ rules return 'Multiple_Faults'.

    Tie-breaking:
        In case of identical maximum severity, rule order is preserved:
        Water_Temp_High > EC_High_Toxicity > EC_Low_Deficiency > pH_Imbalance > ...
    """
    broken = get_tomato_broken_rules(sample)
    if not broken:
        return "Normal"

    if mode == "single":
        if len(broken) == 1:
            return broken[0]
        return "Multiple_Faults"

    severities = compute_tomato_severities(sample)
    # Pick rule with largest severity; in case of ties, earliest in TOMATO_RULES_ORDER wins.
    return max(TOMATO_RULES_ORDER, key=lambda r: severities[r])


def evaluate_tomato_rules_df(
    df: pd.DataFrame,
    *,
    mode: str = "primary",
    scales: dict[str, float] | None = None,
) -> pd.Series:
    """Batch rule evaluation for tomato DataFrame matching build_tomato_dataset.py.

    Args:
        df: DataFrame containing the 8 tomato features.
        mode: 'primary' (default) or 'single'.
        scales: Optional custom severity scale dict; defaults to TOMATO_SEVERITY_SCALES.
    """
    t = TOMATO_THRESHOLDS
    active_scales = pd.Series(scales or TOMATO_SEVERITY_SCALES)

    dev = pd.DataFrame(
        {
            "Water_Temp_High": df["water_temperature_C"] - t["water_high"],
            "EC_High_Toxicity": df["ec_mS_cm"] - t["ec_high"],
            "EC_Low_Deficiency": t["ec_low"] - df["ec_mS_cm"],
            "pH_Imbalance": np.maximum(t["ph_low"] - df["ph"], df["ph"] - t["ph_high"]),
            "Humidity_High_Mildew_Risk": df["relative_humidity_percent"] - t["hum_high"],
            "Humidity_Low_Stress": t["hum_low"] - df["relative_humidity_percent"],
            "N_Deficiency": t["n_low"] - df["nitrogen_mg_L"],
            "P_Deficiency": t["p_low"] - df["phosphorus_mg_L"],
            "K_Deficiency": t["k_low"] - df["potassium_mg_L"],
        },
        index=df.index,
    )
    viol = dev > 0
    n = viol.sum(axis=1)

    if mode == "single":
        multi = np.where(n == 1, viol.idxmax(axis=1), "Multiple_Faults")
        lab = np.where(n == 0, "Normal", multi)
    else:
        severity = dev.clip(lower=0) / active_scales
        lab = np.where(n == 0, "Normal", severity.idxmax(axis=1))

    return pd.Series(lab, index=df.index)


# =============================================================================
# Unified Multi-Crop Interface
# =============================================================================

def predict_crop_rule_label(crop: str, sample: Mapping[str, Any]) -> str:
    """Dispatch single-sample rule label prediction for any supported crop."""
    c = crop.lower().strip()
    if c == "lettuce":
        return predict_lettuce_rule_label(sample)
    if c == "spinach":
        return predict_spinach_rule_label(sample)
    if c == "tomato":
        return predict_tomato_rule_label(sample)
    raise ValueError(f"Unknown crop '{crop}'. Expected 'lettuce', 'spinach', or 'tomato'.")


def evaluate_crop_rules_df(df: pd.DataFrame, crop: str) -> pd.Series:
    """Dispatch batch DataFrame rule evaluation for any supported crop."""
    c = crop.lower().strip()
    if c == "lettuce":
        return evaluate_lettuce_rules_df(df)
    if c == "spinach":
        return evaluate_spinach_rules_df(df)
    if c == "tomato":
        return evaluate_tomato_rules_df(df)
    raise ValueError(f"Unknown crop '{crop}'. Expected 'lettuce', 'spinach', or 'tomato'.")


def explain_rule_prediction(crop: str, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Provide a structured explanation of the rule-based prediction.

    Returns a dictionary containing:
        - crop: The crop name.
        - predicted_label: The rule-assigned label.
        - n_rules_broken: Count of rules/bands violated.
        - broken_rules: List of violated rule names.
        - details: Crop-specific diagnostics (e.g. severities or band checks).
    """
    c = crop.lower().strip()
    if c == "lettuce":
        broken = get_lettuce_broken_rules(sample)
        pred = predict_lettuce_rule_label(sample)
        details = {
            var: {
                "value": sample.get(var),
                "healthy_band": LETTUCE_ACTION_BANDS[var],
                "in_band": (
                    sample.get(var) is not None
                    and LETTUCE_ACTION_BANDS[var][0] <= float(sample[var]) <= LETTUCE_ACTION_BANDS[var][1]
                ),
            }
            for var in LETTUCE_ACTION_BANDS
        }
        return {
            "crop": "lettuce",
            "predicted_label": pred,
            "n_rules_broken": len(broken),
            "broken_rules": broken,
            "details": details,
        }
    if c == "spinach":
        broken = get_spinach_broken_rules(sample)
        pred = predict_spinach_rule_label(sample)
        return {
            "crop": "spinach",
            "predicted_label": pred,
            "n_rules_broken": len(broken),
            "broken_rules": broken,
            "details": {"thresholds": SPINACH_THRESHOLDS},
        }
    if c == "tomato":
        broken = get_tomato_broken_rules(sample)
        pred = predict_tomato_rule_label(sample)
        severities = compute_tomato_severities(sample)
        return {
            "crop": "tomato",
            "predicted_label": pred,
            "n_rules_broken": len(broken),
            "broken_rules": broken,
            "details": {
                "severities": severities,
                "scales": TOMATO_SEVERITY_SCALES,
                "thresholds": TOMATO_THRESHOLDS,
            },
        }
    raise ValueError(f"Unknown crop '{crop}'. Expected 'lettuce', 'spinach', or 'tomato'.")


__all__ = [
    "LETTUCE_ACTION_BANDS",
    "LETTUCE_CLASS_MAP",
    "SPINACH_RULES_ORDER",
    "SPINACH_THRESHOLDS",
    "TOMATO_RULES_ORDER",
    "TOMATO_SEVERITY_SCALES",
    "TOMATO_THRESHOLDS",
    "compute_tomato_deviations",
    "compute_tomato_severities",
    "evaluate_crop_rules_df",
    "evaluate_lettuce_rules_df",
    "evaluate_spinach_rules_df",
    "evaluate_tomato_rules_df",
    "explain_rule_prediction",
    "get_lettuce_broken_rules",
    "get_spinach_broken_rules",
    "get_tomato_broken_rules",
    "predict_crop_rule_label",
    "predict_lettuce_rule_label",
    "predict_spinach_rule_label",
    "predict_tomato_rule_label",
]
