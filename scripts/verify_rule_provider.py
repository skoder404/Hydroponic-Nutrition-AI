"""Verification script for RuleProvider and recommendation rules matrix (T6-008).

Acceptance Criteria:
1. Every class of every crop has a rule or an explicit 'insufficient information'.
2. No rule text contains dosage numbers.
3. Output cites which prediction/rule produced it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from ml.labels import (
    LETTUCE_IMAGE_CLASSES,
    LETTUCE_SENSOR_CLASSES,
    SPINACH_IMAGE_CLASSES,
    SPINACH_SENSOR_CLASSES,
    TOMATO_IMAGE_CLASSES,
    TOMATO_SENSOR_CLASSES,
)
from ml.recommendations.rule_provider import get_rule_provider, reset_rule_provider


def verify_criterion_1_all_classes_covered() -> None:
    """Verify that every single class of every crop has a valid rule."""
    print("[1/3] Testing Criterion 1: Every class of every crop has a rule or explicit 'insufficient information'...")
    provider = get_rule_provider()

    crop_classes = {
        "lettuce": (LETTUCE_IMAGE_CLASSES, LETTUCE_SENSOR_CLASSES),
        "spinach": (SPINACH_IMAGE_CLASSES, SPINACH_SENSOR_CLASSES),
        "tomato": (TOMATO_IMAGE_CLASSES, TOMATO_SENSOR_CLASSES),
    }

    total_classes = 0
    for crop, (img_classes, sensor_classes) in crop_classes.items():
        for cls in img_classes:
            rec = provider.recommend(crop, cls, "image")
            if "insufficient information" in rec.content["summary"].lower():
                raise RuntimeError(f"Missing specific rule for {crop}/image/{cls}")
            total_classes += 1

        for cls in sensor_classes:
            rec = provider.recommend(crop, cls, "sensor")
            if "insufficient information" in rec.content["summary"].lower():
                raise RuntimeError(f"Missing specific rule for {crop}/sensor/{cls}")
            total_classes += 1

    # Verify unsupported combinations return explicit insufficient information
    rec_unsupported = provider.recommend("lettuce", "NonexistentClass", "sensor")
    if "insufficient information" not in rec_unsupported.content["summary"].lower():
        raise RuntimeError("Expected 'insufficient information' for unsupported class, but got valid rule.")
    if rec_unsupported.evidence.get("rule_citation") != "insufficient_information":
        raise RuntimeError(f"Expected citation 'insufficient_information', got {rec_unsupported.evidence.get('rule_citation')}")

    print(f"  -> All {total_classes} classes across lettuce, spinach, and tomato verified. Unsupported class correctly returns 'insufficient information'.")


def verify_criterion_2_no_dosage_numbers() -> None:
    """Verify that no action text or rule description contains numeric dosages."""
    print("[2/3] Testing Criterion 2: No rule text contains dosage numbers...")
    rules_dir = REPO_ROOT / "ml" / "recommendations" / "rules"
    yaml_files = list(rules_dir.glob("*.v1.yaml"))
    if len(yaml_files) < 3:
        raise RuntimeError(f"Expected at least 3 YAML rule files, found {len(yaml_files)}")

    digit_unit_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ml|mg|g|ppm|%|ms|l|liters?|drops?|oz)\b", re.IGNORECASE)

    for yf in yaml_files:
        data = yaml.safe_load(yf.read_text(encoding="utf-8"))
        rules = data.get("rules", {})
        for modality, class_dict in rules.items():
            for cls_name, rule in class_dict.items():
                for field in ("actions", "low_confidence_action"):
                    items = rule.get(field, [])
                    items = items if isinstance(items, list) else [items]
                    for item in items:
                        # Check for dosage units with numbers
                        if digit_unit_pattern.search(str(item)):
                            raise RuntimeError(f"Dosage unit found in {yf.name} -> {modality}/{cls_name} -> {field}: {item}")
                        # Check for any raw digits in actionable instructions
                        if re.findall(r"\d+", str(item)):
                            raise RuntimeError(f"Dosage number found in {yf.name} -> {modality}/{cls_name} -> {field}: {item}")

    print("  -> Verified: Zero dosage numbers found across all action items in all rule sets.")


def verify_criterion_3_citation() -> None:
    """Verify output cites which prediction/rule produced it."""
    print("[3/3] Testing Criterion 3: Output cites which prediction/rule produced it...")
    provider = get_rule_provider()

    rec = provider.recommend("spinach", "EC_High_Toxicity", "sensor", confidence=0.85)

    citation = rec.evidence.get("rule_citation")
    expected_citation = "rules/spinach.v1.yaml#sensor/EC_High_Toxicity"
    if citation != expected_citation:
        raise RuntimeError(f"Expected rule citation '{expected_citation}', got '{citation}'")

    if rec.evidence.get("crop") != "spinach":
        raise RuntimeError("Evidence missing crop citation")
    if rec.evidence.get("predicted_class") != "EC_High_Toxicity":
        raise RuntimeError("Evidence missing predicted_class citation")
    if rec.evidence.get("confidence") != 0.85:
        raise RuntimeError("Evidence missing confidence citation")
    if rec.evidence.get("rules_version") != "v1.0.0":
        raise RuntimeError("Evidence missing rules_version citation")

    print(f"  -> Rule citation verified: '{citation}', rules_version='{rec.rules_version}'.")


def main() -> None:
    """Run all verification checks for T6-008."""
    print("================================================================================")
    print("Verifying T6-008: Rule-Based Recommendation Provider")
    print("================================================================================")
    try:
        reset_rule_provider()
        verify_criterion_1_all_classes_covered()
        verify_criterion_2_no_dosage_numbers()
        verify_criterion_3_citation()
        print("\nSUCCESS: All T6-008 acceptance criteria verified successfully.")
    except Exception as exc:
        print(f"\nFAILURE: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
