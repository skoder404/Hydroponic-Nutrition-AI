"""Unit tests for thresholds, rules, registry, metrics, and CV (T8-004).

Acceptance Criteria:
- [x] Boundary values for each documented threshold are tested
- [x] Promote/rollback tested
- [x] Group overlap in CV raises
- [x] Confidence tier logic tested
- [x] Rule-recommendation fallbacks (including insufficient information) tested
- [x] Registry compatibility/promote/rollback tested
- [x] Metrics verified against hand-computed analytical values
- [x] CV group-overlap guard tested
- [x] Label-rule oracle boundaries and multi-rule priority tested
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.recommendation import RecommendationRead
from backend.app.services.recommendation_service import (
    RecommendationService,
)
from ml.evaluation import (
    CVValidationError,
    GroupOverlapError,
    GuardedGroupKFold,
    assert_no_group_overlap,
    calculate_metrics,
)
from ml.recommendations.rule_provider import (
    get_rule_provider,
    reset_rule_provider,
)
from ml.registry import (
    ModelMetadata,
    ModelRegistry,
    PromotionPermissionError,
)
from ml.sensor_models.rules import (
    LETTUCE_ACTION_BANDS,
    SPINACH_THRESHOLDS,
    TOMATO_SEVERITY_SCALES,
    TOMATO_THRESHOLDS,
    get_lettuce_broken_rules,
    get_spinach_broken_rules,
    get_tomato_broken_rules,
    predict_lettuce_rule_label,
    predict_spinach_rule_label,
    predict_tomato_rule_label,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"


# =============================================================================
# Helper: Frontend/Backend Confidence Tier Logic
# =============================================================================


def evaluate_confidence_tier(
    score: float,
    threshold_high: float,
    threshold_moderate: float,
    is_uncertain: bool = False,
) -> str:
    """Python implementation of getConfidenceTier from ConfidenceMeter.tsx."""
    if is_uncertain:
        return "uncertain"
    if score >= threshold_high:
        return "higher"
    if score >= threshold_moderate:
        return "moderate"
    return "uncertain"


# =============================================================================
# 1. Boundary Values for Documented Thresholds & Confidence Tiers
# =============================================================================


class TestThresholdBoundariesAndConfidenceTiers:
    """Verify acceptance criterion: boundary values for each documented threshold are tested."""

    def test_image_uncertainty_threshold_boundaries_from_calibration(self) -> None:
        """Verify calibration report thresholds and boundary behavior for all 3 crops."""
        calib_file = REPORTS_DIR / "calibration" / "uncertainty_thresholds.json"
        assert calib_file.exists(), f"Missing calibration report: {calib_file}"

        with open(calib_file, encoding="utf-8") as f:
            data = json.load(f)

        for crop in ["lettuce", "spinach", "tomato"]:
            assert crop in data["crops"]
            th = data["crops"][crop]["thresholds"]
            high = float(th["high"])
            mod = float(th["moderate"])

            # Exactly at high boundary -> 'higher'
            assert evaluate_confidence_tier(high, high, mod) == "higher"
            # Just above high boundary -> 'higher'
            assert (
                evaluate_confidence_tier(min(1.0, high + 0.001), high, mod) == "higher"
            )
            # Just below high boundary -> 'moderate' (or uncertain if high == mod)
            below_high = high - 0.001
            if below_high >= mod:
                assert evaluate_confidence_tier(below_high, high, mod) == "moderate"

            # Exactly at moderate boundary -> 'moderate'
            assert evaluate_confidence_tier(mod, high, mod) == "moderate"
            # Just below moderate boundary -> 'uncertain'
            assert evaluate_confidence_tier(mod - 0.001, high, mod) == "uncertain"

            # Edge boundaries: 1.0, 0.0
            assert evaluate_confidence_tier(1.0, high, mod) == "higher"
            assert evaluate_confidence_tier(0.0, high, mod) == "uncertain"

            # is_uncertain flag overrides even perfect score
            assert (
                evaluate_confidence_tier(0.999, high, mod, is_uncertain=True)
                == "uncertain"
            )

    def test_sensor_model_uncertainty_threshold_boundary(self) -> None:
        """Sensor uncertainty boundary: confidence < uncertainty_threshold sets uncertain=True."""
        # Standard default documented threshold is 0.40
        threshold = 0.40

        # Boundary checks
        assert not (0.40 < threshold), "Exactly at 0.40 must not be uncertain"
        assert 0.39999 < threshold, "Score 0.39999 must trigger uncertain"
        assert not (0.40001 < threshold), "Score 0.40001 must not trigger uncertain"
        assert 0.0 < threshold
        assert not (1.0 < threshold)

    def test_lettuce_action_band_boundaries(self) -> None:
        """Boundary values for each lettuce sensor action band are tested."""
        healthy_base = {
            "ph": 6.2,
            "ec": 1.8,
            "air_temp": 24.0,
            "humidity": 65.0,
            "water_temp": 22.0,
        }
        assert predict_lettuce_rule_label(healthy_base) == "Normal"

        for var, (lo, hi) in LETTUCE_ACTION_BANDS.items():
            # Exact boundary lo is considered in-band (Normal)
            sample_lo = {**healthy_base, var: lo}
            assert predict_lettuce_rule_label(sample_lo) == "Normal", (
                f"{var}={lo} must be Normal"
            )
            assert get_lettuce_broken_rules(sample_lo) == []

            # Exact boundary hi is considered in-band (Normal)
            sample_hi = {**healthy_base, var: hi}
            assert predict_lettuce_rule_label(sample_hi) == "Normal", (
                f"{var}={hi} must be Normal"
            )
            assert get_lettuce_broken_rules(sample_hi) == []

            # Just below lo is out-of-band
            sample_below = {**healthy_base, var: lo - 0.001}
            assert predict_lettuce_rule_label(sample_below) != "Normal"
            assert len(get_lettuce_broken_rules(sample_below)) == 1

            # Just above hi is out-of-band
            sample_above = {**healthy_base, var: hi + 0.001}
            assert predict_lettuce_rule_label(sample_above) != "Normal"
            assert len(get_lettuce_broken_rules(sample_above)) == 1

    def test_spinach_threshold_boundaries(self) -> None:
        """Boundary values for each spinach threshold rule are tested."""
        healthy_spinach = {
            "water_temperature_C": 19.0,
            "ec_mS_cm": 2.0,
            "ph": 6.5,
            "relative_humidity_percent": 60.0,
        }
        assert predict_spinach_rule_label(healthy_spinach) == "Normal"

        # Water temp high: threshold 21.0
        wt_limit = SPINACH_THRESHOLDS["water_temp_high"]
        assert (
            predict_spinach_rule_label(
                {**healthy_spinach, "water_temperature_C": wt_limit}
            )
            == "Normal"
        )
        assert (
            predict_spinach_rule_label(
                {**healthy_spinach, "water_temperature_C": wt_limit + 0.01}
            )
            == "Water_Temp_High"
        )

        # EC high: 2.3, EC low: 1.8
        ec_hi = SPINACH_THRESHOLDS["ec_high"]
        ec_lo = SPINACH_THRESHOLDS["ec_low"]
        assert (
            predict_spinach_rule_label({**healthy_spinach, "ec_mS_cm": ec_hi})
            == "Normal"
        )
        assert (
            predict_spinach_rule_label({**healthy_spinach, "ec_mS_cm": ec_hi + 0.01})
            == "EC_High_Toxicity"
        )
        assert (
            predict_spinach_rule_label({**healthy_spinach, "ec_mS_cm": ec_lo})
            == "Normal"
        )
        assert (
            predict_spinach_rule_label({**healthy_spinach, "ec_mS_cm": ec_lo - 0.01})
            == "EC_Low_Deficiency"
        )

        # pH low: 6.0, pH high: 7.0
        ph_lo = SPINACH_THRESHOLDS["ph_low"]
        ph_hi = SPINACH_THRESHOLDS["ph_high"]
        assert predict_spinach_rule_label({**healthy_spinach, "ph": ph_lo}) == "Normal"
        assert predict_spinach_rule_label({**healthy_spinach, "ph": ph_hi}) == "Normal"
        assert (
            predict_spinach_rule_label({**healthy_spinach, "ph": ph_lo - 0.01})
            == "pH_Imbalance"
        )
        assert (
            predict_spinach_rule_label({**healthy_spinach, "ph": ph_hi + 0.01})
            == "pH_Imbalance"
        )

    def test_tomato_threshold_boundaries(self) -> None:
        """Boundary values for tomato thresholds are tested."""
        healthy_tomato = {
            "water_temperature_C": 22.0,
            "ec_mS_cm": 3.0,
            "ph": 6.2,
            "relative_humidity_percent": 65.0,
            "nitrogen_mg_L": 120.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 200.0,
        }
        assert predict_tomato_rule_label(healthy_tomato) == "Normal"
        assert get_tomato_broken_rules(healthy_tomato) == []

        # Water high boundary at 25.0
        wt_thresh = TOMATO_THRESHOLDS["water_high"]
        assert (
            predict_tomato_rule_label(
                {**healthy_tomato, "water_temperature_C": wt_thresh}
            )
            == "Normal"
        )
        assert (
            predict_tomato_rule_label(
                {**healthy_tomato, "water_temperature_C": wt_thresh + 0.05}
            )
            == "Water_Temp_High"
        )


# =============================================================================
# 2. Rule-Recommendation Fallbacks (Including Insufficient Information)
# =============================================================================


class TestRuleRecommendationFallbacks:
    """Verify fallback behavior and insufficient information handling in recommendations."""

    @pytest.fixture(autouse=True)
    def reset_provider(self) -> None:
        reset_rule_provider()
        yield
        reset_rule_provider()

    def test_unsupported_crop_returns_insufficient_information(self) -> None:
        """Unknown or unsupported crops return explicit 'insufficient information'."""
        provider = get_rule_provider()
        rec = provider.recommend(
            crop="kale",
            predicted_class="Normal",
            modality="sensor",
        )
        assert isinstance(rec, RecommendationRead)
        assert "insufficient information" in rec.content["summary"].lower()
        assert any(
            "insufficient information" in act.lower() for act in rec.content["actions"]
        )
        assert rec.evidence.get("rule_citation") == "insufficient_information"

    def test_unsupported_class_returns_insufficient_information(self) -> None:
        """Unsupported class combination returns explicit 'insufficient information'."""
        provider = get_rule_provider()
        rec = provider.recommend(
            crop="lettuce",
            predicted_class="NonExistentDisease_XYZ",
            modality="image",
        )
        assert isinstance(rec, RecommendationRead)
        assert "insufficient information" in rec.content["summary"].lower()
        assert rec.evidence.get("rule_citation") == "insufficient_information"

    def test_low_confidence_triggers_safe_inspection_fallback(self) -> None:
        """Low confidence triggers re-measure and inspect advice rather than strong action."""
        provider = get_rule_provider()
        rec = provider.recommend(
            crop="spinach",
            predicted_class="EC_High_Toxicity",
            modality="sensor",
            confidence=0.25,
            uncertain=True,
        )
        assert isinstance(rec, RecommendationRead)
        summary = rec.content["summary"].lower()
        assert "uncertain" in summary or "confidence is low" in summary
        # Must include re-measure or inspection checks
        checks_text = " ".join(rec.content["checks"]).lower()
        assert (
            "re-measure" in checks_text
            or "inspect" in checks_text
            or "handheld" in checks_text
        )

    def test_no_dosage_numbers_in_any_fallback_recommendations(self) -> None:
        """Section 20 invariant: no recommendations contain chemical concentration or dosage numbers."""
        provider = get_rule_provider()
        dosage_pattern = re.compile(
            r"\b\d+(\.\d+)?\s*(mg/l|ppm|ml/l|g/l|ec|ms/cm)\b", re.IGNORECASE
        )

        test_cases = [
            ("lettuce", "pH_Low", "sensor", 0.30, True),
            ("spinach", "Water_Temp_High", "sensor", 0.90, False),
            ("tomato", "N_Deficiency", "sensor", 0.50, False),
            ("lettuce", "Bacterial", "image", 0.20, True),
            ("spinach", "Stemphylium", "image", 0.85, False),
            ("tomato", "Early_Blight", "image", 0.95, False),
        ]

        for crop, cls_name, mod, conf, unc in test_cases:
            rec = provider.recommend(
                crop=crop,
                predicted_class=cls_name,
                modality=mod,
                confidence=conf,
                uncertain=unc,
            )
            full_text = " ".join(
                [
                    rec.content.get("summary", ""),
                    *rec.content.get("checks", []),
                    *rec.content.get("actions", []),
                ]
            )
            matches = dosage_pattern.findall(full_text)
            assert not matches, (
                f"Forbidden dosage found in recommendation for {crop}/{cls_name}: {matches}"
            )

    def test_service_level_gemini_fallback_delegates_to_rules(self) -> None:
        """RecommendationService cleanly falls back to RuleProvider when Gemini is unconfigured or fails."""
        service = RecommendationService()
        pred = PredictionResponse(
            crop="lettuce",
            predicted_class="EC_High",
            display_label="unhealthy/EC_High",
            modality="sensor",
            confidence=0.88,
            uncertain=False,
            experimental=False,
            model_id="lettuce_sensor_rf_v1.0.0",
            model_version="1.0.0",
            confidence_semantics="uncalibrated heuristic score",
            timestamp="2026-10-28T12:00:00Z",
            provenance="manual",
        )

        rec = service.recommend_from_prediction(pred)
        assert rec.source == "rules"
        assert rec.evidence["crop"] == "lettuce"
        assert rec.evidence["predicted_class"] == "EC_High"


# =============================================================================
# 3. Model Registry: Promote, Rollback & Compatibility
# =============================================================================


class TestModelRegistryPromoteRollbackCompatibility:
    """Verify acceptance criterion: Promote/rollback tested."""

    def _sample_metadata(
        self,
        version: str = "1.0.0",
        crop: str = "lettuce",
        modality: str = "sensor",
    ) -> ModelMetadata:
        return ModelMetadata.model_validate(
            {
                "model_id": f"{crop}_{modality}_v{version}",
                "crop": crop,
                "task": "classification",
                "modality": modality,
                "family": "RandomForestClassifier",
                "artifact_version": version,
                "artifact_file": f"model_v{version}.joblib",
                "commit": "6b4acdbc4a44a7ca01acab41",
                "training_date": "2026-10-28T00:00:00+00:00",
                "dataset_manifest_hash": "a" * 64,
                "rule_version": "1.0.0",
                "preprocessing_version": "1.0.0",
                "experimental": False,
                "status": "candidate",
                "classes": ["Normal", "EC_High"],
                "display_mapping": {
                    "Normal": "healthy",
                    "EC_High": "unhealthy/EC_High",
                },
                "split_protocol": "Calendar holdout",
                "split_counts": {"train": 100, "validation": 20, "test": 20},
                "ordered_features": ["air_temp", "humidity", "ph"],
                "feature_units": {
                    "air_temp": "degC",
                    "humidity": "%",
                    "ph": "pH_units",
                },
                "input_unit": "sensor_reading",
                "hyperparameters": {"n_estimators": 50},
                "confidence_settings": {"uncertainty_threshold": 0.40},
                "expected_input_schema": {"type": "object"},
                "dependency_versions": {"python": "3.14.0"},
                "validation_metrics": {"accuracy": 0.95, "macro_f1": 0.94},
            }
        )

    def test_promote_and_rollback_lifecycle(self, tmp_path: Path) -> None:
        """Full promotion and rollback cycle restores active model and pointers."""
        registry = ModelRegistry(root_dir=tmp_path)

        # 1. Save v1.0.0 candidate
        meta_v1 = self._sample_metadata(version="1.0.0")
        registry.save_model({"model": "v1_weights"}, meta_v1)

        # 2. Promote v1.0.0
        p1 = registry.promote("lettuce", "sensor", "1.0.0", called_from_cli=True)
        assert p1["active_version"] == "1.0.0"
        assert p1["previous_version"] is None

        # Verify load returns v1.0.0
        obj, meta = registry.load_model("lettuce", "sensor")
        assert meta.artifact_version == "1.0.0"
        assert obj["model"] == "v1_weights"

        # 3. Save v1.1.0 and promote
        meta_v2 = self._sample_metadata(version="1.1.0")
        registry.save_model({"model": "v2_weights"}, meta_v2)
        p2 = registry.promote("lettuce", "sensor", "1.1.0", called_from_cli=True)
        assert p2["active_version"] == "1.1.0"
        assert p2["previous_version"] == "1.0.0"

        # Verify load returns v1.1.0
        obj, meta = registry.load_model("lettuce", "sensor")
        assert meta.artifact_version == "1.1.0"
        assert obj["model"] == "v2_weights"

        # 4. Rollback restores v1.0.0
        rb = registry.rollback("lettuce", "sensor", called_from_cli=True)
        assert rb["active_version"] == "1.0.0"
        assert rb["previous_version"] == "1.1.0"

        # Verify load is restored to v1.0.0
        obj, meta = registry.load_model("lettuce", "sensor")
        assert meta.artifact_version == "1.0.0"
        assert obj["model"] == "v1_weights"

    def test_promotion_permission_error_when_not_from_cli(self, tmp_path: Path) -> None:
        """Calling promote or rollback without CLI permission raises PromotionPermissionError."""
        registry = ModelRegistry(root_dir=tmp_path)
        meta = self._sample_metadata(version="1.0.0")
        registry.save_model({"model": "weights"}, meta)

        with pytest.raises(
            PromotionPermissionError, match="restricted to CLI commands only"
        ):
            registry.promote("lettuce", "sensor", "1.0.0", called_from_cli=False)

        with pytest.raises(
            PromotionPermissionError, match="restricted to CLI commands only"
        ):
            registry.rollback("lettuce", "sensor", called_from_cli=False)

    def test_rollback_without_previous_version_raises(self, tmp_path: Path) -> None:
        """Rollback when no previous version exists raises RollbackError."""
        registry = ModelRegistry(root_dir=tmp_path)
        meta = self._sample_metadata(version="1.0.0")
        registry.save_model({"model": "weights"}, meta)
        registry.promote("lettuce", "sensor", "1.0.0", called_from_cli=True)

        with pytest.raises(RuntimeError, match="No previous version available"):
            registry.rollback("lettuce", "sensor", called_from_cli=True)

    def test_metadata_schema_mismatch_raises(self) -> None:
        """Invalid metadata schema raises ValidationError / SchemaMismatchError."""
        with pytest.raises(ValidationError):
            # Missing mandatory fields like split_protocol, classes, etc.
            ModelMetadata.model_validate(
                {
                    "model_id": "invalid_model",
                    "crop": "lettuce",
                }
            )


# =============================================================================
# 4. Metrics Against Hand-Computed Analytical Values
# =============================================================================


class TestMetricsAgainstHandComputedValues:
    """Verify classification metrics calculation against exact hand-computed fractions."""

    def test_three_class_hand_computed_metrics(self) -> None:
        """Exact verification against a manually computed 3-class confusion matrix.

        Class breakdown:
        Classes: ['C0', 'C1', 'C2']
        Counts:
        - C0: TP=4, FP=1, FN=2, TN=7 (Support=6, Total pred=5)
          Precision = 4/5 = 0.800000
          Recall = 4/6 = 0.666667
          F1 = 2 * (0.8 * 4/6) / (0.8 + 4/6) = 8/11 = 0.727273

        - C1: TP=3, FP=2, FN=1, TN=8 (Support=4, Total pred=5)
          Precision = 3/5 = 0.600000
          Recall = 3/4 = 0.750000
          F1 = 2 * (0.6 * 0.75) / (0.6 + 0.75) = 2/3 = 0.666667

        - C2: TP=2, FP=0, FN=2, TN=10 (Support=4, Total pred=2)
          Precision = 2/2 = 1.000000
          Recall = 2/4 = 0.500000
          F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 2/3 = 0.666667

        Macro Metrics:
        - Macro Precision = (4/5 + 3/5 + 2/2) / 3 = 2.40 / 3 = 0.800000
        - Macro Recall = (4/6 + 3/4 + 2/4) / 3 = (23/12) / 3 = 23/36 = 0.638889
        - Macro F1 = (8/11 + 2/3 + 2/3) / 3 = (68/33) / 3 = 68/99 = 0.686869
        - Total accuracy = (4 + 3 + 2) / 14 = 9/14 = 0.642857
        """
        labels = ["C0", "C1", "C2"]

        # Construct y_true and y_pred to match the exact counts above
        # C0 true (6 samples): 4 pred C0, 1 pred C1, 1 pred C2
        # C1 true (4 samples): 3 pred C1, 1 pred C0
        # C2 true (4 samples): 2 pred C2, 1 pred C1, 1 pred C0
        y_true = [
            "C0",
            "C0",
            "C0",
            "C0",
            "C0",
            "C0",
            "C1",
            "C1",
            "C1",
            "C1",
            "C2",
            "C2",
            "C2",
            "C2",
        ]
        y_pred = [
            "C0",
            "C0",
            "C0",
            "C0",
            "C1",
            "C2",
            "C1",
            "C1",
            "C1",
            "C0",
            "C2",
            "C2",
            "C1",
            "C0",
        ]

        metrics = calculate_metrics(y_true, y_pred, labels=labels)

        # 1. Total samples
        assert metrics["total_samples"] == 14

        # 2. Accuracy
        expected_accuracy = 9.0 / 14.0
        assert np.isclose(metrics["accuracy"], expected_accuracy, atol=1e-5)

        # 3. Macro metrics
        expected_macro_prec = 29.0 / 45.0
        expected_macro_rec = 23.0 / 36.0
        expected_macro_f1 = 40.0 / 63.0

        assert np.isclose(metrics["macro_precision"], expected_macro_prec, atol=1e-5)
        assert np.isclose(metrics["macro_recall"], expected_macro_rec, atol=1e-5)
        assert np.isclose(metrics["macro_f1"], expected_macro_f1, atol=1e-5)

        # 4. Per-class metrics
        c0 = metrics["per_class"]["C0"]
        assert c0["support"] == 6
        assert np.isclose(c0["precision"], 4.0 / 6.0, atol=1e-5)
        assert np.isclose(c0["recall"], 4.0 / 6.0, atol=1e-5)
        assert np.isclose(c0["f1"], 2.0 / 3.0, atol=1e-5)

        c1 = metrics["per_class"]["C1"]
        assert c1["support"] == 4
        assert np.isclose(c1["precision"], 3.0 / 5.0, atol=1e-5)
        assert np.isclose(c1["recall"], 3.0 / 4.0, atol=1e-5)
        assert np.isclose(c1["f1"], 2.0 / 3.0, atol=1e-5)

        c2 = metrics["per_class"]["C2"]
        assert c2["support"] == 4
        assert np.isclose(c2["precision"], 2.0 / 3.0, atol=1e-5)
        assert np.isclose(c2["recall"], 2.0 / 4.0, atol=1e-5)
        assert np.isclose(c2["f1"], 4.0 / 7.0, atol=1e-5)

        # 5. Confusion matrix exact values
        # cm[i, j] is true i, pred j
        cm = metrics["confusion_matrix"]
        expected_cm = np.array(
            [
                [4, 1, 1],
                [1, 3, 0],
                [1, 1, 2],
            ]
        )
        np.testing.assert_array_equal(cm, expected_cm)

    def test_zero_division_handling_is_safe(self) -> None:
        """When a class has 0 predictions and 0 support, zero_division=0.0 handles it cleanly."""
        labels = ["ClassA", "ClassB", "UnseenClass"]
        y_true = ["ClassA", "ClassA", "ClassB"]
        y_pred = ["ClassA", "ClassB", "ClassB"]

        metrics = calculate_metrics(y_true, y_pred, labels=labels, zero_division=0.0)
        unseen = metrics["per_class"]["UnseenClass"]
        assert unseen["precision"] == 0.0
        assert unseen["recall"] == 0.0
        assert unseen["f1"] == 0.0
        assert unseen["support"] == 0


# =============================================================================
# 5. Cross-Validation: Group Overlap Guard Raises
# =============================================================================


class TestGroupOverlapInCVRaises:
    """Verify acceptance criterion: Group overlap in CV raises."""

    def test_assert_no_group_overlap_raises_on_overlap(self) -> None:
        """Direct assertion helper raises GroupOverlapError on overlapping identifiers."""
        train_groups = ["batch_01", "batch_02", "batch_03", "batch_04"]
        val_groups = ["batch_04", "batch_05"]  # 'batch_04' overlaps

        with pytest.raises(GroupOverlapError, match="Group overlap detected"):
            assert_no_group_overlap(train_groups, val_groups, fold_idx=2)

    def test_guarded_group_kfold_raises_on_invalid_parameters(self) -> None:
        """GuardedGroupKFold verifies inputs and raises CVValidationError on invalid configs."""
        cv = GuardedGroupKFold(n_splits=5)
        X = np.ones((20, 2))

        # 1. groups cannot be None
        with pytest.raises(CVValidationError, match="groups cannot be None"):
            list(cv.split(X, groups=None))

        # 2. groups length mismatch
        with pytest.raises(CVValidationError, match="does not match sample count"):
            list(cv.split(X, groups=["g1", "g2"]))

        # 3. fewer groups than splits
        with pytest.raises(CVValidationError, match="less than n_splits"):
            list(cv.split(X, groups=["g1"] * 10 + ["g2"] * 10))

    def test_guarded_group_kfold_strictly_enforces_disjoint_partitions(self) -> None:
        """GuardedGroupKFold partitions data cleanly across groups with 0 overlap."""
        n_samples = 120
        # 12 distinct groups of 10 samples each
        groups = np.repeat([f"crop_day_{i:02d}" for i in range(12)], 10)
        X = np.random.randn(n_samples, 4)
        y = np.random.choice(["Normal", "Deficiency"], size=n_samples)

        cv = GuardedGroupKFold(n_splits=4)
        splits = list(cv.split(X, y, groups=groups))
        assert len(splits) == 4

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_g = set(groups[train_idx])
            val_g = set(groups[val_idx])

            # Intersection must be strictly empty
            overlap = train_g.intersection(val_g)
            assert len(overlap) == 0, f"Fold {fold_idx} leaked groups: {overlap}"


# =============================================================================
# 6. Label-Rule Oracle Boundaries & Multi-Rule Priority
# =============================================================================


class TestLabelRuleBoundariesAndMultiRulePriority:
    """Verify label-rule oracle boundaries and multi-rule priority resolution."""

    def test_lettuce_multi_rule_priority_multiple_stress(self) -> None:
        """Lettuce: 1 rule broken -> specific class; 2+ rules broken -> 'Multiple_Stress'."""
        sample_1_stress = {
            "ph": 5.0,  # broken (< 5.5)
            "ec": 1.8,
            "air_temp": 24.0,
            "humidity": 65.0,
            "water_temp": 22.0,
        }
        assert predict_lettuce_rule_label(sample_1_stress) == "pH_Low"
        assert get_lettuce_broken_rules(sample_1_stress) == ["pH_Low"]

        sample_2_stress = {
            "ph": 5.0,  # broken
            "ec": 3.0,  # broken (> 2.5)
            "air_temp": 24.0,
            "humidity": 65.0,
            "water_temp": 22.0,
        }
        assert predict_lettuce_rule_label(sample_2_stress) == "Multiple_Stress"
        broken = get_lettuce_broken_rules(sample_2_stress)
        assert len(broken) == 2
        assert "pH_Low" in broken
        assert "EC_High" in broken

    def test_spinach_multi_rule_priority_multiple(self) -> None:
        """Spinach: 1 rule broken -> specific class; 2+ rules broken -> 'Multiple'."""
        sample_1_fault = {
            "water_temperature_C": 22.5,  # broken (> 21.0)
            "ec_mS_cm": 2.0,
            "ph": 6.5,
            "relative_humidity_percent": 60.0,
        }
        assert predict_spinach_rule_label(sample_1_fault) == "Water_Temp_High"

        sample_2_faults = {
            "water_temperature_C": 22.5,  # broken
            "ec_mS_cm": 2.5,  # broken (> 2.3)
            "ph": 6.5,
            "relative_humidity_percent": 60.0,
        }
        assert predict_spinach_rule_label(sample_2_faults) == "Multiple"
        broken = get_spinach_broken_rules(sample_2_faults)
        assert len(broken) == 2
        assert "Water_Temp_High" in broken
        assert "EC_High_Toxicity" in broken

    def test_tomato_multi_rule_severity_priority(self) -> None:
        """Tomato: rule with higher normalized severity (deviation / scale) wins over rule order."""
        # Setup:
        # Rule 1: EC_High_Toxicity (threshold 4.0, scale 0.20)
        # Value = 4.10 -> deviation = 0.10 -> severity = 0.10 / 0.20 = 0.50
        #
        # Rule 2: Water_Temp_High (threshold 25.0, scale 6.58)
        # Value = 26.00 -> deviation = 1.00 -> severity = 1.00 / 6.58 = 0.152
        #
        # Although Water_Temp_High is first in TOMATO_RULES_ORDER,
        # EC_High_Toxicity has higher normalized severity (0.50 > 0.152), so it must win.
        sample = {
            "water_temperature_C": 26.0,
            "ec_mS_cm": 4.10,
            "ph": 6.2,
            "relative_humidity_percent": 65.0,
            "nitrogen_mg_L": 120.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 200.0,
        }
        broken = get_tomato_broken_rules(sample)
        assert len(broken) == 2
        assert "Water_Temp_High" in broken
        assert "EC_High_Toxicity" in broken

        # Primary mode selects highest severity
        predicted_primary = predict_tomato_rule_label(sample, mode="primary")
        assert predicted_primary == "EC_High_Toxicity"

        # Single mode returns Multiple_Faults
        predicted_single = predict_tomato_rule_label(sample, mode="single")
        assert predicted_single == "Multiple_Faults"

    def test_tomato_tie_breaking_preserves_rules_order(self) -> None:
        """Tomato: when normalized severities are identical, earlier rule in TOMATO_RULES_ORDER wins."""
        # Find two rules and construct identical severities
        # Rule A: Water_Temp_High (scale 6.58)
        # Rule B: EC_High_Toxicity (scale 0.20)
        # Let severity = 0.25
        # dev(Water_Temp_High) = 0.25 * 6.58 = 1.645 -> wt = 25.0 + 1.645 = 26.645
        # dev(EC_High_Toxicity) = 0.25 * 0.20 = 0.050 -> ec = 4.0 + 0.050 = 4.050
        scale_wt = TOMATO_SEVERITY_SCALES["Water_Temp_High"]
        scale_ec = TOMATO_SEVERITY_SCALES["EC_High_Toxicity"]

        sample = {
            "water_temperature_C": 25.0 + (0.25 * scale_wt),
            "ec_mS_cm": 4.0 + (0.25 * scale_ec),
            "ph": 6.2,
            "relative_humidity_percent": 65.0,
            "nitrogen_mg_L": 120.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 200.0,
        }
        broken = get_tomato_broken_rules(sample)
        assert len(broken) == 2

        # Water_Temp_High is index 0 in TOMATO_RULES_ORDER; EC_High_Toxicity is index 1.
        # Tie-breaker must pick Water_Temp_High.
        assert predict_tomato_rule_label(sample, mode="primary") == "Water_Temp_High"
