"""Unit tests for learning scaffold and progress tracker (T1-015)."""
from scripts.verify_learning_scaffold import (
    check_concept_template,
    check_progress_tracker,
    check_teachback_template,
)


def test_concept_template_structure():
    ok, msg = check_concept_template()
    assert ok, msg


def test_teachback_template_structure():
    ok, msg = check_teachback_template()
    assert ok, msg


def test_progress_tracker_contents_and_columns():
    ok, msg = check_progress_tracker()
    assert ok, msg


def test_phase4_teachback_and_concept_check():
    """Acceptance criteria verification for T4-026: Phase 4 teach-back and concept check."""
    from pathlib import Path
    import re

    # 1. Verify all 10 Phase 4 concept files exist and are populated
    phase4_concept_files = [
        "data-leakage-and-grouped-timebased-splits.md",
        "decision-trees-gini-impurity-and-greedy.md",
        "confusion-matrix-precisionrecall-macrof1.md",
        "crossvalidation-and-class-imbalance-oversampling-inside.md",
        "decision-tree-hyperparameters-depth-and-overfitting.md",
        "random-forest-bagging-feature-subsampling-variance.md",
        "svm-margin-c-rbf-kernel-and.md",
        "knn-and-logistic-regression-baselines-distance.md",
        "hyperparameter-search-with-grouped-cv-and.md",
        "why-rulederived-labels-inflate-scores.md",
    ]
    for cf in phase4_concept_files:
        p = Path("docs/learning") / cf
        assert p.is_file(), f"Missing concept note: {cf}"
        assert len(p.read_text(encoding="utf-8")) > 500, f"Concept note too short: {cf}"

    # 2. Verify teach-back session log exists and has both members signed
    tb_log = Path("docs/learning/phase4_teachback_log.md")
    assert tb_log.is_file(), "Missing docs/learning/phase4_teachback_log.md"
    tb_text = tb_log.read_text(encoding="utf-8")
    assert "Yashwant" in tb_text and "Sushil" in tb_text
    assert "PASS" in tb_text

    # 3. Verify PROGRESS.md marks all 10 concepts complete for both members
    progress_text = Path("docs/learning/PROGRESS.md").read_text(encoding="utf-8")
    for cf in phase4_concept_files:
        line_match = re.search(rf"`docs/learning/{re.escape(cf)}`\s*\|\s*\[x\]\s*\|\s*\[x\]\s*\|\s*\[x\].*\|\s*\[x\]\s*\|\s*\[x\]\s*\|\s*\[x\]", progress_text)
        assert line_match is not None, f"Concept {cf} not fully ticked for both members in PROGRESS.md"

