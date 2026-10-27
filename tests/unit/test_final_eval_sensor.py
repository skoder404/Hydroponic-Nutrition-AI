"""Unit tests for the run-once final test evaluation (T4-024).

Acceptance Criteria:
1. First run writes metrics and a log entry.
2. Second run without `--force` refuses.
3. Report labels metrics as test results on the documented split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.final_eval_sensor import (
    check_already_evaluated,
    main,
    read_evaluation_log,
    run_final_test_evaluation,
)


@pytest.fixture
def eval_environment(tmp_path: Path) -> dict[str, Path]:
    """Create isolated temporary environment with real data/reports directories."""
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"
    registry_root = tmp_path / "models"
    log_file = reports_dir / "final_eval_log.jsonl"

    # Copy real sensor_selection.json files for lettuce, spinach, tomato into fixture reports_dir
    import shutil
    for c in ("lettuce", "spinach", "tomato"):
        src = Path("reports") / c / "sensor_selection.json"
        if src.is_file():
            dst = reports_dir / c / "sensor_selection.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return {
        "data_dir": data_dir,
        "reports_dir": reports_dir,
        "registry_root": registry_root,
        "log_file": log_file,
    }


def test_first_run_writes_metrics_and_log_entry(
    eval_environment: dict[str, Path],
) -> None:
    """Criterion 1: First run writes metrics, JSON/Markdown reports, and a log entry."""
    reports_dir = eval_environment["reports_dir"]
    registry_root = eval_environment["registry_root"]
    log_file = eval_environment["log_file"]

    # Run evaluation on lettuce using real processed_data and selection json
    res = run_final_test_evaluation(
        crop="lettuce",
        reports_dir=reports_dir,
        registry_root=registry_root,
        log_file=log_file,
        force=False,
    )

    # 1. Returned metrics
    assert "test_metrics" in res
    metrics = res["test_metrics"]
    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert "macro_recall" in metrics
    assert metrics["macro_f1"] > 0.0
    assert metrics["accuracy"] > 0.0

    # 2. Reports generated
    json_path = res["report_json"]
    md_path = res["report_md"]
    assert json_path.is_file()
    assert md_path.is_file()

    with open(json_path, encoding="utf-8") as f:
        report_data = json.load(f)
    assert report_data["split_evaluated"] == "test"
    assert report_data["crop"] == "lettuce"

    # 3. Log file entry created
    assert log_file.is_file()
    entries = read_evaluation_log(log_file)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["crop"] == "lettuce"
    assert entry["split_evaluated"] == "test"
    assert entry["is_rerun"] is False
    assert entry["rerun_reason"] is None
    assert entry["test_metrics"]["macro_f1"] == round(float(metrics["macro_f1"]), 6)


def test_second_run_without_force_refuses(
    eval_environment: dict[str, Path],
) -> None:
    """Criterion 2: Second run without --force refuses with an explicit error."""
    reports_dir = eval_environment["reports_dir"]
    registry_root = eval_environment["registry_root"]
    log_file = eval_environment["log_file"]

    # First run
    run_final_test_evaluation(
        crop="lettuce",
        reports_dir=reports_dir,
        registry_root=registry_root,
        log_file=log_file,
        force=False,
    )
    assert check_already_evaluated("lettuce", log_file) is True

    # Second run without force must raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        run_final_test_evaluation(
            crop="lettuce",
            reports_dir=reports_dir,
            registry_root=registry_root,
            log_file=log_file,
            force=False,
        )
    msg = str(exc_info.value)
    assert "Final test evaluation has already been run for 'lettuce'" in msg
    assert "Re-running requires --force" in msg


def test_second_run_with_force_requires_written_reason(
    eval_environment: dict[str, Path],
) -> None:
    """Criterion 2: Re-running with --force requires a non-empty written --reason."""
    reports_dir = eval_environment["reports_dir"]
    registry_root = eval_environment["registry_root"]
    log_file = eval_environment["log_file"]

    # First run
    run_final_test_evaluation(
        crop="lettuce",
        reports_dir=reports_dir,
        registry_root=registry_root,
        log_file=log_file,
        force=False,
    )

    # Force with empty or None reason must fail
    with pytest.raises(ValueError, match="requires a non-empty written --reason"):
        run_final_test_evaluation(
            crop="lettuce",
            reports_dir=reports_dir,
            registry_root=registry_root,
            log_file=log_file,
            force=True,
            reason="",
        )

    with pytest.raises(ValueError, match="requires a non-empty written --reason"):
        run_final_test_evaluation(
            crop="lettuce",
            reports_dir=reports_dir,
            registry_root=registry_root,
            log_file=log_file,
            force=True,
            reason="   ",
        )

    # Force with valid written reason succeeds and logs rerun
    run_final_test_evaluation(
        crop="lettuce",
        reports_dir=reports_dir,
        registry_root=registry_root,
        log_file=log_file,
        force=True,
        reason="Auditing test split repeatability across identical random seeds",
    )

    entries = read_evaluation_log(log_file)
    assert len(entries) == 2
    second_entry = entries[1]
    assert second_entry["is_rerun"] is True
    assert (
        second_entry["rerun_reason"]
        == "Auditing test split repeatability across identical random seeds"
    )


def test_report_labels_metrics_as_test_results_on_documented_split(
    eval_environment: dict[str, Path],
) -> None:
    """Criterion 3: Report labels metrics as test results on the documented split."""
    reports_dir = eval_environment["reports_dir"]
    registry_root = eval_environment["registry_root"]
    log_file = eval_environment["log_file"]

    res = run_final_test_evaluation(
        crop="lettuce",
        reports_dir=reports_dir,
        registry_root=registry_root,
        log_file=log_file,
        force=False,
    )

    # 1. JSON report assertion
    with open(res["report_json"], encoding="utf-8") as f:
        json_data = json.load(f)

    assert json_data["split_evaluated"] == "test"
    assert "test" in json_data["split_counts"]
    assert json_data["split_counts"]["test"] == 162
    assert "Preserved calendar day / group holdout" in json_data["split_protocol"]

    # 2. Markdown report assertion
    with open(res["report_md"], encoding="utf-8") as f:
        md_text = f.read()

    assert "> **Evaluated Split:** `test`" in md_text
    assert "exclusively on the **`test`** partition" in md_text
    assert "## 7. Mandatory Limitations & Disclosures" in md_text
    assert "Final Test Evaluation:" in md_text


def test_unsupported_crop_raises_value_error(
    eval_environment: dict[str, Path],
) -> None:
    """Invoking with an unsupported crop raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported crop 'kale'"):
        run_final_test_evaluation(
            crop="kale",
            reports_dir=eval_environment["reports_dir"],
            registry_root=eval_environment["registry_root"],
            log_file=eval_environment["log_file"],
        )


def test_cli_execution(eval_environment: dict[str, Path]) -> None:
    """Test CLI invocation with flags."""
    reports_dir = str(eval_environment["reports_dir"])
    registry_root = str(eval_environment["registry_root"])
    log_file = str(eval_environment["log_file"])

    # 1. Positional crop
    ret = main(
        [
            "lettuce",
            "--reports-dir",
            reports_dir,
            "--registry-root",
            registry_root,
            "--log-file",
            log_file,
        ]
    )
    assert ret == 0

    # 2. Second CLI run without force exits non-zero (refusal)
    ret_refuse = main(
        [
            "lettuce",
            "--reports-dir",
            reports_dir,
            "--registry-root",
            registry_root,
            "--log-file",
            log_file,
        ]
    )
    assert ret_refuse == 1

    # 3. Second CLI run with --force and --reason succeeds
    ret_force = main(
        [
            "--crop",
            "lettuce",
            "--force",
            "--reason",
            "CLI regression test",
            "--reports-dir",
            reports_dir,
            "--registry-root",
            registry_root,
            "--log-file",
            log_file,
        ]
    )
    assert ret_force == 0
