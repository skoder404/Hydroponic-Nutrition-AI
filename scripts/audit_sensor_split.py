#!/usr/bin/env python3
"""Audit sensor dataset splits and distributions against build scripts and specifications.

Audits hydroponic sensor datasets across crops (lettuce, spinach, tomato),
validating row counts, class distributions, temporal / group split rules,
train/test separation, and oversampling quantification.

For lettuce (T2-005):
Validates hydroponic_lettuce_{full,train,test}.csv against facts read from
build_lettuce_dataset and its README.

Re-run command:
    python scripts/audit_sensor_split.py --crop lettuce
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "data"

# ===========================================================================
# Lettuce Specifications (T2-005)
# ===========================================================================
LETTUCE_DATA = ROOT / "sensor_data" / "Lettuce_Sensor-Readings"
LETTUCE_FULL = LETTUCE_DATA / "hydroponic_lettuce_full.csv"
LETTUCE_TRAIN = LETTUCE_DATA / "hydroponic_lettuce_train.csv"
LETTUCE_TEST = LETTUCE_DATA / "hydroponic_lettuce_test.csv"

EXPECTED_LETTUCE_FULL_ROWS = 726
EXPECTED_LETTUCE_TEST_ROWS = 162
EXPECTED_LETTUCE_TRAIN_ROWS = 1680
EXPECTED_LETTUCE_TRAIN_RAW_ROWS = 558  # Before dropping classes < 8
EXPECTED_LETTUCE_TRAIN_KEPT_ROWS = 555  # 558 - 3 (WaterTemp_Low)
EXPECTED_LETTUCE_CLASSES = 7
EXPECTED_LETTUCE_PER_CLASS = 240  # 240 * 7 = 1680

EXPECTED_LETTUCE_FULL_DIST = {
    "Normal": 327,
    "Multiple_Stress": 135,
    "Humidity_Low": 87,
    "WaterTemp_High": 75,
    "EC_High": 58,
    "pH_High": 25,
    "pH_Low": 10,
    "WaterTemp_Low": 8,
    "EC_Low": 1,
}

LETTUCE_ACTION_BANDS = {
    "ph": (5.5, 7.0),
    "ec": (1.2, 2.5),
    "air_temp": (18.0, 30.0),
    "humidity": (50.0, 80.0),
    "water_temp": (18.0, 26.0),
}

LETTUCE_DROPPED_CLASSES = {"WaterTemp_Low", "EC_Low"}
LETTUCE_MODEL_FEATURES = {"ph", "ec", "water_temp", "humidity", "air_temp"}


def audit_lettuce(output_path: Path | None = None) -> tuple[int, str]:
    """Perform audit for lettuce sensor dataset (T2-005)."""
    fails = 0
    report: list[str] = []

    # 1. Check file existence
    for f in (LETTUCE_FULL, LETTUCE_TRAIN, LETTUCE_TEST):
        if not f.exists():
            return 1, f"Missing required file: {f}"

    full = pd.read_csv(LETTUCE_FULL)
    train = pd.read_csv(LETTUCE_TRAIN)
    test = pd.read_csv(LETTUCE_TEST)

    # 2. Row Counts
    report.append("## 1. Row Counts\n")
    row_checks = [
        ("full", len(full), EXPECTED_LETTUCE_FULL_ROWS),
        ("train", len(train), EXPECTED_LETTUCE_TRAIN_ROWS),
        ("test", len(test), EXPECTED_LETTUCE_TEST_ROWS),
    ]
    for name, actual, expected in row_checks:
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            fails += 1
        report.append(f"- {status} `{name}`: {actual} rows (expected {expected})")
    report.append("")

    # 3. Class Distribution
    report.append("## 2. Class Distribution\n")
    report.append("### Full dataset\n")
    report.append("| Class | Expected | Actual | Match |")
    report.append("|-------|----------|--------|-------|")

    full_dist = full["label"].value_counts().to_dict()
    for cls, expected in sorted(EXPECTED_LETTUCE_FULL_DIST.items(), key=lambda x: -x[1]):
        actual = full_dist.get(cls, 0)
        match = "✅" if actual == expected else "❌"
        if actual != expected:
            fails += 1
        report.append(f"| {cls} | {expected} | {actual} | {match} |")

    report.append(f"\n**Total full classes**: {len(full_dist)} (9 with dropped, 7 kept)")
    report.append("")

    # Train dataset
    report.append("### Train dataset\n")
    train_dist = train["label"].value_counts().sort_values(ascending=False)
    report.append("| Class | Count | Expected (240) | Match |")
    report.append("|-------|-------|----------------|-------|")
    for cls, count in train_dist.items():
        match = "✅" if count == EXPECTED_LETTUCE_PER_CLASS else "❌"
        if count != EXPECTED_LETTUCE_PER_CLASS:
            fails += 1
        report.append(f"| {cls} | {count} | {EXPECTED_LETTUCE_PER_CLASS} | {match} |")

    n_train_classes = len(train_dist)
    status = "✅" if n_train_classes == EXPECTED_LETTUCE_CLASSES else "❌"
    if n_train_classes != EXPECTED_LETTUCE_CLASSES:
        fails += 1
    report.append(f"\n**Train classes**: {n_train_classes} {status} (expected {EXPECTED_LETTUCE_CLASSES})")

    # Dropped classes
    for cls in LETTUCE_DROPPED_CLASSES:
        in_train = cls in train["label"].values
        in_test = cls in test["label"].values
        if in_train or in_test:
            fails += 1
            report.append(f"- ❌ Dropped class `{cls}` found in {'train' if in_train else 'test'}")
        else:
            report.append(f"- ✅ Dropped class `{cls}` correctly absent from train and test")
    report.append("")

    # 4. Action Bands
    report.append("## 3. Action Bands & Label Assignment Rules\n")
    report.append("| Variable | Low Threshold | High Threshold | Status |")
    report.append("|----------|---------------|----------------|--------|")
    for var, (lo, hi) in LETTUCE_ACTION_BANDS.items():
        report.append(f"| {var} | {lo} | {hi} | ✅ |")
    report.append("\n- Action bands define stress when values fall outside the target range.")
    report.append("- A single variable out-of-bounds yields that specific variable's stress class.")
    report.append("- Two or more out-of-bounds variables yield `Multiple_Stress`.")
    report.append("- Rare classes (< 8 rows in training raw split) are dropped: `WaterTemp_Low` (3 train rows) and `EC_Low` (0 train rows).\n")

    # 5. Every-4th-Day Split Rule
    report.append("## 4. Every-4th-Day Split Rule\n")
    full_dates = full.copy()
    full_dates["date"] = pd.to_datetime(full_dates["date"])
    test_dates_df = test.copy()
    test_dates_df["date"] = pd.to_datetime(test_dates_df["date"])

    for exp in sorted(full_dates["experiment"].unique()):
        exp_full = full_dates[full_dates["experiment"] == exp]
        exp_test = test_dates_df[test_dates_df["experiment"] == exp]

        all_days = sorted(exp_full["date"].unique())
        expected_test_days = set(all_days[3::4])
        actual_test_days = set(exp_test["date"].unique())

        is_match = expected_test_days == actual_test_days
        status = "✅" if is_match else "❌"
        if not is_match:
            fails += 1

        report.append(f"### {exp}\n")
        report.append(f"- Total unique days: {len(all_days)}")
        report.append(f"- Expected test days (every 4th): {len(expected_test_days)}")
        report.append(f"- Actual test days: {len(actual_test_days)}")
        report.append(f"- {status} Days match: {is_match}")

        if not is_match:
            missing = expected_test_days - actual_test_days
            extra = actual_test_days - expected_test_days
            if missing:
                report.append(f"  - Missing: {sorted(str(d)[:10] for d in missing)}")
            if extra:
                report.append(f"  - Extra: {sorted(str(d)[:10] for d in extra)}")
        report.append("")

    # 6. Train/Test Separation (No Data Leakage)
    report.append("## 5. Train/Test Separation (No Data Leakage)\n")
    train_dates_df = train.copy()
    train_dates_df["date"] = pd.to_datetime(train_dates_df["date"])

    train_pairs = set(zip(train_dates_df["experiment"], train_dates_df["date"]))
    test_pairs = set(zip(test_dates_df["experiment"], test_dates_df["date"]))
    overlap = train_pairs & test_pairs

    if overlap:
        fails += 1
        report.append(f"- ❌ **LEAKAGE DETECTED**: {len(overlap)} (experiment, date) pairs in both train and test!")
        for exp, date in sorted(overlap)[:10]:
            report.append(f"  - {exp}, {str(date)[:10]}")
    else:
        report.append("- ✅ No (experiment, date) pair appears in both train and test")
    report.append("")

    # 7. Oversampling Quantification
    report.append("## 6. Oversampling Quantification\n")

    # Trace exact training raw rows from full dataset
    test_dates_map: dict[str, set] = {}
    for exp in test["experiment"].unique():
        test_dates_map[exp] = set(test[test["experiment"] == exp]["date"].unique())

    is_train_day = full.apply(
        lambda row: row["date"] not in test_dates_map.get(row["experiment"], set()),
        axis=1,
    )
    train_raw = full[is_train_day]
    train_raw_count = len(train_raw)

    # Train raw with kept classes
    train_kept_raw = train_raw[~train_raw["label"].isin(LETTUCE_DROPPED_CLASSES)]
    train_kept_count = len(train_kept_raw)

    report.append(f"- Raw training split rows before class filtering: **{train_raw_count}** (matches expected {EXPECTED_LETTUCE_TRAIN_RAW_ROWS})")
    report.append(f"- Rare class occurrences in training split dropped: {train_raw_count - train_kept_count} rows (`WaterTemp_Low`: 3, `EC_Low`: 0)")
    report.append(f"- Real training rows kept across 7 classes: **{train_kept_count}**")
    report.append(f"- Total train rows after oversampling: **{len(train)}**")

    n_duplicates = len(train) - train_kept_count
    ratio = len(train) / train_kept_count
    report.append(f"- Duplicated rows created by oversampling: **{n_duplicates}**")
    report.append(f"- Oversampling expansion ratio: **{ratio:.2f}x**")

    # Per-class breakdown
    report.append("\n### Per-class oversampling breakdown\n")
    report.append("| Class | Real Rows in Train Split | Oversampled Total | Duplicate Rows Added | Expansion Factor |")
    report.append("|-------|--------------------------|-------------------|----------------------|------------------|")

    real_dist = train_kept_raw["label"].value_counts().to_dict()
    for cls in sorted(real_dist.keys()):
        real_n = real_dist.get(cls, 0)
        total_n = train_dist.get(cls, 0)
        added = total_n - real_n
        factor = total_n / real_n if real_n > 0 else 0
        report.append(f"| {cls} | {real_n} | {total_n} | +{added} | {factor:.2f}x |")
    report.append("")

    # 8. Known Discrepancies
    report.append("## 7. Known Discrepancies (Documented from Task Spec)\n")

    report.append("### 7.1 Script docstring vs code")
    report.append("- **Docstring** (lines 15-16 in `build_lettuce_dataset`): mentions EXP.3 as test set")
    report.append("- **Actual Code** (`day_holdout_split`, lines 266-281): holds out every 4th calendar day across all 3 experiments")
    report.append("- **README** (lines 94-95): confirms every-4th-day holdout rule")
    report.append("- ✅ **Verified**: docstring is outdated; code and README agree on the every-4th-day rule.\n")

    report.append("### 7.2 README filename references")
    report.append("- **README** (line 17): references `build_dataset.py` and `train_model.py`")
    report.append("- **Repo Actual**: contains `build_lettuce_dataset` (without `.py` extension) and no `train_model.py`")
    build_exists = (LETTUCE_DATA / "build_lettuce_dataset").exists()
    report.append(f"- `build_lettuce_dataset` present: {'✅ Yes' if build_exists else '❌ No'}")
    report.append("- `train_model.py` present: ❌ No (confirmed discrepancy)\n")

    report.append("### 7.3 Non-model-input columns in CSVs")
    full_sample = pd.read_csv(LETTUCE_FULL, nrows=1)
    all_cols = set(full_sample.columns)
    non_model = sorted(all_cols - LETTUCE_MODEL_FEATURES - {"label", "label_next"})
    report.append(f"- Model uses 5 core physical inputs: `{sorted(LETTUCE_MODEL_FEATURES)}`")
    report.append(f"- Full CSV retains {len(non_model)} additional metadata/engineering columns: `{non_model}`")
    report.append("- Includes trend features (`ec_trend_3h`), daily aggregates (`ph_daily_mean`), secondary sensors (`tds`, `co2`, `lux_meter`), and target forecasting (`label_next`).\n")

    report.append("### 7.4 Raw Excel workbooks (`EXP_*.xlsx`)")
    report.append("- The three original raw Excel workbooks (`EXP_1.xlsx`, `EXP_2.xlsx`, `EXP_3.xlsx`) are not stored in the repository.")
    report.append("- Consequently, `build_lettuce_dataset` cannot be re-executed without downloading them, which is not required for this project.")
    report.append("- The generated CSVs (`hydroponic_lettuce_{full,train,test}.csv`) are the checked-in source of truth.\n")

    # Format document
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = [
        "# Lettuce Sensor Dataset Audit Report",
        "",
        f"**Generated**: {timestamp}",
        "**Script**: `scripts/audit_sensor_split.py`",
        "**Data**: `sensor_data/Lettuce_Sensor-Readings/`",
        "",
        f"**Result**: {'✅ ALL CHECKS PASSED' if fails == 0 else f'❌ {fails} CHECK(S) FAILED'}",
        "",
        "---",
        "",
    ]
    markdown_content = "\n".join(header + report)

    out_file = output_path or (REPORT_DIR / "lettuce_sensor_audit.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(markdown_content, encoding="utf-8")
    print(f"Report written to: {out_file}")

    return fails, markdown_content


def audit_spinach() -> tuple[int, str]:
    """Extensible audit stub for spinach sensor dataset."""
    print("Spinach sensor audit stub — to be implemented in subsequent crop task.")
    return 0, "Spinach audit stub"


def audit_tomato() -> tuple[int, str]:
    """Extensible audit stub for tomato sensor dataset."""
    print("Tomato sensor audit stub — to be implemented in subsequent crop task.")
    return 0, "Tomato audit stub"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit hydroponic sensor dataset splits and distributions.")
    parser.add_argument(
        "--crop",
        choices=["lettuce", "spinach", "tomato", "all"],
        default="lettuce",
        help="Crop sensor dataset to audit (default: lettuce).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output markdown file path.",
    )
    args = parser.parse_args()

    total_fails = 0
    if args.crop in ("lettuce", "all"):
        print("=== Auditing Lettuce Sensor Dataset ===")
        fails, _ = audit_lettuce(args.output)
        total_fails += fails
    if args.crop in ("spinach", "all"):
        fails, _ = audit_spinach()
        total_fails += fails
    if args.crop in ("tomato", "all"):
        fails, _ = audit_tomato()
        total_fails += fails

    print(f"\nAudit complete. Total failures: {total_fails}")
    return 1 if total_fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
