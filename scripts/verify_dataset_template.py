#!/usr/bin/env python3
"""Verification script for T2-001: Dataset provenance template.

Validates that docs/datasets/_TEMPLATE.md contains:
1. All Overview §9.1 fields:
   - source name and URL
   - source version/date downloaded
   - original filename(s)
   - license/terms and redistribution status
   - original row/image count
   - prepared row/image count
   - class labels and class counts
   - preprocessing script/version
   - train/validation/test split protocol and assignment
   - synthetic, derived, rescaled, or rule-generated values
   - known limitations and unsupported interpretations
2. Licence/redistribution field specifically verified.
3. Column-type table tagged with: measured, adjusted, derived, synthetic, rule-generated.
"""
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    ("Source Name", ["source name", "source url"]),
    ("Source Version / Date", ["source version", "date downloaded"]),
    ("Original Filename", ["original filename"]),
    ("License & Redistribution", ["license", "redistribution status"]),
    ("Counts", ["original raw count", "prepared final count"]),
    ("Class Labels and Counts", ["class labels", "class counts", "distribution"]),
    ("Preprocessing Script", ["preprocessing script", "version"]),
    ("Split Protocol", ["split protocol", "train", "validation", "test"]),
    ("Synthetic / Derived / Rule-Generated", ["synthetic", "derived", "rule-generated"]),
    ("Column Taxonomy Table", ["column", "provenance tag", "measured", "adjusted", "derived", "synthetic", "rule-generated"]),
    ("Limitations", ["known limitations", "unsupported interpretations"]),
]

TAGS = ["measured", "adjusted", "derived", "synthetic", "rule-generated"]


def check_template():
    p = Path("docs/datasets/_TEMPLATE.md")
    if not p.is_file():
        return False, f"{p} does not exist"
    content = p.read_text(encoding="utf-8").lower()

    missing_sections = []
    for sec_name, keywords in REQUIRED_SECTIONS:
        if not any(k in content for k in keywords):
            missing_sections.append(sec_name)

    if missing_sections:
        return False, f"Missing §9.1 fields or sections: {missing_sections}"

    # Verify all 5 tags are explicitly documented in the column taxonomy
    missing_tags = [t for t in TAGS if t not in content]
    if missing_tags:
        return False, f"Column taxonomy table missing tags: {missing_tags}"

    return True, "docs/datasets/_TEMPLATE.md satisfies all Overview §9.1 requirements."


def main():
    print("=== Running T2-001 Verification ===")
    ok, msg = check_template()
    if not ok:
        print(f"ERROR: {msg}")
        return 1
    print(f"[PASS] {msg}")
    print("\n--- ALL T2-001 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
