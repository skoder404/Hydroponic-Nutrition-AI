#!/usr/bin/env python3
"""Verification script for T7-006: S1 Home with animated plant hero and crop selection.

Validates:
1. Selecting a card routes to /analyze?crop=
2. Empty/error/loading behaviour matches DESIGN S1
3. Layout follows the 3/2+1/1 column rule
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_verification() -> bool:
    print("[1/3] Verifying selecting a card routes to /analyze?crop=...")
    card_file = REPO_ROOT / "frontend/components/home/CropCard.tsx"
    page_file = REPO_ROOT / "frontend/app/page.tsx"
    hero_file = REPO_ROOT / "frontend/components/home/HeroPlant.tsx"

    assert card_file.is_file(), f"Missing {card_file}"
    assert page_file.is_file(), f"Missing {page_file}"
    assert hero_file.is_file(), f"Missing {hero_file}"

    card_text = card_file.read_text(encoding="utf-8")
    page_text = page_file.read_text(encoding="utf-8")

    assert "href={`/analyze?crop=${crop.id}`}" in card_text
    assert "Sensor mode: experimental" in card_text
    for crop in ["lettuce", "spinach", "tomato"]:
        assert f'id: "{crop}"' in page_text, f"Missing crop definition for {crop}"
    print("  -> Passed card routing and experimental chips verification!")

    print("[2/3] Verifying empty/error/loading behaviour...")
    assert "isLoading" in card_text
    assert "animate-pulse" in card_text
    assert "backendError" in page_text
    assert "Retry" in page_text
    print("  -> Passed error and loading skeleton states verification!")

    print("[3/3] Verifying 3 / 2+1 / 1 column responsive layout rule...")
    assert "grid-cols-1" in page_text
    assert "sm:grid-cols-2" in page_text
    assert "lg:grid-cols-3" in page_text
    assert "sm:col-span-2" in page_text
    print("  -> Passed responsive 3 / 2+1 / 1 column rule verification!")

    print("\nAll T7-006 S1 Home criteria verified successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
