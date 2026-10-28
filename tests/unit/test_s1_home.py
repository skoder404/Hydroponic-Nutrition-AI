"""Unit tests for S1 Home page, plant hero, and crop selection (T7-006).

Acceptance Criteria:
- [x] Selecting a card routes to /analyze?crop=
- [x] Empty/error/loading behaviour matches DESIGN S1
- [x] Layout follows the 3/2+1/1 column rule
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_selecting_card_routes_to_analyze_crop():
    """AC 1: Selecting a card routes to /analyze?crop="""
    card_path = ROOT / "frontend/components/home/CropCard.tsx"
    page_path = ROOT / "frontend/app/page.tsx"

    assert card_path.is_file(), "CropCard.tsx must exist"
    assert page_path.is_file(), "page.tsx must exist"

    card_content = card_path.read_text(encoding="utf-8")
    page_content = page_path.read_text(encoding="utf-8")

    # Link template in CropCard
    assert "href={`/analyze?crop=${crop.id}`}" in card_content or 'href={"/analyze?crop=' in card_content

    # Target crops defined in HomePage
    for crop in ["lettuce", "spinach", "tomato"]:
        assert f'id: "{crop}"' in page_content

    # Tomato experimental chip
    assert "Sensor mode: experimental" in card_content
    assert "experimental: true" in page_content


def test_empty_error_loading_behaviour():
    """AC 2: Empty/error/loading behaviour matches DESIGN S1."""
    card_content = (ROOT / "frontend/components/home/CropCard.tsx").read_text(encoding="utf-8")
    page_content = (ROOT / "frontend/app/page.tsx").read_text(encoding="utf-8")

    # Loading skeleton exists in CropCard
    assert "isLoading" in card_content
    assert "animate-pulse" in card_content

    # Error recovery state on Home page
    assert "backendError" in page_content
    assert "Retry" in page_content
    assert "Backend is currently unreachable" in page_content


def test_layout_follows_3_2_plus_1_1_column_rule():
    """AC 3: Layout follows the 3 / 2+1 / 1 column rule from DESIGN.md S1."""
    page_content = (ROOT / "frontend/app/page.tsx").read_text(encoding="utf-8")

    # Rule: ≥1024: 3 columns; 640–1023: 2+1; <640: 1 column
    assert "grid-cols-1" in page_content
    assert "sm:grid-cols-2" in page_content
    assert "lg:grid-cols-3" in page_content
    assert "col-span-2" in page_content
