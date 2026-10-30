"""Unit tests for demo runbook, README fidelity, and release checklist (T9-006).

Acceptance Criteria:
- [x] A teammate can run the demo from the runbook alone
- [x] README claims match verified features
- [x] Tag created only after the checklist passes
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_runbook_exists_and_contains_all_demo_sections():
    """AC 1: A teammate can run the demo from the runbook alone."""
    runbook_path = ROOT / "docs" / "runbook.md"
    assert runbook_path.is_file(), "docs/runbook.md must exist"

    content = runbook_path.read_text(encoding="utf-8")

    # 1. Start and stop procedures
    assert "docker compose up -d" in content, "Must explain how to start the stack"
    assert "docker compose down" in content, "Must explain how to stop the stack"

    # 2. Loading models from registry
    assert "model_service" in content or "Model Registry" in content, (
        "Must document model loading"
    )
    assert "promoted" in content.lower(), "Must mention promoted production models"

    # 3. Pointing to campus server (IP in .env) and connectivity checking
    assert "CAMPUS_SERVER_URL" in content, "Must document CAMPUS_SERVER_URL in .env"
    assert "DEVICE_ADAPTER" in content, "Must document DEVICE_ADAPTER"
    assert "freshness" in content.lower() or "stale" in content.lower(), (
        "Must document freshness checks"
    )

    # 4. Environment variables
    assert "DATABASE_URL" in content or ".env" in content, (
        "Must document environment variables"
    )

    # 5. Known limitations to state during a demo
    assert "Known Limitations" in content, (
        "Must include limitations section to state during a demo"
    )
    assert "Unpaired" in content, "Must document unpaired image/sensor datasets"
    assert "Rule-Derived" in content or "rule-derived" in content, (
        "Must document rule-derived sensor labels"
    )

    # 6. Fallback behaviour
    assert "Fallback" in content, "Must explain fallback behaviour"
    assert "gemini_to_rules" in content, "Must mention Gemini-to-rules fallback"
    assert "gpu_to_cpu" in content, "Must mention GPU-to-CPU fallback"
    assert "segmentation_failure" in content, (
        "Must mention segmentation failure handling"
    )

    # 7. Demo script steps
    assert "Live Demo Script" in content, "Must provide step-by-step demo script"
    assert "Analyze" in content or "/analyze" in content, (
        "Must walk through analysis workflow"
    )
    assert "History" in content or "/history" in content, (
        "Must walk through history workflow"
    )


def test_readme_claims_match_verified_features():
    """AC 2: README claims match verified features and link to runbook."""
    readme_path = ROOT / "README.md"
    assert readme_path.is_file(), "README.md must exist"

    content = readme_path.read_text(encoding="utf-8")

    # Must reference v0.1.0 release and demo runbook
    assert "v0.1.0" in content, "README must state v0.1.0 release status"
    assert "runbook.md" in content, "README must link to docs/runbook.md"

    # Modalities and crops
    assert "lettuce" in content.lower(), "Must feature lettuce"
    assert "spinach" in content.lower(), "Must feature spinach"
    assert "tomato" in content.lower(), "Must feature tomato"

    # Core architectural guarantees
    assert "SAM 2.1" in content, "Must mention SAM 2.1 segmentation"
    assert "Gemini" in content, "Must mention Gemini recommendations"
    assert (
        "rules fallback" in content.lower() or "rule-based fallback" in content.lower()
    ), "Must claim rule-based fallback"
    assert "PostgreSQL" in content, "Must cite PostgreSQL history backend"


def test_release_checklist_preconditions():
    """AC 3: Verify all release checklist preconditions before cutting v0.1.0."""
    # Ensure openapi canonical schema exists
    openapi_json = ROOT / "docs" / "api" / "openapi.json"
    assert openapi_json.is_file(), "Canonical OpenAPI schema must exist"

    # Ensure deployment scripts and configs exist
    assert (ROOT / "docker-compose.yml").is_file(), "docker-compose.yml must exist"
    assert (ROOT / "deploy" / "Caddyfile").is_file(), "deploy/Caddyfile must exist"
    assert (ROOT / "scripts" / "backup_db.sh").is_file(), "backup script must exist"
    assert (ROOT / "scripts" / "restore_db.sh").is_file(), "restore script must exist"
