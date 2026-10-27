#!/usr/bin/env python3
"""CLI utility for model artifact promotion and rollback (T4-009).

Adheres to TECH_RULES and Overview Section 24.2:
- Model promotion is an explicit, audit-trailed command (never 'newest wins').
- Promotion is CLI-only.
- Rollback cleanly restores the previous known-good model version.

Usage:
    python scripts/promote_model.py promote --crop lettuce --modality sensor --version 1.0.0
    python scripts/promote_model.py rollback --crop lettuce --modality sensor
    python scripts/promote_model.py show --crop lettuce --modality sensor
    python scripts/promote_model.py list --crop lettuce --modality sensor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.registry import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model artifact promotion and rollback CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: promote
    promote_parser = subparsers.add_parser("promote", help="Promote a model version to active.")
    promote_parser.add_argument("--crop", required=True, choices=["lettuce", "spinach", "tomato"], help="Crop family")
    promote_parser.add_argument("--modality", required=True, choices=["image", "sensor", "feature"], help="Input modality")
    promote_parser.add_argument("--version", required=True, help="Semantic version to promote (e.g. 1.0.0)")
    promote_parser.add_argument("--root", default=None, help="Optional model root directory override")

    # Subcommand: rollback
    rollback_parser = subparsers.add_parser("rollback", help="Roll back to the previous known-good model version.")
    rollback_parser.add_argument("--crop", required=True, choices=["lettuce", "spinach", "tomato"], help="Crop family")
    rollback_parser.add_argument("--modality", required=True, choices=["image", "sensor", "feature"], help="Input modality")
    rollback_parser.add_argument("--root", default=None, help="Optional model root directory override")

    # Subcommand: show
    show_parser = subparsers.add_parser("show", help="Display the active promoted model.")
    show_parser.add_argument("--crop", required=True, choices=["lettuce", "spinach", "tomato"], help="Crop family")
    show_parser.add_argument("--modality", required=True, choices=["image", "sensor", "feature"], help="Input modality")
    show_parser.add_argument("--root", default=None, help="Optional model root directory override")

    # Subcommand: list
    list_parser = subparsers.add_parser("list", help="List all registered versions for a crop/modality.")
    list_parser.add_argument("--crop", required=True, choices=["lettuce", "spinach", "tomato"], help="Crop family")
    list_parser.add_argument("--modality", required=True, choices=["image", "sensor", "feature"], help="Input modality")
    list_parser.add_argument("--root", default=None, help="Optional model root directory override")

    args = parser.parse_args(argv)
    registry = ModelRegistry(root_dir=args.root)

    try:
        if args.command == "promote":
            res = registry.promote(
                crop=args.crop,
                modality=args.modality,
                version=args.version,
                called_from_cli=True,
            )
            print(f"[PROMOTED] {args.crop}/{args.modality} active version set to v{res['active_version']}")
            print(f"Previous version: {res.get('previous_version') or 'None'}")
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "rollback":
            res = registry.rollback(
                crop=args.crop,
                modality=args.modality,
                called_from_cli=True,
            )
            print(f"[ROLLED_BACK] {args.crop}/{args.modality} active version restored to v{res['active_version']}")
            print(f"Previous version: {res.get('previous_version')}")
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "show":
            res = registry.get_active_pointer(crop=args.crop, modality=args.modality)
            print(f"Active model for {args.crop}/{args.modality}:")
            print(json.dumps(res, indent=2))
            return 0

        elif args.command == "list":
            versions = registry.list_versions(crop=args.crop, modality=args.modality)
            print(f"Registered versions for {args.crop}/{args.modality}:")
            for v in versions:
                print(f"  - v{v}")
            return 0

    except (ValueError, RuntimeError, FileNotFoundError, PermissionError, OSError) as exc:
        print(f"Error executing '{args.command}': {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
