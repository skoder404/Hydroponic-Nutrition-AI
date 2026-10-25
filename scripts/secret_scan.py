#!/usr/bin/env python3
"""Secret scanner for pre-commit and CI.

Scans files for committed secrets, API keys, private keys, and tokens.
Exits with code 1 if any unredacted secret is detected.

Re-run command:
    python scripts/secret_scan.py [files...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Safe placeholder tokens that are allowed in documentation or examples
ALLOWLIST = {
    "your_gemini_api_key_here",
    "your_postgres_password_here",
    "YOUR_MYSQL_ROOT_PASSWORD",
    "**********",
    "<REDACTED>",
    "<REDACTED_IP>",
    "AIzaSyDfakeGeminiKeyForTesting1234567890",  # Test-only fake key in verify scripts
    "AIzaSyFakeSecretKeyForTestingRedaction123",  # Test-only fake key in verify scripts
}

# Regex patterns for common secrets and API keys
SECRET_PATTERNS = [
    (re.compile(r"(?:AIzaSy[0-9A-Za-z_-]{33})"), "Google API Key (AIzaSy...)"),
    (re.compile(r"(?:sk-[a-zA-Z0-9]{32,})"), "OpenAI Secret Key (sk-...)"),
    (re.compile(r"(?:AKIA[0-9A-Z]{16})"), "AWS Access Key ID (AKIA...)"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), "Private Key Header"),
    (re.compile(r'(?i)(?:api[_-]?key|secret[_-]?key|auth[_-]?token)\s*[:=]\s*["\']([a-zA-Z0-9_\-\.]{20,})["\']'), "API Key Assignment"),
    (re.compile(r'(?i)(?:password|passwd|pwd)\s*[:=]\s*["\']([^"\'\s]{8,})["\']'), "Password Assignment"),
]

# Files/folders to skip
SKIP_PATHS = {
    Path(".env"),
    Path(".env.example"),
    Path(".env.local"),
}


def is_skipped(path: Path) -> bool:
    name = path.name
    if name in {".env", ".env.local"} or name.endswith((".log", ".pt")):
        return True
    parts = path.parts
    return any(p in {".git", ".next", "node_modules", ".venv", "venv", "__pycache__"} for p in parts)


def scan_file(path: Path) -> list[str]:
    violations: list[str] = []
    if is_skipped(path):
        return violations

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return violations

    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        # Skip comment explanations in docs or tests that explain patterns
        if "re.compile" in line or "SECRET_PATTERNS" in line:
            continue

        for pattern, label in SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                matched_str = match.group(0)
                # Check allowlist
                if any(allowed in matched_str for allowed in ALLOWLIST):
                    continue
                # In example files, allow explicit 'your_' placeholders
                if "your_" in matched_str.lower() or "example" in matched_str.lower() or "dummy" in matched_str.lower():
                    continue
                # Skip obviously fake test keys (contain 'fake', 'test', or 'mock')
                if any(tok in matched_str.lower() for tok in ("fake", "test", "mock")):
                    continue

                # Redact matched text for safe display
                redacted = matched_str[:4] + "..." + matched_str[-4:] if len(matched_str) > 8 else "***"
                violations.append(f"{path}:{idx}: Detected {label} -> {redacted}")

    return violations


def main(argv: list[str] | None = None) -> int:
    files_to_scan: list[Path] = []
    args = argv if argv is not None else sys.argv[1:]

    if args:
        files_to_scan = [Path(p) for p in args if Path(p).is_file()]
    else:
        # Scan entire repo if no files passed
        root = Path(__file__).resolve().parent.parent
        for ext in ["*.py", "*.ts", "*.tsx", "*.js", "*.json", "*.yml", "*.yaml", "*.md", "*.env*"]:
            files_to_scan.extend(root.rglob(ext))

    all_violations: list[str] = []
    for f in sorted(set(files_to_scan)):
        all_violations.extend(scan_file(f))

    if all_violations:
        print("=" * 60)
        print("SECRET SCANNER: Potential secrets or API keys detected!")
        print("=" * 60)
        for v in all_violations:
            print(f"  [X] {v}")
        print("\nPlease remove or redact secrets before committing.")
        return 1

    print("Secret scan passed: No unredacted secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
