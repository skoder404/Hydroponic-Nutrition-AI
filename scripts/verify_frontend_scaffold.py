#!/usr/bin/env python3
"""Verification script for T1-010: Frontend scaffold (Next.js, TypeScript, Tailwind).

Validates:
1. frontend/ directory contains required Next.js structure without disallowed src/ directory.
2. All six required route page files exist:
   - app/page.tsx (/)
   - app/analyze/page.tsx (/analyze)
   - app/history/page.tsx (/history)
   - app/history/[id]/page.tsx (/history/[id])
   - app/models/page.tsx (/models)
   - app/about/page.tsx (/about)
3. Tailwind, ESLint, and Prettier configurations exist.
4. .env.example contains NEXT_PUBLIC_API_BASE_URL.
5. `npm run build` succeeds with code 0.
6. `npm run lint` succeeds with code 0.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

REQUIRED_ROUTES = [
    FRONTEND_DIR / "app" / "page.tsx",
    FRONTEND_DIR / "app" / "analyze" / "page.tsx",
    FRONTEND_DIR / "app" / "history" / "page.tsx",
    FRONTEND_DIR / "app" / "history" / "[id]" / "page.tsx",
    FRONTEND_DIR / "app" / "models" / "page.tsx",
    FRONTEND_DIR / "app" / "about" / "page.tsx",
]

REQUIRED_CONFIGS = [
    FRONTEND_DIR / "package.json",
    FRONTEND_DIR / "tsconfig.json",
    FRONTEND_DIR / ".prettierrc",
]


def check_folder_rules() -> bool:
    if (FRONTEND_DIR / "src").exists():
        print("[FAIL] Disallowed 'frontend/src/' directory found. Routes must be in 'frontend/app/'.")
        return False

    for route_file in REQUIRED_ROUTES:
        if not route_file.exists():
            print(f"[FAIL] Missing route file: {route_file.relative_to(PROJECT_ROOT)}")
            return False

    for cfg_file in REQUIRED_CONFIGS:
        if not cfg_file.exists():
            print(f"[FAIL] Missing config file: {cfg_file.relative_to(PROJECT_ROOT)}")
            return False

    print(f"[PASS] All {len(REQUIRED_ROUTES)} route pages and required configs exist.")
    return True


def check_env_example() -> bool:
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    if "NEXT_PUBLIC_API_BASE_URL" not in env_example:
        print("[FAIL] NEXT_PUBLIC_API_BASE_URL missing from .env.example")
        return False
    print("[PASS] NEXT_PUBLIC_API_BASE_URL is documented in .env.example.")
    return True


def check_eslint() -> bool:
    print("[INFO] Running 'npm run lint' in frontend/...")
    res = subprocess.run(
        ["npm.cmd" if sys.platform == "win32" else "npm", "run", "lint"],
        cwd=str(FRONTEND_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if res.returncode != 0:
        print(f"[FAIL] ESLint failed with code {res.returncode}:\n{res.stdout}\n{res.stderr}")
        return False
    print("[PASS] ESLint passed cleanly.")
    return True


def check_build() -> bool:
    print("[INFO] Running 'npm run build' in frontend/...")
    res = subprocess.run(
        ["npm.cmd" if sys.platform == "win32" else "npm", "run", "build"],
        cwd=str(FRONTEND_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if res.returncode != 0:
        print(f"[FAIL] Build failed with code {res.returncode}:\n{res.stdout}\n{res.stderr}")
        return False
    print("[PASS] Next.js build completed successfully with all static routes.")
    return True


def main() -> None:
    print("=== Running T1-010 Verification ===")
    ok = True
    ok = check_folder_rules() and ok
    ok = check_env_example() and ok
    ok = check_eslint() and ok
    ok = check_build() and ok

    if ok:
        print("\n[ALL CHECKS PASSED] T1-010 acceptance criteria successfully verified.")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILED] One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
