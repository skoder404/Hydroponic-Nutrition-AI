# ADR 0004: Google Gemini API Configuration and Fallback Architecture

- **Status:** Accepted
- **Date:** 2026-10-25
- **Driver:** Yashwant
- **Navigator:** Sushil
- **Task ID:** T1-008
- **References:** `PROJECT_OVERVIEW.md` §2.3, §18; `docs/PRD.md` Non-Functional Requirements; `docs/TECH_RULES.md` Tech Stack & Verify at Setup §3.

---

## 1. Context & Problem Statement

Hydroponic Nutrition AI provides advisory recommendations when sensor telemetry indicates environmental anomalies (e.g., pH imbalance, EC out of range, water temperature spikes) or computer vision flags nutrient deficiencies on lettuce, spinach, or tomato crops.

The system utilizes an LLM to generate clear, grower-friendly agronomic guidance while mandating a deterministic rule-based provider as an automatic fallback when network connectivity is absent, API quota is exceeded, or the API key is not configured.

We must define:
1. The supported Python SDK and integration interface.
2. The default model selection (`gemini-1.5-flash` vs. higher models) and rationale.
3. The secret isolation and verification status.

---

## 2. Decision: SDK & Model Selection

### 2.1 Python SDK: `google-genai`
- **Primary SDK:** Modern Google GenAI SDK (`google-genai >= 2.33.0`), invoked via `from google import genai; client = genai.Client(api_key=...)`.
- **Legacy Compatibility:** Fallback support for `google-generativeai` is retained during migration.
- **Dependency Location:** Added to `requirements/backend.txt` and `requirements/backend.in`.

### 2.2 Default Model: `gemini-1.5-flash`
- **Default Config:** `GEMINI_MODEL="gemini-1.5-flash"`.
- **Configurability:** Fully configurable via the `GEMINI_MODEL` environment variable (never hardcoded, per TECH_RULES §3).
- **Rationale for `gemini-1.5-flash` over higher models (e.g., Gemini 1.5 Pro, Gemini 2.5 Pro):**
  1. **Latency & Interactivity:** `gemini-1.5-flash` yields sub-second to ~1.5-second response times, essential for interactive web dashboards, whereas Pro/Ultra models introduce multi-second latencies.
  2. **Free-Tier Quota & Rate Limits:** Flash provides generous rate limits (15 RPM, 1,500 RPD on Google AI Studio free tier), preventing quota exhaustion during local developer testing and student demonstration.
  3. **Task Scope & Compactness:** The recommendation task produces 2–3 structured sentences summarizing the anomaly and recommending corrective dosing checks. Large reasoning models (Pro/Ultra) are unnecessary and over-engineered for bounded tabular/CV summaries.
  4. **Cost & Reliability:** Minimizes API timeouts and token costs, aligning with the project's graceful degradation policy.

---

## 3. Secret Isolation and Security Boundaries

1. **Local-Only Credential:** `GEMINI_API_KEY` is loaded exclusively from `.env` or system environment variables. `.env` is strictly excluded in `.gitignore`.
2. **Redaction Guarantee:** `backend/app/core/config.py` wraps the key in Pydantic's `SecretStr` and redacts any key field to `**********` in `__repr__` and `__str__`.
3. **No Key in Logs or Output:** `scripts/gemini_smoke_test.py` enforces regex and substring scrubbing on all exception messages and standard outputs.
4. **Untrusted Output Policy:** Output from Gemini is treated as untrusted external text. It is parsed strictly against Pydantic schemas before reaching the client and cannot execute commands or alter system state (ADR 0005).

---

## 4. Verification Evidence & Live Call Status

- **Smoke Test Tool:** `scripts/gemini_smoke_test.py`.
- **Unconfigured / Missing Key Verification:**
  - When `GEMINI_API_KEY` is absent or set to empty/placeholder, the smoke test outputs:
    ```text
    [INFO] GEMINI_API_KEY is not configured.
    [INFO] Local rule-based recommendation fallback remains active as default.
    ```
  - The script terminates cleanly with **exit code 0**, confirming that local rule-based recommendations function as the default operational path.
- **Secret Redaction Verification:**
  - Tested with invalid and probe tokens; verified error payloads are sanitized and raw keys are never leaked to logs or console.
- **Live Call Verification Status:**
  - In the repository baseline environment without an external Google AI Studio API key provisioned in `.env`, the live online call could not be verified; the local rule-based recommendation provider remains active, verified, and authoritative.
