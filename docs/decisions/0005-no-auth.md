# ADR 0005: No Authentication in v1 Scope

- **Status:** Accepted
- **Date:** 2026-10-25
- **Driver:** Sushil
- **Navigator:** Yashwant
- **Task ID:** T3-001
- **References:** `PROJECT_OVERVIEW.md` §2.3, §23; `docs/PRD.md` Non-Functional Requirements; `docs/TECH_RULES.md` Security Rules.

---

## 1. Context & Problem Statement

Hydroponic Nutrition AI (v1) is designed as a focused academic and laboratory decision-support application evaluated in local development and demonstration environments on campus. 

Introducing user accounts, passwords, session tokens, JWTs, role-based access control (RBAC), and authentication middleware into v1 would introduce significant architectural complexity, credential management overhead, and potential security vulnerabilities (e.g., credential stuffing, token leaks, session hijacking) without providing meaningful defensive value for a local, single-operator demo system.

We must define the explicit security boundaries and capability restrictions that make running without HTTP authentication safe.

---

## 2. Decision: Omit Authentication in v1

We intentionally omit user login, account registration, and authentication tokens from v1.

Instead of authenticating users, the system enforces **strict capability withholding** at the API layer. The application is designed such that unauthenticated HTTP clients possess only read-only or transient analytical capabilities.

---

## 3. Withheld Capabilities Over HTTP

To ensure that the unauthenticated API cannot be abused to compromise the system, the server, or the machine learning artifacts, the following capabilities are **strictly withheld over HTTP**:

1. **No Model Promotion or Rollback:**
   - Model promotion, tagging, and rollback actions are restricted entirely to internal CLI scripts (e.g., `scripts/promote_model.py`).
   - No HTTP endpoint exists to modify active model weights, swap registry pointers, or deploy new checkpoints.
2. **No Dataset Writes or Modification:**
   - Raw datasets (`image_data/`, `sensor_data/`) and processed splits (`processed_data/`) are strictly read-only to HTTP requests.
   - No HTTP endpoint permits uploading, modifying, deleting, or re-partitioning datasets.
3. **No Code Execution:**
   - No endpoint allows executing user-supplied Python code, shell commands, or arbitrary dynamic expressions.
   - LLM (Gemini) outputs are treated as untrusted text strings and parsed strictly against static schemas; they are never passed to `eval()` or executed as code.
4. **No Arbitrary File System Writes:**
   - Image upload endpoints do not write to client-controlled paths; uploaded files receive server-generated UUIDs and are validated for magic bytes, MIME types, and PIL/OpenCV decodability.
5. **No Actuator or Hardware Control:**
   - The application is purely advisory (decision support). No endpoint controls physical hardware, nutrient pumps, valves, or lighting.

---

## 4. Triggers for a New Threat Model & Authentication Requirement

Authentication and authorization must not be added incrementally without a comprehensive security redesign. Any of the following conditions **triggers the requirement for a new formal threat model and mandatory authentication**:

1. **Multi-Tenancy or Public Deployment:**
   - Deploying the web application to a public, internet-facing domain outside a trusted local network or localhost.
2. **Multi-User Private Data Isolation:**
   - Storing user-specific crop diagnosis history, farm records, or grower identities across multiple separate individuals or organizations.
3. **HTTP-Exposed Administrative Operations:**
   - Introducing web endpoints for model management, configuration tuning, or server restarts.
4. **Actuator or Control Loop Integration:**
   - Integrating closed-loop nutrient dosing pumps, pH adjustment relays, or greenhouse environmental controls.
5. **Paid or Rate-Restricted Multi-User API Access:**
   - Providing API access with per-user billing, quota tracking, or commercial tenancy.

---

## 5. Consequences

### Positive
- Minimizes architectural complexity and attack surface for v1.
- Prevents false senses of security from naive custom authentication implementations.
- Enforces security by design through capability restrictions rather than relying solely on access credentials.

### Negative / Trade-offs
- The server must only be bound to trusted interfaces (e.g., `127.0.0.1` or firewalled campus network).
- All users accessing the frontend share the local session history and diagnosis log.
