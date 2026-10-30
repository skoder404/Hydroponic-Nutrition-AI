# Contributing to Hydroponic Nutrition AI

Welcome! This document outlines our development workflow, commit conventions, pairing protocol, review checklist, and Definition of Done. Following these standards ensures consistency, traceability, and high code quality.

---

## 1. Commit-to-`main` Workflow

Our team uses a **trunk-based development workflow** directly on the `main` branch:

- **No Feature Branches:** By team decision, all work is developed and committed directly to `main`.
- **Single-Task Commits:** Exactly one task per commit. Each commit encapsulates the implementation, tests, and evidence for that specific task.
- **`git pull --rebase` Routine:**
  - Run `git pull --rebase origin main` **before starting each task**.
  - Run `git pull --rebase origin main` **before each push**.
  - On conflict: stop immediately and resolve cleanly before proceeding.
- **Never Force-Push:** Never force-push (`git push --force` or `--force-with-lease` on public history) or rewrite pushed commits.
- **Keep `main` Green:** If CI on `main` turns red, fixing the build/tests takes absolute priority over starting the next task. Never leave `main` broken.
- **Large Files:** Do not commit raw datasets, large weights, or heavy artifacts. Commit manifests, hashes, and metadata instead.

---

## 2. Conventional Commits with Task ID

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification, extended with the task ID in brackets:

```
<type>(<scope>): <subject> [T<phase>-<number>]
```

### Format Guidelines
- **Subject line:** Imperative mood, present tense, lowercase, no trailing period, $\le 72$ characters.
- **Types:**
  - `feat`: New feature, endpoint, or pipeline stage
  - `fix`: Bug fix, error correction, or patch
  - `docs`: Documentation, ADRs, or learning notes
  - `test`: Unit, integration, or verification tests
  - `refactor`: Code restructuring without changing behavior
  - `perf`: Performance optimization
  - `ci`: CI workflows or GitHub Actions updates
  - `chore` / `devops`: Build configurations, lint rules, scripts
- **Scope:** e.g., `sensor`, `image`, `seg`, `rec`, `api`, `data`, `frontend`, `devops`
- **Examples:**
  - `feat(data): audit lettuce sensor dataset splits [T2-005]`
  - `fix(seg): reject empty segmentation mask [T3-004]`
  - `docs(learning): add SVM note [T2-010]`
  - `test(api): add contract tests for recommendation endpoint [T4-002]`

### Commit Template (`.gitmessage`)
We provide a commit template at `.gitmessage`. Enable it in your local clone with:

```bash
git config commit.template .gitmessage
```

The template prompts for:
1. **What & Why:** Concise rationale and implementation details.
2. **Tests Run:** Specific test commands executed and their output status.
3. **Splits & Data Touched:** Explicit confirmation whether dataset splits or source files were modified (splits must remain frozen).
4. **Metrics:** Measured values with exact split and metric named (e.g., test macro-F1: 0.942).
5. **Limitations:** Known caveats or deferred edge cases.
6. **Learning Focus Note:** Link to the concept note in `docs/learning/` if applicable.

---

## 3. Pairing Protocol

Development follows a strict **Driver / Navigator pairing protocol**:

- **Roles Alternate per Task:** Assignments are tracked in [`todo.md`](file:///d:/Hydroponic-Nutrition-AI/todo.md).
- **Driver Responsibilities:**
  - Writes the implementation and tests.
  - Ensures all linters and test suites pass locally.
  - Commits with the task ID in brackets and pushes to `main`.
  - For tasks with a Learning Focus: gives a $\le 8$-line explanation (what, why, where used here) and 3-line plan in chat before coding.
- **Navigator Responsibilities:**
  - Reviews each task's commit(s) **after they are pushed** (`git log --grep "[T<id>]"`).
  - Verifies the review checklist (see Section 4).
  - For tasks with a Learning Focus: writes the formal concept note in `docs/learning/`.
  - Issues follow-up `fix` requests to the driver if discrepancies or defects are found.
- **Teach-Back & Review:** Team members review findings together to ensure mutual comprehension of architectural decisions and machine-learning trade-offs.

---

## 4. Navigator Review Checklist

When reviewing pushed commits for task `[T<id>]`, the navigator verifies:

1. **What / Why:** Does the commit message clearly state the rationale and technical approach?
2. **Tests Run:** Were tests or verification scripts executed and reported truthfully?
3. **Data / Splits Touched:** Were any dataset splits modified? (Verify zero data leakage, no reshuffling, frozen test sets).
4. **Metrics Reported:** If the task involves ML model evaluation, are metrics reported with the exact split name and metric name (e.g., test macro-F1, validation accuracy)?
5. **Limitations:** Are known constraints, uncertainties, or failure modes documented?
6. **Learning Note Linked:** Is the corresponding concept note present and linked in `docs/learning/`?
7. **Linter & Type Health:** Are `ruff check`, `mypy backend --ignore-missing-imports`, and all tests passing?

Any defects identified during review must be resolved via follow-up commits by the driver before moving ahead.

---

## 5. Definition of Done (DoD)

Our Definition of Done strictly enforces **Overview §27.4**:

> **Acceptance conditions (Overview §27.4):**
> *"Do not mark a feature complete until the implementation, tests, and/or artifacts demonstrate the behaviour. A UI mockup or written plan is not proof that the backend integration, model, or hardware path works."*

To consider a task or feature **DONE**:
- [x] **Implementation Exists:** Clean, readable code adhering to `docs/TECH_RULES.md`.
- [x] **Evidence Exists:** Automated verification script (`scripts/verify_*.py`), unit test (`tests/`), or generated audit report (`reports/`).
- [x] **Linters Clean:** `ruff check` and `mypy` pass with zero errors.
- [x] **No Untested Claims:** Never report tests as passing unless they were actually executed.
- [x] **Checklist Ticked in `todo.md`:** Acceptance criteria are verified and ticked using `python scripts/next_task.py --tick <task-id>`.
- [x] **Committed & Reviewed:** Committed with `[T<id>]` following Conventional Commits, pushed to `main`, and reviewed by the navigator.

---

## 6. Test Coverage Gate & Quality Standards

To prevent regression and guarantee system stability across releases:

- **Coverage Target Gate:** A minimum planned line coverage gate of **70%** (`--cov-fail-under=70`) is enforced across `backend/` and `ml/`.
- **Measured Coverage Baseline:** Real line execution is measured via `pytest --cov=backend --cov=ml`, achieving **83%** overall statement coverage across 5,838 statements.
- **Execution Command:**
  ```bash
  pytest --cov=backend --cov=ml --cov-report=term-missing tests/unit/ tests/integration/
  ```
- **Unmeasured Areas & Known Gaps:** Any components omitted from automated CI runs (e.g., physical serial UART sensors, real GPU SAM 2.1 kernels, live external LLM API billing) are explicitly documented with justifications in [`docs/testing.md`](docs/testing.md).

---

## 7. Issue Templates

Standard issue templates are maintained in `.github/ISSUE_TEMPLATE/`:
- **Bug Report (`bug_report.md`):** For reproducible defects, regressions, or unexpected behaviors.
- **Task / Feature Request (`task_or_feature.md`):** For proposed tasks, architectural spikes, and enhancements aligned with the roadmap.
- **Configuration (`config.yml`):** Disables blank issues to ensure structured tracking.
