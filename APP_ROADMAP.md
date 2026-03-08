# Rex AI Assistant Application Roadmap
Last updated: 2026-03-08

This roadmap is the execution compass for the Ralph Circle.
It is grounded in:
- CODEX_REPO_AUDIT.md
- CODEX_REPO_AUDIT_ISSUES.json
- ROADMAP_BIBLE_UPDATED_2026-02-24_v2.md
- CLAUDE.md

## Project reality

Rex is a large, real assistant platform with meaningful code, a broad CLI surface, many integrations, and a strong test suite.
It is not production-ready yet.

The highest-value truths from the current audit are:

1. The repo is much more real than a toy, but still prototype-grade.
2. The planner, registry, and router do not currently agree on what is executable.
3. Packaging and deployment are not safe enough yet.
4. Documentation drifts from executable reality.
5. Quality gates and deployment validation are not green.

## Global definition of done

The project is only considered done when all of the following are true:

- Install and setup paths are accurate and documented
- Critical audit issues are resolved or explicitly de-scoped truthfully
- Tests pass reliably without brittle integrity assumptions
- The planner and execution surface agree on what is real
- Security and deployment tooling are trustworthy
- Runtime modes and entrypoints are clearly documented
- Docs match code
- Roadmap phases through current target completion are closed and verified
- The task board contains no unresolved work items

## Execution order

The Ralph Circle must use this order:

1. Truth and containment
2. Execution-surface correctness
3. Dependency and validation alignment
4. Entry surface consolidation
5. Roadmap feature phases
6. Quality debt paydown

## Phase 0: Truth and containment

Goal:
Stop unsafe packaging and stop misleading claims.

Primary issues:
- SEC-001
- DOC-001
- DOC-002
- DOC-003

Tasks:
- Fix Docker build context exposure
- Standardize runtime-config documentation
- Rewrite Windows quickstart to reflect real entrypoints
- Archive, relabel, or correct stale architecture and status docs
- Align branch strategy docs with automation reality

Exit criteria:
- Docker build context no longer captures local secrets or runtime state
- Setup docs consistently describe JSON runtime config plus secrets-only `.env`
- Windows guide correctly distinguishes text chat, voice loop, dashboard, and TTS API
- Stale “production-ready” or false status docs are corrected or archived

## Phase 1: Execution-surface correctness

Goal:
Make the autonomous surface truthful.

Primary issues:
- COR-001
- INC-002

Tasks:
- Define one authoritative executable tool catalog
- Ensure Planner emits only executable tools
- Implement or de-scope misleadingly exposed tools
- Clarify scheduler workflow-triggering status
- Add integration tests proving planner-emitted tools execute end to end

Exit criteria:
- Every planner-emitted tool is truly executable or removed from that path
- Tool docs and autonomous docs match the real execution surface
- Scheduler workflow execution is either implemented or truthfully documented as not supported

## Phase 2: Dependency and validation alignment

Goal:
Make install and validation coherent.

Primary issues:
- DEP-001
- TST-001
- OPS-001
- OPS-002

Tasks:
- Define one supported runtime matrix
- Align requirements files, pyproject, Dockerfile, and validation scripts
- Fix brittle repo-integrity tests
- Reduce false positives in the security audit script
- Rewrite deployment validation around current reality

Exit criteria:
- Dependency matrix is coherent
- Validation scripts reflect the current runtime model
- Integrity tests compare against a session baseline instead of pre-existing dirtiness
- Security audit output is useful, not noisy

## Phase 3: Runtime surface consolidation

Goal:
Reduce entrypoint drift and duplicated runtime surfaces.

Primary issues:
- ARC-001

Tasks:
- Choose and document canonical runtime modes
- Reduce duplicated voice-loop surfaces
- Clarify official startup paths
- Ensure config loading happens once, early, and consistently

Exit criteria:
- One canonical voice runtime path is documented
- Runtime modes are clearly separated
- Startup docs are truthful and minimal

## Phase 4: Productionize notifications

Source:
ROADMAP_BIBLE_UPDATED_2026-02-24_v2.md

Cycles:
- 4.5 Minimal notification inbox UI
- 4.6 Codex verify UI/API/persistence/security
- 4.7 Retention cleanup scheduling
- 4.8 Codex verify scheduling correctness and idempotency

Exit criteria:
- Notification UI exists and works with real and stub stores
- Mark read and mark all read work
- Retention cleanup is wired safely and idempotently
- Verification reports are complete

## Phase 5: Windows computer control hardening

Source:
ROADMAP_BIBLE_UPDATED_2026-02-24_v2.md

Cycles:
- 5.2b Policy engine integration for `rex pc run`
- 5.4 Codex verify agent server if needed
- 5.5 Windows service or scheduled-task wrapper and boot persistence
- 5.6 Codex verify install docs, safety defaults, rollback

Exit criteria:
- `rex pc run` flows through policy approval
- Allowlist rules still block denied commands
- Windows agent service-wrapper docs are real and tested
- Verification reports are complete

## Phase 6: WordPress and WooCommerce integration

Source:
ROADMAP_BIBLE_UPDATED_2026-02-24_v2.md

Cycles:
- 6.1 Read-only monitoring
- 6.2 Codex verify security, pagination, error handling
- 6.3 Write actions with policy approval
- 6.4 Codex verify approval gating and audit logs

Exit criteria:
- WordPress health monitoring works
- WooCommerce read paths work
- Write actions are approval-gated
- Verification reports are complete

## Phase 7: Voice identity MVP

Source:
ROADMAP_BIBLE_UPDATED_2026-02-24_v2.md

Cycles:
- 7.1 Enrollment and calibration
- 7.2 Codex verify safety and fallback UX

Exit criteria:
- Enrollment CLI exists
- Embedding persistence works
- Threshold calibration works
- Unknown-speaker behavior is defined
- Verification report is complete

## Phase 8: Real-time notifications and remaining hardening

Source:
ROADMAP_BIBLE_UPDATED_2026-02-24_v2.md

Cycles:
- 8.1 SSE-based real-time push
- 8.2 Codex verify SSE auth and stability
- 8.3 Optional follow-ups

Exit criteria:
- Authenticated SSE path works
- UI subscribes and updates correctly
- Verification report is complete

## Phase 9: Repo hygiene and debt paydown

Primary issues:
- QLT-001

Tasks:
- Stage 1: touched-file Ruff cleanup
- Stage 2: touched-file Black normalization
- Stage 3: highest-risk mypy cleanup
- Keep debt paydown isolated from feature work

Exit criteria:
- Touched-file quality gates are trustworthy
- No accidental behavioral changes are hidden inside cleanup work

## Operating policy

The Ralph Circle must always prefer:
- truthful narrowing over inflated capability claims
- bounded batches over broad rewrites
- verification over optimism
- root-cause fixes over cosmetic suppression
