# Assignment 002 — Audit-to-backlog reconciliation

Read and obey `CLAUDE.md` first. Do not perform another broad repository audit.

Use these existing inputs:

- `docs/planning/CODEX_CURRENT_STATE_AUDIT.md`
- `docs/planning/source-of-truth/REX_Unified_Build_Spec_UPDATED.md`
- `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md`
- `docs/planning/TEAM_LEAD_OPERATING_RULES.md`
- `docs/planning/HISTORICAL_PRD_RECONCILIATION.md` if present

The Codex audit is the current-state evidence base. Check source only when a specific high-impact claim needs confirmation; do not narrate or reread the full repositories.

Goal: convert the evidence and requirements into one concise, implementation-ready delivery backlog for shippable desktop and private mobile AskRex products.

## Deliverable

Create `docs/planning/ASKREX_UNIFIED_DELIVERY_BACKLOG.md` with:

1. A requirement traceability summary for both authoritative documents.
2. A compact capability matrix: capability, verified state, desktop UI/settings, mobile state, security/permissions, evidence, and release gap.
3. A dependency-ordered roadmap of 20–35 atomic stories grouped into no more than 10 implementation batches.
4. Every story must include ID, priority, exact files/areas, evidence/root cause, implementation steps, tests, validation commands, and definition of done.
5. Explicit coverage of truthful surfaces, runtime/data/credential isolation, pairing/capability broker, intelligence/context/model fallback, voice, OpenClaw, design parity, packaging/signing, and private mobile distribution.
6. A first-cycle selection of no more than five stories that can start immediately.

Preserve verified positive foundations. Do not mark historical checkboxes as proof. Label external/hardware dependencies. Do not modify product code or the mobile repository.

Commit only this assignment and the backlog using Conventional Commits. Keep the response concise and stay within the configured budget.