# Ralph Task Board
Last updated: 2026-03-08

This is the live execution queue for the Ralph Circle.

Legend:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked pending human decision or external validation

## Critical path now

### P0 - Security and containment
- [ ] SEC-001 - Lock down Docker packaging
  - tighten `.dockerignore`
  - remove broad runtime `COPY . .`
  - document runtime mounts
  - validate build context no longer captures secrets or local state

### P1 - Truthfulness of exposed behavior
- [ ] COR-001 - Make the planner, registry, and router agree on executable tools
  - choose canonical executable tool catalog
  - make planner emit only supported tools
  - implement or remove misleading tool exposure
  - add end-to-end execution tests

- [ ] DOC-001 - Rewrite runtime configuration docs around JSON runtime config plus secrets-only `.env`
- [ ] DOC-002 - Rewrite Windows quickstart and startup instructions
- [ ] DOC-003 - Correct or archive stale architecture, release, and status documents

### P1 - Environment and dependency alignment
- [ ] DEP-001 - Unify dependency and packaging runtime matrix
  - align `pyproject.toml`
  - align requirements files
  - align Dockerfile
  - align validation scripts

## Near-term stability work

### P2 - Validation and test correctness
- [ ] TST-001 - Fix repo-integrity tests to compare against a baseline state
- [ ] OPS-001 - Reduce false positives in `scripts/security_audit.py`
- [ ] OPS-002 - Rewrite `scripts/validate_deployment.py` around the current runtime model

### P2 - Runtime architecture cleanup
- [ ] ARC-001 - Choose one canonical voice-loop implementation and reduce others to wrappers or remove them
- [ ] INC-002 - Either implement scheduler workflow triggering or remove the implied feature claim

## Roadmap execution after truth and stability

### Phase 4 - Notifications
- [ ] C4.5 - Minimal notification inbox UI
- [ ] C4.6 - Codex verification for notification UI
- [ ] C4.7 - Retention cleanup scheduling
- [ ] C4.8 - Codex verification for retention scheduling

### Phase 5 - Windows computer control
- [ ] C5.2b - Policy engine integration for `rex pc run`
- [ ] C5.4 - Codex verify Windows agent server if needed
- [ ] C5.5 - Windows service wrapper and boot persistence
- [ ] C5.6 - Codex verify Windows service docs and rollback

### Phase 6 - WordPress and WooCommerce
- [ ] C6.1 - Read-only monitoring
- [ ] C6.2 - Codex verify read-only monitoring
- [ ] C6.3 - Approval-gated write actions
- [ ] C6.4 - Codex verify write actions

### Phase 7 - Voice identity MVP
- [ ] C7.1 - Enrollment and calibration
- [ ] C7.2 - Codex verify voice identity MVP

### Phase 8 - Real-time notifications and follow-ups
- [ ] C8.1 - SSE push for notifications
- [ ] C8.2 - Codex verify SSE auth and stability
- [ ] C8.3 - Optional follow-up hardening

## Quality debt lane

- [ ] QLT-001A - stage 1 ruff cleanup for touched core files only
- [ ] QLT-001B - stage 2 black normalization for touched core files only
- [ ] QLT-001C - stage 3 mypy cleanup in highest-risk modules
- [ ] DX-001 - clean up repo root and archive historical reports

## Human decision queue

- [ ] Decide canonical default branch strategy: `master` vs `main`
- [ ] Decide canonical voice-loop implementation
- [ ] Decide whether scheduler workflow execution should be real or explicitly de-scoped
- [ ] Decide whether replay should be implemented or removed from active claims

## Current batch target

Batch ID: BATCH-001

Objective:
- complete SEC-001
- complete DOC-001
- complete DOC-002
- make no unrelated code changes

Validation required:
- docker build context review
- targeted tests if docs or scripts change
- security audit if Docker artifacts change
- final summary with changed files and exact commands run

Stop conditions:
- if Docker packaging changes reveal dependency matrix conflicts broader than SEC-001
- if docs cannot be corrected without first choosing unresolved branch or entrypoint strategy
- if any proposed fix requires silently changing actual runtime behavior
