# Ralph Task Board
Last updated: 2026-03-08

Legend:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked pending human decision, external dependency, or live verification

## Current critical path

### P0
- [ ] SEC-001 - Docker build context can capture secrets and local state

### P1
- [ ] DOC-001 - Runtime configuration documentation is inconsistent with the code
- [ ] DOC-002 - Windows quickstart and startup instructions are wrong
- [ ] DOC-003 - Architecture, release, and status documents are stale or false
- [ ] COR-001 - Planner, registry, and router disagree on executable tools
- [ ] DEP-001 - Dependency and packaging artifacts disagree on the supported runtime matrix

### P2
- [ ] TST-001 - Repo-integrity tests are brittle and fail on pre-existing dirtiness
- [ ] OPS-001 - Security audit script produces high false-positive noise
- [ ] OPS-002 - Deployment validation script is stale and contradicts the current repo
- [ ] ARC-001 - Voice loop and root-level entry surfaces are duplicated and drifting

## Roadmap phases after critical truth work

### Phase 4
- [ ] C4.5 - Minimal notification inbox UI
- [ ] C4.6 - Verify UI, API, persistence, and security
- [ ] C4.7 - Retention cleanup scheduling
- [ ] C4.8 - Verify scheduling correctness and idempotency

### Phase 5
- [ ] C5.2b - Policy engine integration for `rex pc run`
- [ ] C5.4 - Verify Windows agent server if needed
- [ ] C5.5 - Windows service or scheduled-task wrapper and boot persistence
- [ ] C5.6 - Verify install docs, safety defaults, and rollback steps

### Phase 6
- [ ] C6.1 - WordPress and WooCommerce read-only monitoring
- [ ] C6.2 - Verify security, pagination, and error handling
- [ ] C6.3 - Write actions with policy approval
- [ ] C6.4 - Verify approval gating and audit logs

### Phase 7
- [ ] C7.1 - Voice identity enrollment and calibration
- [ ] C7.2 - Verify safety, false-positive controls, and fallback UX

### Phase 8
- [ ] C8.1 - Real-time push with SSE
- [ ] C8.2 - Verify SSE auth and stability
- [ ] C8.3 - Optional remaining hardening

### Phase 9
- [ ] QLT-001A - Touched-file Ruff cleanup
- [ ] QLT-001B - Touched-file Black normalization
- [ ] QLT-001C - Highest-risk mypy cleanup

## Human decision queue

- [ ] Decide canonical branch strategy: `master` or `main`
- [ ] Decide canonical voice-loop surface
- [ ] Decide whether scheduler-triggered workflow execution becomes real or is formally de-scoped
- [ ] Decide whether replay remains stub-only or becomes a real recovery feature

## Current batch target

- [ ] BATCH-001 - Resolve SEC-001, DOC-001, and DOC-002 only

## Loop stop conditions

The autonomous runner must stop if:
- any task is marked `[!]`
- the primary agent exits non-zero
- the verifier exits non-zero
- the task board stops changing for too many iterations
- git backup fails
- max iterations is reached
