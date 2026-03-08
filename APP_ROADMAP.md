# Rex AI Assistant Application Roadmap
Last updated: 2026-03-08

This file is the execution roadmap for the Ralph Circle. It merges the current Codex audit with the existing Rex roadmap so the agent works from repo truth, not wishful thinking.

## Current reality

Rex is not production-ready yet.

Right now the repo is best described as:
- a large prototype
- with several real or near-real subsystems
- but with important truth, packaging, validation, and documentation drift

Current hard truths from the audit:
- Docker packaging is unsafe as written and can capture secrets or local state
- planner, registry, and router do not agree on what tools are actually executable
- dependency and packaging artifacts disagree about the supported runtime matrix
- quality gates are not green
- documentation does not fully match executable reality

The Ralph Circle must treat those items as first-order work, not side quests.

## Definition of done for the whole project

Rex is only considered complete when all of the following are true:

- clean install path works and is documented
- tests pass reliably
- one real smoke path works end to end
- security and validation tooling are trustworthy
- docs match reality
- runtime modes are clear and stable
- the voice loop is reliable on the intended machine
- the supported tool and automation surface is truthful
- packaging and deployment are safe
- no major subsystem is claiming capabilities it does not actually provide

## Phase 0 - Truth and containment

Goal:
Stop unsafe packaging and stop misleading users about what the repo can currently do.

Priority issues:
- SEC-001
- DOC-001
- DOC-002
- DOC-003

Tasks:
- tighten `.dockerignore`
- replace broad Docker `COPY` patterns with an allowlist approach
- fix README and Windows docs so they match the JSON config plus secrets-only `.env` model
- archive or relabel stale status reports
- resolve branch strategy drift between CI, release automation, and docs

Exit criteria:
- Docker build context no longer captures secrets or local runtime state
- top-level docs no longer contradict executable reality
- branch strategy is clearly documented and aligned

## Phase 1 - Execution surface truthfulness

Goal:
Make the planner and tool execution surface internally consistent.

Priority issues:
- COR-001
- INC-002

Tasks:
- define one authoritative executable tool catalog
- make planner emit only actually executable tools
- either implement or remove misleading tool exposure for:
  - `web_search`
  - `send_email`
  - `calendar_create_event`
  - `home_assistant_call_service`
- either implement scheduled workflow execution or explicitly de-scope it
- add integration tests proving every planner-emitted tool executes end to end

Exit criteria:
- every tool the planner emits is actually executable
- docs no longer imply workflow execution that does not exist
- scheduler workflow behavior is either real or clearly removed from claims

## Phase 2 - Dependency, validation, and packaging alignment

Goal:
Make installation and validation coherent.

Priority issues:
- DEP-001
- OPS-002
- TST-001
- OPS-001

Tasks:
- define a canonical supported runtime matrix for base, CPU, cu118, and cu124
- align `pyproject.toml`, requirements files, Dockerfile, and validation scripts
- fix repo-integrity tests so they compare against a baseline instead of failing on pre-existing dirtiness
- rewrite stale deployment validation logic
- reduce false positives in the security audit script

Exit criteria:
- one coherent dependency matrix exists
- deployment validation matches current repo reality
- integrity tests fail only for new dirtiness caused by the session
- security audit output is useful instead of noisy

## Phase 3 - Core runtime and entrypoint consolidation

Goal:
Choose and document the canonical runtime surfaces.

Priority issues:
- ARC-001
- DOC-002
- DOC-003

Tasks:
- choose the canonical voice-loop implementation
- reduce duplicate voice surfaces to wrappers or remove them
- define the official runtime modes:
  - text CLI
  - voice loop
  - dashboard or API
  - service mode if supported
- document official entrypoints clearly
- make config loading happen once, early, and consistently

Exit criteria:
- one canonical voice-loop path exists
- runtime modes are clear
- Windows and general setup docs point to the correct entrypoints
- config flow is consistent

## Phase 4 - Notification usability

Goal:
Finish notification usability end to end.

Roadmap source:
- Cycle 4.5
- Cycle 4.6
- Cycle 4.7
- Cycle 4.8

Tasks:
- minimal notification inbox UI
- notification filters
- mark read and mark all read actions
- retention cleanup scheduling for notifications and inbound SMS
- Codex verification after implementation

Exit criteria:
- notification inbox works in dashboard
- persistence works
- retention cleanup is scheduled and idempotent
- verification report is complete

## Phase 5 - Windows computer control hardening

Goal:
Make Windows computer control safe, auditable, and operationally usable.

Roadmap source:
- Cycle 5.2b
- Cycle 5.4
- Cycle 5.5
- Cycle 5.6

Tasks:
- route `rex pc run` through policy engine
- ensure approval gating is enforced
- validate Windows agent behavior
- add Windows Scheduled Task or service wrapper docs and scripts
- verify install and rollback steps

Exit criteria:
- computer control is approval-gated
- allowlist enforcement works
- Windows service or task mode is documented and validated
- Codex verification is complete

## Phase 6 - WordPress and WooCommerce

Goal:
Complete read-only monitoring first, then gated write actions.

Roadmap source:
- Cycle 6.1
- Cycle 6.2
- Cycle 6.3
- Cycle 6.4

Tasks:
- WordPress health checks
- WooCommerce order and product listing
- approval-gated writes for order status and coupon actions
- security, pagination, and error-handling verification

Exit criteria:
- read-only monitoring is stable
- write actions are approval-gated
- verification reports are complete

## Phase 7 - Voice identity MVP

Goal:
Make voice identity usable while keeping it optional.

Roadmap source:
- Cycle 7.1
- Cycle 7.2

Tasks:
- enrollment CLI
- embedding persistence per user
- threshold calibration
- unknown speaker behavior
- fallback identity flow
- verification of safety and false-positive controls

Exit criteria:
- enrollment works
- calibration works
- fallback flow works
- optional dependency policy remains intact

## Phase 8 - Real-time notifications and follow-up hardening

Goal:
Make the system feel responsive and tighten remaining gaps.

Roadmap source:
- Cycle 8.1
- Cycle 8.2
- Cycle 8.3

Tasks:
- SSE notification stream
- authenticated dashboard subscription
- UI live updates
- optional follow-up work such as calendar RRULE and TZID upgrades and Home Assistant TTS completion

Exit criteria:
- SSE is stable and authenticated
- dashboard updates live
- verification report is complete

## Phase 9 - Repo hygiene and debt paydown

Goal:
Reduce friction and stop future drift.

Priority issues:
- QLT-001
- DX-001

Tasks:
- staged ruff cleanup
- staged black normalization
- staged mypy cleanup
- root cleanup and archival of historical reports and artifacts

Exit criteria:
- quality gates are green or intentionally baseline-gated
- repo layout is cleaner
- no accidental behavioral changes from cleanup work

## Ralph Circle operating strategy

The Ralph Circle must follow this execution order:

1. always fix truth and safety issues before growth
2. prefer narrowing exposed capability over pretending unfinished capability is done
3. do not expand feature surface while the execution surface is lying
4. complete one bounded batch at a time
5. after each Claude batch, require Codex verification
6. update docs and CLAUDE.md whenever commands, structure, dependencies, config, or integrations change
