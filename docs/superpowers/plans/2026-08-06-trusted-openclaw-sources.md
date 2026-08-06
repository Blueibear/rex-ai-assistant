# Trusted OpenClaw Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan. Security behavior is fail-closed and must be built test-first.

**Goal:** Establish a trustworthy source, provenance, permission, and audit authority before AskRex exposes external OpenClaw plugins or skills to ordinary users.

**Architecture:** Household-scoped source policy and manifests are persisted beneath canonical household data. Existing local plugin/skill loaders consult the policy. Electron exposes a truthful management surface; it does not invent a remote marketplace API.

**Tech Stack:** Python 3.11 dataclasses/JSON, existing AskRex permissions and audit logger, canonical Electron bridge, React/TypeScript, pytest/Vitest.

## Global Constraints

- Built-in AskRex content is the only trusted source by default.
- No remote source is hardcoded without independently verified official API and signing semantics.
- Unknown/malformed source or manifest data is denied.
- Admin approval never bypasses per-user execution permissions or Rex policy.
- Do not download or execute arbitrary remote code in this workstream.
- OpenClaw remains optional; core AskRex works without it.

## Task 1: Define Source and Manifest Contracts

**Files:**
- Create `rex/openclaw/source_policy.py`
- Create `tests/test_openclaw_source_policy.py`
- Create `docs/security/OPENCLAW_TRUSTED_SOURCES.md`

**Steps:**
1. Write failing tests for source types, trust states, verification policy, stable IDs, schema validation, timestamps, and default built-in source.
2. Define typed source and manifest records with provenance, publisher, version, update date, capabilities, requested permissions, risk, checksum/signature status, and denial reason.
3. Reject path traversal, invalid URLs/types, unknown trust values, oversized fields, and duplicate IDs.
4. Persist atomically under `household_data_path('openclaw', 'sources.json')` and `manifests.json`.
5. Document supported evidence and explicitly state that remote catalog installation is not yet implemented.
6. Run `pytest -q tests/test_openclaw_source_policy.py`.
## Task 2: Add Admin-Gated Source Management and Audit

**Files:**
- Modify `rex/permissions.py`
- Modify `rex/openclaw/source_policy.py`
- Modify `rex/audit.py` only if a reusable event shape is required
- Modify `tests/test_openclaw_source_policy.py`
- Modify permission tests

**Steps:**
1. Add failing tests proving ordinary users cannot add, approve, deny, enable, or remove sources.
2. Add narrowly scoped permissions such as `plugin_source_manage` and `plugin_use`, preserving existing `admin` behavior.
3. Require an explicit warning acknowledgement for marketplace, repository, Git, and local advanced sources.
4. Record source changes and manifest decisions through `get_audit_logger()` without secrets or local file contents.
5. Add per-user plugin grants that intersect with live AskRex permissions at execution time.
6. Run source-policy, permission, and audit tests.

## Task 3: Gate Existing Plugin and Skill Loading

**Files:**
- Modify `rex/plugins/__init__.py`
- Modify `rex/skills/loader.py`
- Modify `rex/skills/registry.py` as needed
- Modify `tests/test_plugin_loader.py`
- Modify `tests/test_skill_loader.py`
- Create `tests/test_external_capability_trust.py`

**Steps:**
1. Write failing tests proving built-in content loads, approved external manifests load, and unknown/denied/source-mismatched content does not load.
2. Keep backwards-compatible trusted behavior for repository-bundled plugins/skills.
3. Require an approved source and matching manifest before loading content outside the built-in roots.
4. Verify checksums when required and fail closed on mismatch; record signature status without pretending to verify unsupported signatures.
5. Attach source/provenance/risk metadata to loaded plugin and skill records.
6. Ensure loader failures degrade gracefully and are auditable.
7. Run plugin, skill, and trust tests.

## Task 4: Expose Trusted Sources through Electron IPC

**Files:**
- Create `bridge/rex_openclaw_sources_bridge.py`
- Modify `gui/src/main/bridgeResolver.ts`
- Create `gui/src/main/handlers/openclawSources.ts`
- Modify `gui/src/main/ipc.ts`
- Modify `gui/src/preload/index.ts`
- Modify `gui/src/types/ipc.ts`
- Create `tests/test_openclaw_sources_bridge.py`
- Create `gui/tests/openclawSourcesHandlers.test.ts`

**Steps:**
1. Write failing tests for list, add, approve, deny, enable/disable, remove, manifest list, and permission denial.
2. Bind every operation to the immutable Electron session user and live permissions.
3. Return only non-secret source and provenance data.
4. Register handlers and bridge validation; avoid nested promises.
5. Run Python bridge and GUI handler tests.
## Task 5: Build the Trusted Sources Management UI

**Files:**
- Modify `gui/src/pages/IntegrationsPage.tsx` or create `gui/src/pages/OpenClawSourcesPage.tsx`
- Modify `gui/src/renderer/src/App.tsx` if a dedicated route is added
- Create `gui/tests/openclawSourcesUi.test.ts`

**Steps:**
1. Write failing UI/source-contract tests for trust state, publisher, version, update date, permissions, risk, checksum/signature status, health, and denial reason.
2. Show the built-in source and approved manifests to all users with appropriate visibility.
3. Show add/approve/deny/enable/remove controls only when live permissions allow them.
4. Require an explicit advanced-source warning acknowledgement before submission.
5. Do not show a remote install action when no catalog adapter exists; state this truthfully.
6. Make untrusted and denied states visually distinct and accessible.
7. Run `npm.cmd test -- --run openclawSourcesUi openclawSourcesHandlers`.

## Task 6: Record CALL-E and the Acceptance Matrix

**Files:**
- Create `docs/roadmap/CALL_E_INTEGRATION.md`
- Finalize `docs/acceptance/AUGUST_2026_DESKTOP_ACCEPTANCE.md`
- Modify `REX_ACTIVE_CHECKLIST.md` only if it is present in the repository and active

**Steps:**
1. Record CALL-E as deferred with outbound-call-first scope, recipient/purpose confirmation, profile permissions, call status, transcript/result handling, and audit requirements.
2. Separate automated evidence from physical/live validation for Hold-to-Talk, wake word, STT, TTS, Home Assistant, profile isolation, page smoke, and installed restart behavior.
3. Do not mark a hardware/live check complete without direct evidence.
4. Cross-reference merged PRs and exact validation commands.

## Task 7: Document and Validate PR C

**Files:**
- Modify `CLAUDE.md`
- Modify relevant integration/security documentation

**Steps:**
1. Document the canonical trusted-source authority, default-deny rule, persisted paths, loader enforcement, and admin override.
2. Run all targeted source, permission, loader, audit, bridge, and GUI tests.
3. Run full pytest and all GUI quality gates.
4. Run release doctor, security release gate, pre-commit, diff check, and package smoke because a bridge was added.
5. Push, open a PR to `master`, independently inspect source and CI output, and merge only when all required checks pass.

## Completion Evidence

The PR description must enumerate trusted defaults, prove unknown sources fail closed, show admin and ordinary-user permission tests, state that no arbitrary remote installation was implemented, and confirm no paid service or credential was introduced.
