# Onboarding, Profile, and Integration UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan. Tasks use test-driven development and must be reviewed independently before merge.

**Goal:** Make setup deferrable, remove misleading primary navigation, and add a real per-user profile surface using existing identity, permission, voice, and runtime-path authorities.

**Architecture:** Add a reusable Python `user_profile_service` and one canonical Electron bridge. Typed IPC exposes the immutable active session profile to React. User-private files live under `data/users/<user>/`; shared household configuration remains unchanged.

**Tech Stack:** Python 3.11, pytest, Electron, React 18, TypeScript, Vitest.

## Global Constraints

- Read and follow `CLAUDE.md`.
- Preserve immutable Electron session identity.
- Validate every user ID before path/database access.
- Permissions remain authoritative in `rex.permissions`, never profile JSON.
- No new renderer `/api/` fetches or plaintext secrets.
- Do not remove the SMS route/backend; remove it from primary navigation only.

## Task 1: Lock Setup Deferral Semantics with Tests

**Files:**
- Create `gui/src/pages/setupWizardModel.ts`
- Create `gui/tests/setupWizard.test.ts`
- Modify `gui/src/pages/SetupWizardPage.tsx`
- Modify `bridge/rex_setup_bridge.py`
- Modify `tests/test_us058_setup_wizard.py`
- Modify `tests/test_us059_ha_setup.py`

**Steps:**
1. Write failing tests for `buildSetupSubmission(..., { deferHomeAssistant: true })` returning empty HA URL/token and an explicit defer flag.
2. Write a source/UI contract test that requires a visible `Do this later` action in the Home Assistant action row.
3. Write Python tests proving deferred setup persists neither HA URL nor HA token.
4. Implement the pure submission helper and wire the secondary action into the primary button row.
5. Update the bridge to fail closed: deferred HA values are ignored even if supplied.
6. Run: `npm.cmd test -- --run setupWizard` and `pytest -q tests/test_us058_setup_wizard.py tests/test_us059_ha_setup.py`.
## Task 2: Add the Canonical User Profile Service

**Files:**
- Create `rex/user_profile_service.py`
- Modify `rex/identity.py`
- Create `tests/test_user_profile_service.py`
- Modify `tests/test_identity.py`

**Steps:**
1. Write failing tests for profile composition, invalid user IDs, missing profiles, live permission projection, avatar isolation, and safe preference updates.
2. Route identity profile discovery and CRUD through `rex.runtime_paths.memory_dir()` while preserving optional test overrides.
3. Implement `UserProfileView` or an equivalent typed result that composes identity metadata, preferences, live permissions/role, voice enrollment summary, avatar metadata, and explicit private scope labels.
4. Store avatar bytes at `user_data_path(user_id, 'profile', 'avatar.jpg')` with JPEG/PNG validation, 2 MB limit, and deterministic 256x256 JPEG output when Pillow is available.
5. Do not return raw filesystem paths or secret values to the renderer.
6. Run: `pytest -q tests/test_user_profile_service.py tests/test_identity.py tests/test_identity_hardening.py`.

## Task 3: Expose Profile Operations through Typed Electron IPC

**Files:**
- Create `bridge/rex_profile_bridge.py`
- Modify `gui/src/main/bridgeResolver.ts`
- Create `gui/src/main/handlers/profile.ts`
- Modify `gui/src/main/ipc.ts`
- Modify `gui/src/preload/index.ts`
- Modify `gui/src/types/ipc.ts`
- Create `tests/test_profile_bridge.py`
- Create `gui/tests/profileHandlers.test.ts`
- Modify `gui/tests/bridgeResolver.test.ts`

**Steps:**
1. Write failing bridge tests for `get`, `update`, `set_avatar`, and `remove_avatar` using private session payloads.
2. Require bridge user identity to match the immutable Electron session user.
3. Return avatar as bounded base64 plus MIME type or initials; never a Flask URL.
4. Write handler tests for success, malformed bridge output, cross-user rejection, and missing optional avatar.
5. Register the bridge and typed methods without nested unresolved promises.
6. Run: `pytest -q tests/test_profile_bridge.py` and `npm.cmd test -- --run profileHandlers bridgeResolver`.
## Task 4: Build the Profile Page and Working Avatar Control

**Files:**
- Create `gui/src/pages/ProfilePage.tsx`
- Modify `gui/src/renderer/src/App.tsx`
- Modify `gui/src/layouts/AppLayout.tsx`
- Create `gui/tests/profileUi.test.ts`

**Steps:**
1. Write failing source/component contract tests requiring `/profile`, a real profile button, IPC-loaded avatar/initials, and no `/api/user/avatar` reference.
2. Replace the passive blue circle with an accessible button that navigates to `/profile`.
3. Add a Profile page showing name, role, live permissions, preferences, memory/private-data scope, and voice enrollment status.
4. Add avatar upload/remove controls using typed IPC and clear validation messages.
5. Label shared household settings separately and link to Settings without duplicating them.
6. Explain that active-user switching requires a new authenticated desktop session or restart.
7. Run: `npm.cmd test -- --run profileUi`.

## Task 5: Simplify Primary Navigation

**Files:**
- Modify `gui/src/layouts/AppLayout.tsx`
- Create `gui/tests/navigation.test.ts`

**Steps:**
1. Write failing tests proving scrolling navigation omits Settings and SMS, while the persistent Settings shortcut and direct `/sms` route remain.
2. Remove the static `beta` field and BETA tooltip rendering.
3. Keep Email visible without an unsupported readiness claim.
4. Preserve active-section and collapsed-sidebar behavior.
5. Run: `npm.cmd test -- --run navigation`.

## Task 6: Add Actionable Integration Detail

**Files:**
- Modify `gui/src/types/ipc.ts`
- Modify `gui/src/main/integrationInventory.ts`
- Modify `gui/src/pages/IntegrationsPage.tsx`
- Create or modify `gui/tests/settingsHandlers.test.ts`
- Create `gui/tests/integrationTruth.test.ts`

**Steps:**
1. Add failing tests for state-specific detail and next-action text.
2. Extend inventory items with non-secret `detail` and `next_action` fields.
3. Derive copy from existing evidence vocabulary; never infer authenticated/connected from credentials.
4. Render the detail and next safe action without duplicating the configure link.
5. Preserve explicit Outlook-unavailable messaging and SMS inventory truth.
6. Run: `npm.cmd test -- --run integrationTruth settingsHandlers`.
## Task 7: Document Durable Architecture and Validate PR A

**Files:**
- Modify `CLAUDE.md`
- Create `docs/acceptance/AUGUST_2026_DESKTOP_ACCEPTANCE.md`

**Steps:**
1. Document the canonical user profile service/bridge, private avatar location, immutable Electron user session, and SMS primary-navigation decision.
2. Record automated coverage completed in PR A and the remaining physical profile-isolation/page-smoke checks.
3. Run targeted tests from Tasks 1-6.
4. Run `pytest -q`.
5. From `gui/`, run `npm.cmd ci`, `npm.cmd run lint`, `npm.cmd run typecheck`, `npm.cmd test -- --run`, `npm.cmd run build`, and `npm.cmd audit --audit-level=high`.
6. Run `python -m rex doctor --release-gate`, `python scripts/security_audit.py --release-gate`, `pre-commit run --all-files --show-diff-on-failure`, and `git diff --check`.
7. Commit in logical Conventional Commit units, push, open a PR to `master`, wait for every required GitHub check, review the actual diff, and merge only when green.

## Completion Evidence

The PR description must list changed files by task, commands and outcomes, any remaining manual validation, and explicit confirmation that no paid service or new credential was introduced.
