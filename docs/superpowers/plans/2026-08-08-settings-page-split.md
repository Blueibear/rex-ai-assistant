# SettingsPage Section Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `gui/src/pages/SettingsPage.tsx` to a thin category/router facade and move every settings section into focused modules under 1,000 lines without changing renderer behavior, persistence, or packaged-app navigation.

**Architecture:** Keep category metadata/navigation in `SettingsPage.tsx`. Move reusable display primitives (`Toggle`, `SavedIndicator`) to `settings/shared.tsx`. Straightforward panels move one-for-one. The oversized Voice and Integrations panels split at their existing logic/render boundary: controller hooks own state/effects/handlers and view modules own JSX, so every file stays below 1,000 lines while preserving the same IPC calls and field names.

**Tech Stack:** React 18, TypeScript strict mode, Electron typed IPC, Vitest, electron-vite.

## Global Constraints

- Every settings section module must be `< 1,000` lines.
- `cd gui && npm run typecheck && npm run build` must pass.
- Every existing settings section must still render and save through the same `window.rex` IPC methods.
- No new dependencies.
- OpenClaw secret values remain main-process/vault-only; renderer fields retain existing redaction behavior.
- Preserve exact query-string routing (`#/settings?section=<id>`) and category IDs.

---

### Task 1: Add structural regression contract

**Files:**
- Create: `gui/tests/settingsSections.test.ts`
- Modify later: `gui/src/pages/SettingsPage.tsx`
- Create later: `gui/src/pages/settings/**`

**Interfaces:**
- Consumes: filesystem source tree.
- Produces: a test that rejects section modules >= 1,000 lines, a facade >= 700 lines, missing section exports, or missing category mappings.

- [ ] **Step 1: Write the failing test**

```ts
const sectionFiles = readdirSync(settingsDir, { recursive: true })
  .filter((name) => /\.(ts|tsx)$/.test(String(name)))
for (const name of sectionFiles) {
  expect(readFileSync(join(settingsDir, String(name)), 'utf8').split(/\r?\n/).length).toBeLessThan(1000)
}
expect(readFileSync(settingsPage, 'utf8').split(/\r?\n/).length).toBeLessThan(700)
```

Also assert the facade imports and maps: General, Voice, AI, Integrations, Notifications, Users, Audio, System, About.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd gui && npx -y npm@10.9.2 test -- --run tests/settingsSections.test.ts`
Expected: FAIL because `settings/` modules do not exist and the facade is 5,688 lines.

- [ ] **Step 3: Keep this test red until all extraction tasks are complete**

### Task 2: Extract shared primitives and simple sections

**Files:**
- Create: `gui/src/pages/settings/shared.tsx`
- Create: `GeneralSettingsSection.tsx`
- Create: `NotificationsSettingsSection.tsx`
- Create: `AboutSettingsSection.tsx`
- Create: `AudioOutputSettingsSection.tsx`
- Create: `UsersSettingsSection.tsx`
- Create: `SystemSettingsSection.tsx`
- Modify: `gui/src/pages/SettingsPage.tsx`

**Interfaces:**
- `shared.tsx` exports `Toggle` and `SavedIndicator` with the exact existing prop signatures.
- Each section exports a named React component with no props and continues reading/writing through `window.rex` exactly as before.

- [ ] **Step 1:** Move `Toggle`/`SavedIndicator` unchanged and move each complete panel body with its required type imports/constants.
- [ ] **Step 2:** Replace each moved panel in `SettingsPage.tsx` with imports.
- [ ] **Step 3:** Run `npx -y npm@10.9.2 run typecheck`; fix import/type errors without behavior changes.
- [ ] **Step 4:** Run full Vitest suite.

### Task 3: Extract AI section

**Files:**
- Create: `gui/src/pages/settings/AiSettingsSection.tsx`
- Modify: `gui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Exports `AiSettingsSection(): React.ReactElement`.
- Retains existing `AiSettings`, preference/model-routing state, save indicators, and typed IPC calls.

- [ ] **Step 1:** Move `SavedField`, `PERSONALITIES`, and `AiPanel` as one module (current logic+view is ~710 lines).
- [ ] **Step 2:** Import shared UI primitives and existing UI components by their current paths.
- [ ] **Step 3:** Typecheck and run `aiSettings.test.ts` plus full Vitest.

### Task 4: Split Voice into controller and view

**Files:**
- Create: `gui/src/pages/settings/voice/useVoiceSettingsController.ts`
- Create: `gui/src/pages/settings/voice/VoiceSettingsSection.tsx`
- Modify: `gui/src/pages/SettingsPage.tsx`

**Interfaces:**
- `useVoiceSettingsController()` owns current Voice state/effects/handlers and returns the values referenced by the JSX.
- `VoiceSettingsSection()` calls the hook, destructures its return, and renders the current Voice JSX unchanged.

- [ ] **Step 1:** Move voice constants/audio helper functions plus current Voice logic (before the existing `return (` at line ~1129) into the controller.
- [ ] **Step 2:** Return all state, refs, constants, and handlers consumed by the render tree.
- [ ] **Step 3:** Move the current Voice render tree into `VoiceSettingsSection.tsx` and destructure `ReturnType<typeof useVoiceSettingsController>`.
- [ ] **Step 4:** Run TypeScript to identify any omitted controller exports, add only the missing values, and repeat until clean.
- [ ] **Step 5:** Run full Vitest and production build.

### Task 5: Split Integrations into controller/helpers and view

**Files:**
- Create: `gui/src/pages/settings/integrations/useIntegrationsSettingsController.ts`
- Create: `gui/src/pages/settings/integrations/IntegrationControls.tsx`
- Create: `gui/src/pages/settings/integrations/IntegrationsSettingsSection.tsx`
- Modify: `gui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Controller owns integration form state, credential metadata, inventory/status synchronization, save/test/remove handlers.
- `IntegrationControls.tsx` exports the current password, status, connection badge, and test-button helpers.
- View renders Email, Calendar, SMS, Home Assistant, OpenClaw, Phone, and Telegram blocks with current field IDs and calls.

- [ ] **Step 1:** Move helper types/functions currently around lines ~2746-2999 into controller/control modules.
- [ ] **Step 2:** Move logic before the existing render boundary (~3354) into the controller and return all consumed values.
- [ ] **Step 3:** Move render tree into `IntegrationsSettingsSection.tsx` unchanged.
- [ ] **Step 4:** Run `openClawSettings`, `settingsHandlers`, `settingsMirror`, `settingsRedaction` tests and TypeScript.
- [ ] **Step 5:** Run full Vitest/build.

### Task 6: Reduce facade and prove all sections

**Files:**
- Modify: `gui/src/pages/SettingsPage.tsx`
- Modify: `gui/tests/settingsSections.test.ts`
- Existing smoke: `gui/src/main/artifactSmoke.ts`, `scripts/test_installed_electron_artifact.ps1`

**Interfaces:**
- Facade owns category metadata, query-string selection, left navigation, and `renderPanel` only.

- [ ] **Step 1:** Delete all extracted definitions/imports from the facade and map each category to its section component.
- [ ] **Step 2:** Run structural test; every section/helper module and facade must satisfy line limits.
- [ ] **Step 3:** Run both TypeScript configs, full Vitest, changed-file ESLint, and production build.
- [ ] **Step 4:** Use the existing installed-artifact Windows smoke as packaged render/save evidence; do not create a separate manual-only path.
- [ ] **Step 5:** Update PRD/progress and commit with Conventional Commits.

## Self-Review

- Spec coverage: all nine settings categories are explicitly assigned; oversized Voice/Integrations are subdivided; line-size, type/build, render/save, and packaged verification criteria are covered.
- Placeholder scan: no implementation placeholders; each task names exact files, boundaries, interfaces, and validation commands.
- Type consistency: section components are zero-prop named exports; Voice/Integrations controller return types are inferred via `ReturnType`, avoiding duplicated interface drift.