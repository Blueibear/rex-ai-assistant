# US-071 AI Provider Persistence Implementation Plan

**Goal:** Make Settings > AI provider selection persist immediately and reload from the canonical runtime config across section navigation and app restart.

**Architecture:** Keep `models.llm_provider` in `config/rex_config.json` as the runtime source of truth, with `gui_settings.json` as the renderer-facing mirror. Reuse the existing `normalizeGuiAiProvider()` / `toRuntimeAiProvider()` mapping. Provider selection must be persisted independently from provider-specific model validation so selecting Ollama or Local Transformers does not fail merely because its model field is still blank.

## Task 1: Prove the provider-only save bug

**Files:**
- Modify: `gui/tests/settingsMirror.test.ts`

Add a regression test that switches from runtime `transformers` to GUI `ollama` with a blank `customModelId`. Assert the provider write succeeds and writes `models.llm_provider = "ollama"` without requiring a model identifier.

Run: `npx vitest run tests/settingsMirror.test.ts`
Expected before fix: FAIL with `Model identifier is required`.

## Task 2: Make provider persistence independent from model editing

**Files:**
- Modify: `gui/src/main/settingsMirror.ts`
- Modify: `gui/tests/aiSettings.test.ts`

Only update `models.llm_model` when `customModelId` is explicitly nonblank. Preserve existing model validation when the model field itself is being saved. Add reload/mapping tests proving `models.llm_provider` overrides stale GUI state, `transformers` maps to `local`, and invalid provider values fail safely to the default.

Run focused Vitest tests until green.

## Task 3: Prove section-navigation persistence and source-of-truth behavior

**Files:**
- Modify: `gui/tests/settingsHandlers.test.ts` or a focused persistence test if clearer

Exercise the registered settings handlers with in-memory GUI/runtime stores: save the AI provider, reload `rex:getSettings('ai')` as a fresh section load, and confirm the canonical runtime provider is returned. This models leaving and re-entering the AI section without relying on component-local state.

## Task 4: Validate and close the story locally

Run:
- `cd gui && npx vitest run tests/aiSettings.test.ts tests/settingsMirror.test.ts tests/settingsHandlers.test.ts`
- `cd gui && npm run typecheck`
- `cd gui && npm run build`
- relevant full GUI tests if focused tests pass
- `git diff --check`

Update `PRD-production-readiness.md`, `docs/archive/progress/progress-production-readiness.txt`, and `CLAUDE.md` only if the implementation changes a durable architecture/maintenance rule. Leave the GitHub-check criterion open until the exact PR head is green remotely.
