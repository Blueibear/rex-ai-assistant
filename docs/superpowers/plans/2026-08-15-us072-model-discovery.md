# US-072 Model Discovery Implementation Plan

**Goal:** Discover real Ollama and LM Studio model IDs from configured endpoints, expose truthful UI states, and persist the selected model through existing AI settings.

**Architecture:** Keep LM Studio on Rex's existing OpenAI-compatible runtime path. Add one bounded main-process discovery module and one typed IPC method. The renderer chooses only discovery kind; the main process owns endpoint lookup and HTTP access.

## Task 1: Add red discovery contract tests

**Files:**
- Create: `gui/tests/modelDiscovery.test.ts`
- Create: `gui/src/main/modelDiscovery.ts`

Write tests first for:
- Ollama configured endpoint -> `/api/tags` -> unique model names.
- LM Studio configured OpenAI-compatible endpoint -> `/models` -> unique IDs.
- successful empty responses.
- non-2xx/network/malformed responses return safe provider-specific errors.
- missing/invalid configured endpoints do not call fetch.

Run `cd gui && npx vitest run tests/modelDiscovery.test.ts`; confirm red before implementation, then implement the smallest passing module.

## Task 2: Persist the OpenAI-compatible base URL

**Files:**
- Modify: `gui/src/types/ipc.ts`
- Modify: `gui/src/main/aiSettings.ts`
- Modify: `gui/src/main/settingsMirror.ts`
- Modify: `gui/tests/aiSettings.test.ts`
- Modify: `gui/tests/settingsMirror.test.ts`

Add `openaiBaseUrl` to `AiSettings`. Load it from `openai.base_url`, mirror trimmed nonblank values, and clear runtime `openai.base_url` when the GUI value is blank. Keep `models.llm_provider` unchanged as provider authority.

Add round-trip tests before implementation changes.

## Task 3: Add typed discovery IPC

**Files:**
- Modify: `gui/src/types/ipc.ts`
- Modify: `gui/src/preload/index.ts`
- Modify: `gui/src/main/handlers/settings.ts`
- Modify: `gui/tests/settingsHandlers.test.ts`

Add `discoverAiModels(provider: 'ollama' | 'lmstudio')`. The handler reads canonical `rex_config`, passes only the configured endpoint to the discovery module, rejects unsupported discovery kinds, and never accepts a renderer-provided URL.

Test handler registration, configured endpoint selection, unsupported provider handling, and safe error return.

## Task 4: Add truthful discovery UI

**Files:**
- Modify: `gui/src/pages/settings/AiSettingsSection.tsx`
- Create or modify focused GUI/source tests under `gui/tests/`

OpenAI section:
- optional OpenAI-compatible Base URL field;
- explicit `Discover LM Studio Models` button when configured;
- discovered model selector bound to `form.model`.

Ollama section:
- explicit `Discover Models` button;
- discovered model selector bound to `form.customModelId`.

Both:
- explicit loading, error, empty, and populated states;
- no network call on mount;
- selecting a model uses the existing `setSettings` path.

## Task 5: Documentation and durable rule

**Files:**
- Modify: `CLAUDE.md`
- Modify: `PRD-production-readiness.md`
- Modify: `docs/archive/progress/progress-production-readiness.txt`

Record that model discovery is user-initiated main-process IPC over configured endpoints, LM Studio remains the OpenAI-compatible runtime path, and renderer URLs are never accepted for discovery.

Check local acceptance criteria, leaving only GitHub checks unchecked until the exact PR implementation head is green.

## Task 6: Verification

Run fresh:
- `cd gui && npx vitest run` / documented Windows `npm.cmd test`
- `cd gui && npm.cmd run typecheck`
- `cd gui && npm.cmd run build`
- `cd gui && npm.cmd run lint`
- `cd gui && npm.cmd audit --audit-level=high`
- `python -m pytest -q tests/test_llm_client.py tests/test_model_router.py`
- `python scripts/security_audit.py --release-gate`
- `git diff --check`

Commit, push only after PR #401 is merged, open the US-072 PR, wait for exact-head checks, record evidence, rerun final merge checks on the closure head, and merge when green.
