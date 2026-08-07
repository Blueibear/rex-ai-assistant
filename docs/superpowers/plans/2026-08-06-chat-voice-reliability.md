# Chat and Wake-Word Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan. Use systematic debugging and test-driven development for every behavior change.

**Goal:** Eliminate per-turn chat cold starts, prevent development-model misconfiguration, tighten default response style, and make wake-word state/phrase/score evidence truthful and diagnosable.

**Architecture:** One persistent Python chat worker is owned by Electron for the immutable desktop session. The wake-word bridge publishes structured detector metadata and throttled score evidence already produced by the canonical listener.

**Tech Stack:** Python 3.11 asyncio/NDJSON, Electron child processes, React, TypeScript, pytest, Vitest.

## Global Constraints

- Preserve `Assistant`, tool routing, user-scoped memory, and verification semantics.
- Preserve chat cancellation/barge-in; killing and cleanly restarting the worker is acceptable for cancellation if the Python generator cannot be interrupted safely.
- Never mutate `Assistant._user_id` per request.
- Do not claim physical wake-word reliability from automated tests.
- No cloud purchase or model download is authorized.

## Task 1: Specify the Persistent Chat Worker Protocol

**Files:**
- Create `tests/test_chat_worker.py`
- Create `gui/tests/chatWorkerManager.test.ts`
- Create `docs/architecture/CHAT_WORKER_PROTOCOL.md`

**Steps:**
1. Write failing Python protocol tests for startup `ready`, request IDs, token order, done/error events, one immutable user, sequential turns, and graceful shutdown.
2. Write failing TypeScript manager tests for worker reuse, first-token routing, completion, cancellation, crash rejection, and restart.
3. Document NDJSON request/event schemas and lifecycle before implementation.
4. Run the failing tests and capture expected failures.
## Task 2: Implement and Package the Chat Worker

**Files:**
- Create `bridge/rex_chat_worker.py`
- Modify `gui/src/main/bridgeResolver.ts`
- Modify `tests/test_chat_worker.py`
- Modify `gui/tests/bridgeResolver.test.ts`

**Steps:**
1. Initialize services, plugins, `Assistant(user_id=...)`, and model state once after validating the command-line user.
2. Emit non-secret provider/model identity and startup timing in the `ready` event.
3. Process one request at a time; emit token, first-token timing, done, and structured error events keyed by request ID.
4. Reject any payload containing a different user or non-private data scope.
5. Implement graceful shutdown and plugin cleanup.
6. Keep `rex_chat_stream_bridge.py` unchanged for compatibility.
7. Add the worker to bridge validation/package resources and run `pytest -q tests/test_chat_worker.py`.

## Task 3: Replace Per-Turn Electron Spawns with a Worker Manager

**Files:**
- Create `gui/src/main/chatWorker.ts`
- Modify `gui/src/main/handlers/chat.ts`
- Modify `gui/src/main/index.ts` or app shutdown wiring
- Modify `gui/tests/chatWorkerManager.test.ts`

**Steps:**
1. Implement a single worker manager per Electron session identity with an explicit state machine: stopped, starting, ready, busy, failed.
2. Queue or reject overlapping turns deterministically; do not interleave token streams.
3. Route events by request ID and preserve the existing renderer IPC contract.
4. On cancellation, cancel safely or terminate the worker, reject the affected request as `AbortError`, and lazily restart for the next turn.
5. Reject all pending requests on crash and record structured logs.
6. Stop the worker during app shutdown.
7. Run `npm.cmd test -- --run chatWorkerManager` and existing Hold-to-Talk tests.

## Task 4: Make Local Provider Selection Explicit and Truthful

**Files:**
- Modify `gui/src/pages/SetupWizardPage.tsx`
- Modify `gui/src/pages/setupWizardModel.ts`
- Modify `bridge/rex_setup_bridge.py`
- Modify `gui/src/main/aiSettings.ts`
- Modify `gui/src/pages/SettingsPage.tsx`
- Modify related IPC types/tests
- Modify `tests/test_us058_setup_wizard.py`
- Modify `gui/tests/aiSettings.test.ts`

**Steps:**
1. Add failing tests for explicit LM Studio, Ollama, and development Transformers choices.
2. Map LM Studio to the existing OpenAI-compatible local base URL without requiring a cloud credential.
3. Add endpoint/model validation and model discovery using existing local-provider patterns.
4. Label `sshleifer/tiny-gpt2` development-only and block silent production selection.
5. Migrate ambiguous `local` setup values safely; do not rewrite an intentional existing provider.
6. Run targeted Python and GUI settings tests.
## Task 5: Tighten Default Screen-Chat Responses

**Files:**
- Modify `rex/context/builder.py`
- Modify `tests/test_assistant.py`
- Modify `tests/test_assistant_identity_binding.py`

**Steps:**
1. Write failing tests requiring simple greetings/factual prompts to receive a direct proportional-response instruction while preserving explicit requests for detail.
2. Add one screen-chat system instruction distinct from the existing voice-only concise instruction.
3. Keep user personality, memory, tools, and safety context intact.
4. Do not truncate tool results or force one sentence for complex tasks.
5. Run `pytest -q tests/test_assistant.py tests/test_assistant_identity_binding.py`.

## Task 6: Publish Active Wake-Word Detector Metadata

**Files:**
- Modify `rex/wakeword/listener.py`
- Modify `bridge/rex_voice_bridge.py`
- Modify `gui/src/main/handlers/voice.ts`
- Modify `gui/src/types/ipc.ts`
- Modify `gui/src/preload/index.ts`
- Create or modify wake-word Python and GUI tests

**Steps:**
1. Write failing tests for configured vs active phrase, backend, threshold, fallback, detector generation, and selected PortAudio device.
2. Add a bounded `WakeWordRuntimeStatus` event emitted when the detector loads, arms, falls back, rebuilds, or changes phrase.
3. Forward throttled attempt evidence: latest/max confidence, threshold, RMS, peak, reject reason, and attempt count. Never forward raw audio.
4. Ensure the ready event follows an armed listener and includes the active phrase.
5. Update status immediately after fallback activation.
6. Run targeted wake-word listener, bridge, and microphone-routing tests.

## Task 7: Show Exact Phrase and Diagnostics in the Voice Page

**Files:**
- Modify `gui/src/pages/VoicePage.tsx`
- Modify related voice components
- Create `gui/tests/wakeWordDiagnostics.test.ts`

**Steps:**
1. Write failing UI contract tests for exact active phrase, fallback notice, backend, threshold, microphone, and bounded score evidence.
2. Replace generic "configured wake word" copy with "Say: <active phrase>".
3. Add a collapsible diagnostic panel with actionable low-audio, below-threshold, unavailable-model, and fallback messages.
4. Keep Hold-to-Talk visually available and labeled as the reliable immediate path.
5. Do not imply aliases are active unless detector metadata confirms them.
6. Run `npm.cmd test -- --run wakeWordDiagnostics wakeWordMicrophoneRouting holdToTalk`.
## Task 8: Document and Validate PR B

**Files:**
- Modify `CLAUDE.md`
- Modify `docs/acceptance/AUGUST_2026_DESKTOP_ACCEPTANCE.md`
- Modify `RUNNING.md` or provider setup docs where commands change

**Steps:**
1. Document the persistent chat worker lifecycle, explicit local provider choices, and wake-word runtime metadata contract.
2. Record measured automated framework overhead separately from real-model/hardware results.
3. Run all targeted tests from Tasks 1-7.
4. Run `pytest -q`.
5. From `gui/`, run lint, typecheck, all Vitest tests, build, and high-severity audit.
6. Run release doctor, security release gate, pre-commit, diff check, and packaged Electron smoke because bridge/runtime contents changed.
7. Push, open a PR to `master`, independently inspect the actual diff and CI logs, and merge only when every required check is green.

## Completion Evidence

Report before/after process-count behavior, worker startup/first-token timings from deterministic tests, the exact active wake phrase exposed by metadata, remaining physical wake-word acceptance steps, and confirmation that no model/service purchase occurred.
