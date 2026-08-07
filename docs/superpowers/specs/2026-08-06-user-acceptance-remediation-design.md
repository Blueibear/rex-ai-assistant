# User Acceptance Remediation Design

**Date:** 2026-08-06
**Owner:** AskRex project supervisor
**Authorization boundary:** The user granted autonomous design, implementation, push, PR creation, and merge for the findings recorded in this document only. Existing no-new-spend rules remain in force.

## Purpose

Turn the August 6 live desktop acceptance findings into verified product changes without broadening scope or weakening AskRex security boundaries.

## Source of Truth

Current `origin/master` code and `CLAUDE.md` override older PRD claims. User-observed behavior is accepted as real-world evidence, while hardware-dependent success still requires a later physical acceptance pass.

## Delivery Structure

Use three sequential pull requests so each risk domain can be reviewed and rolled back independently:

1. **PR A - onboarding, navigation, profiles, and integration truth**
2. **PR B - persistent chat and wake-word reliability**
3. **PR C - trusted OpenClaw sources and acceptance documentation**

All three PRs are covered by the user's bounded autonomy authorization. Each PR starts from the latest green `master`, uses Conventional Commits, and merges only after required checks pass.
## Confirmed Current-State Findings

### Setup

`SetupWizardPage.tsx` already contains a low-emphasis "Skip Home Assistant" button below the setup card. The user did not perceive it. Skipping currently submits the same populated HA fields rather than explicitly clearing or marking the step deferred.

### Chat

Every GUI message starts a fresh `rex_chat_stream_bridge.py` process. That process initializes services, loads plugins, constructs `Assistant`, loads the selected model, handles one request, and exits. The live configuration also selects the development fixture `sshleifer/tiny-gpt2` through the Transformers provider. These facts explain both severe cold-start latency and poor response quality.

### Wake word

The live configuration selects built-in openWakeWord with the exact phrase `hey jarvis`, threshold `0.5`. The Voice page says only "configured wake word" and does not surface the active phrase, model, backend, threshold, selected PortAudio device, or score evidence. Saying `Rex` or `Hey Rex` cannot trigger a detector that only loaded `hey jarvis`.

### Navigation and profiles

The scrolling sidebar contains Settings even though a persistent bottom Settings control already exists. SMS is also in primary navigation. The bottom avatar uses a dead Flask URL (`/api/user/avatar`) in Electron and has no click handler.

The codebase has useful profile pieces but no unified Electron surface: identity metadata in `Memory/<user>/core.json`, live permissions in `data/users.db`, voice enrollment storage, per-user runtime data, and a Flask-only avatar route. Configuration profiles under `profiles/*.json` are a separate concept and must not be conflated with people.
### Integration status

AskRex already has the required truthful state vocabulary: unavailable, unconfigured, configured, reachable, authenticated, degraded, read-only, write-capable, write-tested, and verified. The Integrations page uses it correctly. The problem is the sidebar's vague static BETA badges and lack of actionable status detail beside each integration.

### OpenClaw sources

OpenClaw HTTP tool integration exists, but AskRex has no marketplace installer or trusted-source policy. Local plugins and skills can be dynamically loaded from disk. Therefore this work must establish a fail-closed source and manifest authority before adding remote installation. It must not pretend that a remote catalog or signature service exists when none is implemented.

## PR A Design - Onboarding, Navigation, Profiles, Integration Truth

### Setup wizard

- Move the Home Assistant defer action into the primary action row as an obvious secondary button labeled **Do this later**.
- When deferred, clear HA URL/token in the submission and persist no HA credential or configured URL.
- Keep Back and Finish semantics accessible.
- State clearly that Home Assistant can be configured later in Settings.
- Add renderer tests proving the button is visible and that deferral submits empty HA fields.

### Navigation

- Remove Settings from `navItems`; keep `/settings` route and persistent bottom shortcut.
- Remove SMS from `navItems`; preserve `/sms`, its backend, settings, inventory, and direct-route compatibility.
- Remove static BETA flags from navigation. Email remains visible without a vague badge.
- Add route/navigation regression tests.
### Unified user profile service

Create a reusable Python service that composes one person's profile from authoritative stores rather than inventing a replacement database:

- identity metadata and preferences from `Memory/<user>/core.json`, resolved through `rex.runtime_paths.memory_dir()`;
- live permissions from `rex.permissions`;
- presentation role derived from live permissions;
- voice enrollment metadata from the existing voice identity service;
- avatar under `data/users/<validated-user-id>/profile/avatar.jpg`;
- explicit private-data and memory-scope labels.

The service must validate user IDs before every path or database operation. It must never copy permissions into profile JSON as authority. Household configuration remains in the existing household config and vault.

Add a canonical Electron bridge and typed IPC handlers for:

- reading the immutable active session profile;
- updating permitted display/profile fields and preferences;
- uploading/removing an avatar with size/type validation;
- listing voice enrollment status and live permissions.

### Profile UI

- Replace the dead avatar `<div>` with a real button that loads the active profile through IPC.
- Show the avatar when present and initials otherwise.
- Navigate to `/profile`.
- Add a Profile page that clearly separates private user data from shared household settings.
- Display identity, avatar upload, role, live permissions, preferences, memory/data scope, and voice enrollment state.
- Do not implement hot user switching inside an established Electron session; session identity remains immutable. Explain that switching users requires a new authenticated session or restart.

### Integration truth

- Preserve the existing shared state vocabulary.
- Add actionable detail text to integration inventory items, including the next safe action.
- Keep credentials-only state at `configured`, never connected/authenticated.
- Keep unsupported Outlook paths explicitly unavailable.
- Use the Integrations page as the authoritative status surface.
## PR B Design - Persistent Chat and Wake-Word Reliability

### Persistent chat worker

Replace the one-process-per-message pattern with one long-lived Python chat worker per immutable Electron session identity.

- The worker initializes services, plugins, `Assistant`, and model state once.
- It accepts newline-delimited request objects with unique request IDs.
- It emits ready, token, done, error, and timing events.
- It supports cancellation without changing the session's user identity.
- Electron restarts it after failure and rejects all pending requests truthfully.
- App shutdown terminates it and plugins cleanly.
- The non-streaming bridge remains for compatibility and tests.

Measure and log worker startup, request receipt, first-token latency, completion latency, and provider/model identity without exposing secrets.

### Provider and response quality

- Remove the ambiguous setup choice "Local (Transformers / Ollama)."
- Offer explicit local choices: LM Studio, Ollama, and bundled/development Transformers.
- Never silently select `sshleifer/tiny-gpt2` as a production conversational model.
- If Transformers still uses the fixture model, label it development-only and show a blocking quality warning.
- LM Studio uses its OpenAI-compatible local endpoint and model discovery/validation already supported by AskRex patterns.
- Add a default screen-chat instruction to answer directly and proportionally: brief for simple requests, detail only when useful or requested.
- Preserve tool routing, user memory, streaming, and action verification.

Automated acceptance targets should measure framework overhead with mocked/local deterministic models. Real model speed remains a hardware/provider acceptance item, not a CI promise.
### Wake-word truth and diagnostics

The GUI must show the phrase that the detector actually loaded, not merely the requested phrase.

- Extend bridge ready/status events with active phrase, requested phrase, backend, fallback state, threshold, model/embedding path presence, selected PortAudio device label/index, and detector generation.
- Surface a compact diagnostic panel on the Voice page.
- Stream bounded score diagnostics: recent/max confidence, threshold, audio RMS/peak, and reject reason. Do not flood logs or the renderer.
- If the requested phrase is unavailable and fallback activates, update the UI immediately to the active fallback phrase.
- Phrase examples in the UI must be generated from active detector metadata.
- Keep Hold-to-Talk available as the supported production path while wake word remains hardware-beta.

The code cannot honestly guarantee that `Rex`, `Hey Rex`, and `Hey Jarvis` all trigger simultaneously unless models for all phrases are loaded. AskRex should support comma-separated built-in aliases only when openWakeWord reports those models as available. Otherwise it must display the one active phrase.

### Wake-word verification

Automated tests cover model selection, fallback metadata, score event throttling, mic routing, ready-state truth, and renderer presentation. Physical tests cover real microphone audio, phrase detection rate, false positives, distance/noise, acknowledgement playback, and repeated-cycle recovery.

## PR C Design - Trusted OpenClaw Sources

### Trust authority

Create a fail-closed source policy service with canonical persisted records under household data. A source record includes:

- stable source ID, name, source type, URL/path, publisher;
- trust state: built-in trusted, administrator approved, untrusted, denied;
- enabled state, creation/update timestamps, approving admin;
- required verification policy and optional pinned signing/checksum identity;
- explicit warning acknowledgement for advanced sources.

Only the built-in AskRex source is trusted by default. No remote marketplace is hardcoded until its official endpoint and verification semantics are independently verified.
### Plugin and skill manifests

Every discovered external plugin or skill must carry provenance before it can be enabled:

- source ID, publisher, package name, version, update date;
- declared capabilities and requested permissions;
- risk classification and sandbox expectations;
- checksum and signature status where the source provides them;
- install/enable status and denial reason.

Unknown, malformed, unsigned-when-required, denied, or source-mismatched manifests fail closed. Existing local plugin/skill loaders must consult this authority before loading externally sourced code.

### Permissions, audit, and admin override

- Reuse live AskRex permissions; add narrowly scoped plugin-management permissions only when needed.
- Normal users may view approved capabilities but cannot add sources or approve untrusted code.
- Administrators may add marketplace, repository, Git, or local sources through an advanced workflow that requires an explicit warning acknowledgement.
- Source changes, installation decisions, permission grants, tool calls, verification results, and failures are appended to the canonical audit system.
- Per-user plugin permission profiles restrict use after installation; installation approval never grants every user access.
- Execution remains subject to Rex policy, confirmation, timeout, output limits, and verification rules.

### GUI scope

Add a Plugin Sources and External Capabilities section under Integrations or Settings that truthfully shows source trust, provenance, requested permissions, risk, verification evidence, health, and available administrative actions. Do not expose a fake install button for catalog APIs that do not exist.

## Deferred Work

CALL-E is a definite future integration, not part of these PRs. Record a roadmap item describing outbound-call-first scope, profile permissions, explicit confirmation, recipient/purpose display, status tracking, transcript/result handling, and audit requirements.

## Acceptance and Test Inventory

Automated gates include targeted Python and GUI tests, full pytest, GUI lint/typecheck/tests/build/audit, release doctor, security release gate, pre-commit, and package smoke where touched.

Manual acceptance remains required for:

- Hold-to-Talk on the physical microphone and selected output device;
- wake phrase recognition and false-positive testing;
- real STT accuracy/latency and TTS playback quality;
- live Home Assistant read, safe mutation, and independent state readback;
- James/Cole privacy and permission isolation;
- installed Electron page smoke and restart behavior.
## Non-Goals

- Completing or removing the SMS backend.
- Implementing CALL-E.
- Claiming wake-word hardware reliability from mocks.
- Introducing hot profile switching that weakens immutable Electron session identity.
- Replacing the canonical permission database with profile JSON.
- Adding a paid marketplace, signing service, cloud model, or credential.
- Treating OpenClaw as required for core AskRex operation.

## Security and Data Boundaries

- User IDs are validated authorization keys.
- Profile/avatar data is private per-user state.
- Household integration configuration and plugin-source policy remain household-scoped.
- Secrets stay in the OS-backed vault.
- Renderer access uses typed IPC and canonical bridges, never new raw `/api/` fetches.
- External code is denied unless source and manifest policy both permit it.
- Success messages distinguish attempted, completed, and independently verified outcomes.

## Documentation Changes

Update `CLAUDE.md` only for durable new architecture or commands: the persistent chat worker, canonical profile service/bridge, and trusted-source authority. Add a user acceptance checklist and the CALL-E roadmap item. Existing historical PRDs remain history and are not marked complete solely from this work.

## Definition of Done

The bounded workstream is complete only when all three PRs are merged to `master`, all repository-required checks pass, automated acceptance coverage exists for every non-hardware requirement, and remaining physical/live checks are explicitly documented without unsupported success claims.
