# AskRex Current-State Independent Audit

- **Audit date:** 2026-08-01
- **Desktop worktree:** `C:\Users\james\rex-ai-test\rex-ai-team-codex` (`team/codex-capability-security-audit`)
- **Mobile worktree (read-only):** `C:\Users\james\rex-ai-test\askrex-mobile\AskRex-lead` (`lead/shippable-mobile`, `9a6f9b4`)
- **Assignment:** `docs/planning/CODEX_ASSIGNMENT_001.md`

## Post-audit security remediation

After the source inspection completed, mobile PR #11 was merged into `main` at
`fb307f626cbaed2b4184e866dc4a56084dcc5946`. It removed the credential-bearing
clone URL and added a dependency-free tracked-file secret scan plus a GitHub
Actions security gate. The source-level exposure is therefore remediated on
current mobile `main`. Provider-side revocation and any approved history purge
remain unverified and are still release blockers under F-26.

## Scope, method, and evidence rules

This audit read `CLAUDE.md` before repository inspection and used these authoritative requirements:

- `docs/planning/source-of-truth/REX_Unified_Build_Spec_UPDATED.md`
- `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md`
- `docs/planning/TEAM_LEAD_OPERATING_RULES.md`

The operating rules require every real desktop capability to be discoverable, usable, and configurable in Electron; mobile actions to run through a paired desktop session; incomplete features to be hidden or truthfully disabled; OpenClaw to remain optional; and pairing to be device-bound, scoped, revocable, encrypted, and strongly reauthenticated for high-risk actions (`TEAM_LEAD_OPERATING_RULES.md:28-44`). The build spec additionally requires reliable wake-to-speech, barge-in, model fallback, planning, retrieval, permission-aware tools, and independent verification (`REX_Unified_Build_Spec_UPDATED.md:25-36,102-119,163-199,516-540`).

Completion marks and historical PRD claims were not accepted as evidence. The audit used current source, configuration, workflows, tests, targeted commands, and direct visual inspection of the current logo files. No paid service, live external account, deployment, credential rotation, installer publication, or physical audio/mobile hardware was used.

Evidence labels:

- **Verified fact:** directly observed in current files, command output, or a test run in this audit.
- **Inference:** a conclusion from verified code structure that was not reproduced dynamically. Inferences are explicitly labeled.
- **Not verified:** outside the permitted or available boundary, including live providers, GitHub-hosted job state, signed production artifacts, LAN/WAN pairing, physical iPhone behavior, and physical audio behavior.

Unless a finding says otherwise, each **Root cause** is the auditor's architectural inference from the cited facts; it is not represented as a reproduced runtime fact.

Severity scale: **Critical** (immediate credential/safety exposure), **High** (release blocker, security boundary missing, or materially false product behavior), **Medium** (important capability/reliability/parity gap), **Low** (maintainability or presentation defect with bounded impact).

## Executive verdict

AskRex is not currently shippable as the unified desktop/mobile product described by the authoritative requirements.

The strongest implemented foundations are real: Electron has broad IPC-backed pages; the canonical assistant fails closed on absent identity; memory/history are user-keyed at their APIs; the canonical tool lifecycle performs permission, risk, confirmation, deduplication, execution, verification, and audit stages; the mobile chat/auth/refresh/SSE/WebSocket/voice-upload/TTS paths are implemented; mobile refresh-token reuse revokes the token family and session; cross-transport message idempotency exists; and the Windows workflow builds and smoke-tests a managed-runtime installer.

The release blockers are equally concrete:

1. A credential-like GitHub personal-access token is embedded in a tracked mobile reset script.
2. Mobile login creates a session from username/password plus untrusted device metadata; there is no desktop-approved pairing, device ownership proof, capability grant, expiry, or action scope.
3. Mobile Home, Inbox, Automate, and History surfaces display fabricated operational data when their server routes are explicit `501 NOT_IMPLEMENTED` scaffolds.
4. Electron SMS reports an in-memory message as `sent` without delivery, notifications are seeded with fabricated events, and the Test Voice handler always returns success without speaking.
5. Electron does not expose/configure the full desktop capability registry, and several integration cards lead to missing or unrelated settings.
6. Desktop credentials/settings/data roots are global, plaintext, and fragmented across install-relative paths, launch-relative `data/`, and `~/.rex`; they do not meet the OS credential-store or coherent ownership boundary.
7. Model failure fallback, core-chat planning, relevance-based memory retrieval, dynamic OpenClaw plugin/skill discovery, and external-tool verification are not wired end to end.
8. Wake-word mode lacks demonstrated barge-in and physical restart/device-routing validation; only hold-to-talk is identified as the supported production path in `CLAUDE.md:291-295`.
9. Installer signing is optional, no signed updater exists, release versions drift, and the mobile repo has no native signing/private-distribution pipeline.
10. Mobile retains OnSpace application identity and an unrelated logo/token system, so branding, navigation terminology, and version identity do not match desktop.

## 1. Desktop capability inventory and Electron exposure

### Current inventory

`gui/src/renderer/src/App.tsx:106-127` defines 21 Electron routes. `gui/src/layouts/AppLayout.tsx:133-336` exposes 17 top-level navigation destinations. `gui/src/main/ipc.ts:1-46` registers 21 handler groups. The canonical tool registry defines 28 tools in `rex/tools/registry.py:236-568`, but the Electron capability inventory exposes only six capability records (`gui/src/main/integrationInventory.ts:135-146`).

| Capability or surface | Actual backend/IPC | Electron exposure | Electron configuration | Audit state |
|---|---|---|---|---|
| Chat | Assistant bridge and streaming handlers | Top-level Chat page | AI/provider/model settings | Exposed; configuration contains an invalid default pairing (F-03) |
| Voice, wake word, STT, TTS | Voice bridge, persistent wake process, renderer hold-to-talk | Top-level Voice page | Microphone, speaker, STT/TTS, wake settings | Broadly exposed; Test Voice is fake and wake path is not production-verified (F-04, F-15, F-16) |
| Tasks | Task handler/bridge | Top-level Tasks page | No dedicated provider/settings model | Exposed, partially configurable |
| Calendar | Calendar bridge | Top-level Calendar page | Google fields; Outlook labeled not live | Exposed; provider scope incomplete |
| Reminders | Reminder handlers | Top-level Reminders page | No dedicated configuration | Exposed |
| Memories | Private identity bridge | Top-level Memories page | User selection exists; data-root/retention configuration absent | Exposed; storage ownership fragmented (F-11) |
| Email | Read/draft bridge; GUI send unavailable | Top-level Email (Beta) page | Gmail/IMAP account fields | Truthfully read/draft-limited, but reply generation is a template stub |
| SMS | List bridge only; renderer send is in-memory | Top-level SMS (Beta) page | Twilio fields | **Misleading:** reports `sent` without delivery (F-04) |
| Notifications | In-memory Electron handler | Top-level Notifications page | No authoritative notification source config | **Misleading:** seeded fabricated events (F-04) |
| Shopping list | User-scoped handler | Top-level Shopping List page | No dedicated settings | Exposed |
| Logs | Log handler | Top-level Logs page | Debug logging toggle | Exposed |
| Command history | Private history bridge | Route exists | No dedicated retention/export policy | Route has no navigation item; partially discoverable (F-02) |
| Usage | Usage handler | Top-level Usage page | No budget/provider accounting configuration beyond AI limits | Exposed, partial |
| Home Assistant | HA service/verified mutation path | Top-level HA page plus settings route | URL/token and HA page | Best-exposed integration; live behavior not tested here |
| Devices/system state | Device and system IPC | Nested `/home/devices` route | Voice device settings; limited system controls | Partial discoverability |
| Quick Actions | IPC handler and page | Direct route only | Page-local management | Hidden from primary navigation (F-02) |
| File operations | Canonical `file_ops` plus file IPC | Indirect chat/file interactions | `allowedFileRoots`, confirmation toggle | Partial; capability/risk not discoverable |
| Time/weather | Canonical tools | Chat only | Weather location depends on general config | Usable indirectly; not listed in capability inventory |
| Web search | Canonical tool and integration card | Integration card/chat | No complete provider-key settings flow; key naming is inconsistent | Partial/misconfigured (F-02) |
| Windows diagnostics | 10 read tools | Chat only | General tool timeout | Usable indirectly; not discoverable as capabilities |
| Windows mutations | Volume, brightness, power plan, DNS, SFC | Chat only | Confirmation/allowed roots are generic | Not discoverable; verifier coverage varies |
| Music Assistant | Five canonical music tools | Chat/voice only | No Electron Music Assistant URL/token settings | Not configurable in Electron (F-02) |
| Phone/Telegram | Integration settings and inventory cards | Integrations/settings | Credential fields exist | Exposed as configuration; live capability not verified |
| MQTT/push/OpenClaw | Inventory/status cards | Integrations page | No corresponding complete settings sections | Card is ahead of configuration (F-02, F-22) |
| n8n/ComfyUI/Plex/WooCommerce/WordPress | Modules or requirements references exist | No Electron capability page | No Electron settings | Not discoverable/configurable (F-02) |
| About/setup | Electron pages/handlers | About and initial setup | General settings | Exposed |

### F-01 — Electron capability inventory is not the canonical registry

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `rex/tools/registry.py:236-568`; `gui/src/main/integrationInventory.ts:135-146`; `gui/src/main/ipc.ts:1-46`
- **Observed evidence:** The canonical registry contains 28 named tools spanning information, communications, files, Windows diagnostics/mutations, Home Assistant, and music. Electron's `buildCapabilityInventory()` returns only chat, voice, Home Assistant, email, calendar, and SMS. IPC registers more handler groups than either list.
- **Impact:** Users and auditors cannot determine from Electron what Rex can do, what is enabled, what is read-only, what requires confirmation, or what can be independently verified. This violates the required Electron discoverability/configurability boundary.
- **Root cause:** Three independently maintained inventories—tool registry, IPC handlers, and UI capability records—have no shared projection contract.
- **Recommended fix:** Project the canonical registry and integration status into one typed, user-facing capability model including source, availability, read/write status, risk, confirmation, verifier state, settings destination, and last health check. Keep UI-only surfaces as explicit non-tool capabilities in the same model.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_tool_registry.py tests/test_tool_execution_lifecycle.py`; `cd gui; npm.cmd test; npm.cmd run typecheck; npm.cmd run build`

### F-02 — Real capabilities are hidden or lack Electron configuration

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `gui/src/renderer/src/App.tsx:106-127`; `gui/src/layouts/AppLayout.tsx:133-336`; `gui/src/main/integrationInventory.ts:50-146`; `gui/src/main/settingsDefaults.ts:16-86`; `gui/src/pages/SettingsPage.tsx`
- **Observed evidence:** History and Quick Actions have routes but no sidebar entry. Music Assistant, n8n, ComfyUI, Plex, WooCommerce, WordPress, file/system tools, and several registry capabilities have no capability/settings destination. MQTT, push, web search, and OpenClaw appear as inventory cards but lack complete matching settings fields; the OpenClaw card points to an AI settings section containing no OpenClaw control.
- **Impact:** Capabilities are usable only through undocumented chat phrasing or direct URLs, while integration cards imply configuration that does not exist. Household users cannot safely discover or govern them.
- **Root cause:** Electron pages were added story-by-story without enforcing the team-lead rule that every real capability has a discoverable, truthful configuration/state surface.
- **Recommended fix:** Add one Capabilities page backed by F-01, link every supported capability to its correct settings/status surface, and explicitly mark unsupported/read-only/degraded states. Remove cards for capabilities that cannot be configured or provide a truthful disabled explanation.
- **Validation commands:** `cd gui; npm.cmd test; npm.cmd run typecheck; npm.cmd run build`; Electron harness: `cd gui; npm.cmd run build; node tmp_verify_capability_navigation.cjs`

### F-03 — Default Electron AI provider/model combination is invalid

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `gui/src/main/settingsDefaults.ts:41-53`; `gui/src/main/aiSettings.ts:38-56`
- **Observed evidence:** The default provider is `openai` while the default model is `claude-sonnet-4`. `buildAiSettings()` accepts Claude, Gemini, and OpenAI model names in a single field, defaults to Claude, and does not require provider/model compatibility.
- **Impact:** A fresh or partially configured Electron session can present a valid-looking AI configuration that the selected provider cannot execute, causing first-run chat failure or opaque backend errors.
- **Root cause:** Provider-specific model catalogs are represented as one unvalidated enum.
- **Recommended fix:** Make the model catalog provider-specific, choose a valid local-safe or configured-provider default, reject incompatible pairs at save time, and expose an actual connection/model check before claiming readiness.
- **Validation commands:** `cd gui; npm.cmd test -- aiSettings; npm.cmd run typecheck`; `py -3.11 -m pytest -q tests/test_model_router.py`

### F-04 — Electron contains false success and fabricated operational data

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `gui/src/main/handlers/sms.ts:79-109`; `gui/src/pages/SmsPage.tsx:118`; `gui/src/main/handlers/notifications.ts:5-94`; `gui/src/main/handlers/settings.ts:58-60`; `gui/src/main/handlers/email.ts:62-67`
- **Observed evidence:** `rex:sendSMS` creates an in-memory message with status `sent` and does not call a send bridge. The SMS page labels only `stub` status, so this fake result appears sent. Notifications initialize with fabricated TTS, profile, Home Assistant, GitHub, report, and digest events. `rex:testVoice` always returns `{ok:true}` without audio. Email reply generation returns a fixed template.
- **Impact:** Users can believe a text was delivered, a voice configuration worked, or an operational event occurred when none did. This directly violates the no-misleading-controls/success rule and is unsafe for communications and household operations.
- **Root cause:** Development fixtures and scaffolds are registered in the production Electron IPC surface without an explicit demo mode or truthful status.
- **Recommended fix:** Fail closed with `not_configured`/`not_implemented`, remove fabricated data from production initialization, implement real adapters only through the canonical lifecycle, and gate any sample data behind an unmistakable developer-demo flag and banner.
- **Validation commands:** `cd gui; npm.cmd test; npm.cmd run typecheck; npm.cmd run build`; `Select-String -Path gui/src/main/handlers/*.ts -Pattern 'Stub:|status: .sent.|sampleNotifications'`

## 2. Mobile functionality, transport, authentication, sessions, and desktop readiness

### F-05 — Mobile auth/chat/streaming/voice foundation is real and well tested

- **Severity:** Low (positive control; preserve)
- **Evidence class:** Verified fact
- **Exact location:** `rex/mobile_api/routes/auth.py:96-186`; `rex/mobile_api/db.py:83-137`; `rex/mobile_api/sessions.py:284-430`; `rex/mobile_api/capabilities.py:55-93`; mobile `services/authCore.ts`, `services/chatWebSocket.ts`, `services/secureStorage.ts:14-51`
- **Observed evidence:** The gateway implements login, short-lived access JWTs, hashed rotating refresh tokens, logout/revoke-all, refresh-reuse revocation, per-device session metadata, HTTP chat, SSE, first-frame-auth WebSocket chat, message idempotency, voice upload, and TTS. Native credentials use secure storage and fail closed. The audit ran 114 mobile tests successfully, including refresh, authoritative restoration, replay, stale-socket, cancellation, cross-transport ID, malformed-frame, and capability-contract cases.
- **Impact:** This is a dependable foundation for a paired mobile client and should not be replaced during pairing work.
- **Root cause:** The transport/auth slice has explicit contracts, dependency injection, structured errors, and focused tests.
- **Recommended fix:** Preserve these contracts; extend the session principal with pairing/grant data instead of creating a second transport stack.
- **Validation commands:** `cd C:\Users\james\rex-ai-test\askrex-mobile\AskRex-lead; npm.cmd test; npx.cmd tsc --noEmit`; `py -3.11 -m pytest -q tests/mobile_api`

### F-06 — Mobile operational tabs fabricate success-state data over 501 scaffolds

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `rex/mobile_api/routes/scaffolds.py:1-59`; `rex/mobile_api/capabilities.py:77-82`; mobile `services/homeService.ts:36-76`; `hooks/useHomeDevices.ts:29-76`; `services/notificationService.ts:32-91`; `hooks/useInbox.ts:32-70`; `services/automationService.ts:61-180`; `hooks/useAutomations.ts:34-76`; `services/auditService.ts:26-130`; `app/(tabs)/_layout.tsx:62-127`
- **Observed evidence:** The server truthfully returns authenticated `501 NOT_IMPLEMENTED` for Home Assistant, notifications, approvals, tasks, workflows, audit, and settings and advertises false capabilities. The mobile client nevertheless exposes Home, Inbox, Automate, and History tabs. Those data layers initialize or fall back to realistic mock locks, garage doors, approvals, workflows, reminders, security actions, and verified audit successes when calls fail or return empty.
- **Impact:** A user can see fabricated door/garage/security state, fake pending approvals, fake successful actions, and fake business workflows. This is a safety and trust release blocker, especially because the content resembles real household and business state.
- **Root cause:** A polished prototype UI was connected to failure-fallback fixtures instead of the server capability contract.
- **Recommended fix:** Remove production mock fallbacks. Gate tabs from `/mobile/capabilities`; show an explicit unavailable/not-implemented empty state; enable each tab only after its server-authoritative persistence, authorization, mutation, verification, and tests exist.
- **Validation commands:** `cd C:\Users\james\rex-ai-test\askrex-mobile\AskRex-lead; npm.cmd test; npm.cmd run lint; npx.cmd tsc --noEmit`; `Select-String -Path services/*.ts,hooks/*.ts -Pattern 'return MOCK_|useState.*MOCK_'`

### F-07 — Mobile session rotation is strong, but a session is not a paired device grant

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** `rex/mobile_api/db.py:83-120`; `rex/mobile_api/routes/auth.py:96-174`; `rex/mobile_api/sessions.py:284-430`; `rex/mobile_api/validation.py:105-148`
- **Observed evidence:** Sessions are per user and carry device metadata, expiry, and revocation. Refresh tokens are hashed and rotated atomically; reuse revokes the family/session. Device IDs are explicitly treated as untrusted metadata. No session record contains a device public key, pairing approval, grant/scopes, desktop owner, strong-auth time, or capability authorization.
- **Impact:** Session theft is meaningfully constrained, but any valid account password can enroll an arbitrary client directly. The backend cannot answer which desktop approved the device or which actions it may request.
- **Root cause:** Authentication/session delivery was implemented before the required desktop pairing and capability-grant model.
- **Recommended fix:** Extend the existing session model with a separate immutable device identity and pairing grant: desktop ID, device public key/thumbprint, approved user, scopes, grant creation/expiry, last strong-auth time, revocation reason, and audit trail.
- **Validation commands:** `py -3.11 -m pytest -q tests/mobile_api/test_login.py tests/mobile_api/test_refresh.py tests/mobile_api/test_session_endpoint.py tests/mobile_api/test_pairing.py`; mobile `npm.cmd test`

## 3. Desktop/mobile pairing and capability authorization

### F-08 — Required desktop-approved pairing and scoped authorization do not exist

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `docs/planning/TEAM_LEAD_OPERATING_RULES.md:29,39-44`; `rex/mobile_api/routes/auth.py:96-115`; `rex/mobile_api/db.py:83-120`; `gui/src/renderer/src/App.tsx:106-127`; `gui/src/main/ipc.ts:1-46`
- **Observed evidence:** Login verifies username/password and immediately creates a session. There are no QR/one-time-code endpoints, pending enrollment, desktop approval UI/IPC, device ownership keys, grant scope table, per-device capability list, approval expiry, or desktop revocation page.
- **Impact:** Mobile is a generic remote login client, not a securely paired extension of a specific trusted desktop. Desktop-native actions cannot be safely delegated under least privilege.
- **Root cause:** The current architecture equates account authentication with device enrollment and authorization.
- **Recommended fix:** Implement a desktop-owned pairing authority: short-lived single-use QR/code, proof-of-possession key exchange, explicit desktop approval, scoped capability grants, expiry, session binding, per-device revoke, and redacted audit. Mobile action requests must be evaluated against both the authenticated user and the paired-device grant.
- **Validation commands:** `py -3.11 -m pytest -q tests/mobile_api`; `cd gui; npm.cmd test; npm.cmd run typecheck`; mobile `npm.cmd test`; Electron pairing harness after build.

### F-09 — Remote encrypted transport is not an enforced product boundary

- **Severity:** High
- **Evidence class:** Verified fact plus stated inference
- **Exact location:** `rex/config.py:252-270`; `rex/mobile_api/app.py:95-105`; `rex/commands/mobile.py:60-61`; mobile `constants/config.ts:1`; `app/settings.tsx:291-532`
- **Observed evidence:** The gateway is disabled and loopback-bound by default, CORS is deny-by-default, and rate limits exist—positive local controls. However, `require_tls` defaults false and is an expectation/warning rather than an in-process TLS enforcement mechanism. The mobile default points to `https://askrex.app`, not a paired local desktop endpoint.
- **Impact:** **Inference:** Making a desktop reachable for mobile use currently requires an undocumented external reverse proxy/tunnel or manual server URL configuration. The repository does not establish certificate trust, desktop identity pinning, or an end-to-end encrypted pairing channel.
- **Root cause:** Safe local gateway defaults and a hosted-mobile URL were implemented without the intervening secure desktop discovery/tunnel/trust architecture.
- **Recommended fix:** Define one supported topology. Require TLS for non-loopback, bind the paired desktop identity into certificate/public-key trust, reject insecure URLs in production mobile builds, document LAN/WAN behavior, and test certificate/host mismatch and replay cases.
- **Validation commands:** `py -3.11 -m pytest -q tests/mobile_api/test_config.py tests/mobile_api/test_cli.py tests/mobile_api/test_app.py`; mobile `npm.cmd test`; physical LAN test with certificate pin mismatch and paired-device revocation.

## 4. Identity, isolation, memory, ownership, and credentials

### F-10 — Core assistant, Electron private bridges, history, and memory fail closed on identity

- **Severity:** Low (positive control; preserve)
- **Evidence class:** Verified fact; focused Python tests were not runnable in this environment
- **Exact location:** `gui/src/main/sessionIdentity.ts:6-116`; `bridge/rex_memories_bridge.py:53-74`; `bridge/rex_history_bridge.py:31-40`; `rex/assistant.py:727-790`; `rex/actions/dispatcher.py:145-171`; `rex/history_store.py:68-154`; `rex/memory.py:73-91,228-238,459-468,782-802`
- **Observed evidence:** Electron derives a validated OS-session user and emits explicit `private` or `shared_household` payloads. Private history/memory bridges require private scope and canonical user validation. Assistant resolves identity before intent/cache/context/tool work and never invents a default identity. History and memory CRUD validate the user key.
- **Impact:** User ID is correctly treated as an authorization boundary in the primary assistant and private data paths, materially reducing accidental cross-user reads/writes.
- **Root cause:** Recent identity hardening consistently passes an explicit validated user through session and bridge contracts.
- **Recommended fix:** Preserve this contract and require the same explicit principal + grant object in all new pairing, OpenClaw, retrieval, automation, and credential flows.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_assistant_identity_binding.py tests/test_electron_session_isolation.py tests/test_memory_isolation.py tests/mobile_api`

### F-11 — Runtime data ownership is fragmented and launch-directory dependent

- **Severity:** High
- **Evidence class:** Verified fact; launch-path impact is an inference
- **Exact location:** `gui/src/main/configStore.ts:11-22`; `rex/memory.py:91`; `rex/history_store.py:21`; `rex/assistant.py:133`; `rex/mobile_api/db.py:28-29`; `rex/autonomy/preferences.py:27`; `rex/notifications/models.py:31`; `gui/src/main/handlers/settings.ts:63-69`
- **Observed evidence:** Electron configuration uses `app.getAppPath()/../config`; memory and history default to relative `data/`; mobile users honor `REX_DATA_DIR` but default to relative `data`; preferences and notifications use global `~/.rex`; Electron preference suggestions also read global `~/.rex/preferences.json`. No `process.chdir()` establishes one Electron runtime root.
- **Impact:** **Inference:** Installed versus development launches can place data in different directories, and multiple Rex profiles share global preferences/notifications/config. Backup, deletion, migration, ownership, and privacy guarantees cannot be stated coherently.
- **Root cause:** Subsystems selected storage locations independently before a canonical per-install/per-user data-root contract existed.
- **Recommended fix:** Establish one OS-appropriate application data root, set it explicitly for every managed Python bridge (`REX_DATA_DIR`), partition private stores by validated user, distinguish household-shared stores explicitly, and migrate existing roots with dry-run/conflict reporting.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_memory_isolation.py tests/test_electron_session_isolation.py tests/test_command_history.py`; `cd gui; npm.cmd test; npm.cmd run build`; packaged Electron two-user ownership harness.

### F-12 — Desktop secrets are plaintext and globally shared, not OS-vault credentials

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `gui/src/main/configStore.ts:59-114`; `gui/src/main/handlers/settings.ts:31-55`; `gui/src/main/settingsRedaction.ts:8-40`; `gui/src/main/settingsMirror.ts:95-116,143-145`; `gui/src/pages/SettingsPage.tsx:3036-3058`; `rex/credentials.py:1-7,26-46,111-233`; `docs/planning/TEAM_LEAD_OPERATING_RULES.md:42`
- **Observed evidence:** Electron writes API keys to a plaintext `.env` adjacent to the app path. Python `CredentialManager` loads environment variables and optionally plaintext `config/credentials.json`. Integration settings are global rather than bound to a validated Rex user. No Windows Credential Manager/keyring/DPAPI-backed implementation is present. The integration UI reports “Saved” when `setSettings()` resolves, but recursive redaction removes passwords, client secrets, and tokens; only the Home Assistant token has an explicit `.env` redirect, and the integration mirror does not persist the other secret fields. Mirror failures are swallowed.
- **Impact:** A local process or another OS/Rex profile with file access can obtain persisted household/service credentials. Other integration secrets can appear saved but be discarded and fail after reload. Per-user credential ownership and revocation cannot be enforced, and installed-path writes may be unreliable under standard Windows permissions.
- **Root cause:** Secret redaction from JSON was implemented without a complete persistence contract; the secrets that are persisted were moved to files/environment rather than an OS credential vault with user/account references.
- **Recommended fix:** Introduce an OS-backed credential provider (Windows Credential Manager or DPAPI-protected vault), store only opaque references in user/account config, bind references to validated user and integration account, migrate `.env` secrets securely, and keep process environment injection minimal and short-lived.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_credentials.py tests/test_email_account_isolation.py`; `cd gui; npm.cmd test; npm.cmd run typecheck`; `python scripts/security_audit.py --release-gate`; packaged two-OS-user credential isolation test.

### F-13 — Mobile biometric confirmation is client-local and has fail-open branches

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** mobile `hooks/useBiometric.ts:61-123`; `services/biometricService.ts:65-85`; `constants/config.ts:44-49`; `rex/mobile_api/routes/scaffolds.py:26-31`
- **Observed evidence:** When the security setting is off or a risk is below a client-selected threshold, actions pass. If hardware is unavailable and alert fallback is disabled, the hook returns true; a later `no_hardware` result also returns true. The server approvals endpoint is a 501 scaffold and no server principal carries a recent strong-auth assertion.
- **Impact:** Client UI checks can be bypassed or can fail open, and even a successful biometric prompt is not cryptographically bound to a server-authorized high-risk request. It does not satisfy strong reauthentication.
- **Root cause:** Risk classification and biometric gating live in presentation code ahead of a server-authoritative approval protocol.
- **Recommended fix:** Make all high/critical mobile actions server-denied until a short-lived, challenge-bound strong-auth assertion is verified. Fail closed on unavailable/cancel/error, let the server assign risk and required factors, and bind approval to action hash, device grant, user, expiry, and one-time nonce.
- **Validation commands:** mobile `npm.cmd test`; `py -3.11 -m pytest -q tests/mobile_api/test_approvals.py tests/mobile_api/test_pairing.py`; physical Face ID/passcode unavailable, cancel, replay, and expired-challenge tests.

## 5. Voice path

### F-14 — Electron hold-to-talk has a coherent real path

- **Severity:** Low (positive control; preserve)
- **Evidence class:** Verified fact; physical behavior not verified
- **Exact location:** `gui/src/pages/VoicePage.tsx:245-336,627-713,742-790,1027`; `gui/src/main/handlers/voice.ts:60-145`; `rex/voice/stt.py:83-112`; `rex/voice/tts.py`; `CLAUDE.md:291-295`
- **Observed evidence:** Renderer recording requests the selected microphone, captures MediaRecorder audio, sends STT, streams chat, synthesizes configured TTS, directs playback with `setSinkId`, aborts active playback/reply on a new turn, handles microphone device loss, and records stage timings. STT supports background model warmup. `CLAUDE.md` explicitly identifies hold-to-talk as the supported production voice path.
- **Impact:** This is the best current end-user voice path and provides the right seams for device routing, recovery, and latency measurement.
- **Root cause:** Voice work converged on one Electron renderer pipeline with explicit cancellation and timing.
- **Recommended fix:** Preserve the architecture; replace only the fake settings test and add packaged/hardware release evidence.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_us008_voice_pipeline.py tests/test_us010_voice_pipeline_timeouts.py tests/test_us167_voice_latency.py`; `cd gui; npm.cmd run build`; Electron voice harness plus physical microphone/speaker matrix.

### F-15 — Wake-word and voice-identity modes lack production device behavior

- **Severity:** High
- **Evidence class:** Verified fact plus inference
- **Exact location:** `gui/src/main/handlers/voice.ts:42-145`; `gui/src/main/settingsMirror.ts:41-80`; `bridge/rex_voice_bridge.py:517-557`; `rex/voice/loop.py:74-126,180-249,325-360,521-853`; `rex/voice_identity/ui_service.py:45-66`; `rex/voice_identity/embedding_backends.py:70-100`; `CLAUDE.md:291-295`; `REX_Unified_Build_Spec_UPDATED.md:163-199`
- **Observed evidence:** Wake mode starts a persistent bridge, has startup failure states, timeouts, acknowledgement handling, wake-buffer priming, post-interaction cooldown, and recovery logging. Repository search found renderer interruption only for hold-to-talk; the wake loop has no explicit barge-in/cancel-TTS path. The wake bridge constructs its microphone and TTS path without the Electron-selected input/output device IDs, and voice settings mirroring omits those IDs and volume. GUI voice enrollment defaults to a `synthetic` backend documented for testing/development; it hashes raw audio bytes and ignores sample rate. `CLAUDE.md` calls wake-word mode beta and says physical verification is absent.
- **Impact:** **Inference:** A user cannot reliably interrupt a wake-word response, the selected Electron devices need not control wake mode, and byte-hash embeddings do not establish real same-speaker recognition across different utterances. Restart/device-switch behavior is not established on real hardware. This blocks the authoritative primary voice and ownership experience.
- **Root cause:** Wake detection, speaker identity, and pipeline resilience were developed with separate renderer/Python paths and test backends, but production device routing, full-duplex interruption, real embeddings, and packaged validation were not closed.
- **Recommended fix:** Share one input/output/volume contract across hold-to-talk and wake mode; add a cancellation primitive that stops playback/generation and rearms capture; disable release enrollment unless a real, healthy embedding backend is active; then validate false wakes, no-pause commands, speaker recognition, device removal, sleep/resume, restart, echo/retrigger suppression, and repeated turns on target hardware.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_voice_loop.py tests/test_voice_loop_fixes.py tests/test_us137_voice_rearm.py tests/test_us138_voice_roundtrip.py tests/test_voice_enrollment.py tests/test_voice_identifier.py`; packaged physical wake-to-speech/barge-in/device-routing and multi-speaker matrix with timing/error-rate logs.

### F-16 — Voice readiness and latency are not release-gated truthfully

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** `gui/src/main/handlers/settings.ts:58-60`; `gui/src/pages/VoicePage.tsx:648-689`; `rex/voice/loop.py:74-126`; `tests/test_us167_voice_latency.py`
- **Observed evidence:** Runtime stages emit timings and have timeouts, but the settings Test Voice operation always succeeds without TTS playback. No current hardware latency measurements or release thresholds were produced in this audit.
- **Impact:** Users can save a broken TTS/device route and receive a false success. Engineering has instrumentation but no current end-to-end latency/quality release evidence.
- **Root cause:** Diagnostics UI was stubbed while timing work remained test/log focused.
- **Recommended fix:** Make Test Voice perform actual synthesis and selected-device playback with explicit provider/device failure. Define and gate wake-to-ack, STT, first-token, synthesis, first-audio, and total-response thresholds separately for local/cloud and cold/warm starts.
- **Validation commands:** `cd gui; npm.cmd test; npm.cmd run build`; `py -3.11 -m pytest -q tests/test_us167_voice_latency.py tests/test_us010_voice_pipeline_timeouts.py`; packaged selected-device smoke with structured timing export.

## 6. Assistant intelligence

### F-17 — Core identity/context/tool-verification orchestration is substantial

- **Severity:** Low (positive control; preserve)
- **Evidence class:** Verified fact; focused Python execution unavailable in this audit
- **Exact location:** `rex/assistant.py:727-835`; `rex/actions/dispatcher.py:122-320`; `rex/context/builder.py:214-341`; `rex/tools/execution.py:125-328`
- **Observed evidence:** The reply pipeline resolves identity, routes intent, checks a per-user cache, builds profile/facts/recent-history context, dispatches skills/tools/HA/LLM, post-processes, builds a response, and records history. The canonical tool lifecycle orders availability, argument/identity validation, permission/risk, confirmation, dedupe, retry, execution, normalization, independent verification, truthful outcome, and audit. Unverified mutations are labeled `attempted_unverified`.
- **Impact:** Rex already has the correct core safety vocabulary and should remain the orchestrator even when adding OpenClaw or mobile actions.
- **Root cause:** The core assistant and tool lifecycle have explicit typed stages and fail-closed identity rules.
- **Recommended fix:** Preserve and reuse this lifecycle for all new Electron, mobile, automation, and OpenClaw mutations; do not let UI or remote adapters bypass it.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_assistant_identity_binding.py tests/test_tool_execution_lifecycle.py tests/test_us014_context_builder.py`

### F-18 — Model routing lacks real provider availability/failure fallback and may race

- **Severity:** High
- **Evidence class:** Verified facts; concurrency outcome is an inference
- **Exact location:** `rex/model_router.py:252-341`; `rex/assistant.py:766-825`; `rex/actions/dispatcher.py:303-314`; `rex/llm_client.py:573-665,772-861`; `tests/test_model_routing_integration.py:56-65,123-235`; `gui/src/main/aiSettings.ts:38-56`
- **Observed evidence:** Non-Ollama models are always considered available. `resolve_model()` returns the default even if unavailable and says the caller handles final fallback, but Assistant only assigns `self._llm.model_name`; LLM generation catches `TypeError` only. `LanguageModel` creates a provider strategy once with the initial model name, and generation calls that strategy; no setter updates or rebuilds it after Assistant changes the outer model field. Existing routing integration tests use a capturing fake whose `generate()` reads the outer field, so they do not exercise the real strategy boundary. `cloud_limit_hit()` exists but has no production call site. Assistant mutates one shared LLM object's model name per request and restores it afterward without a request-local model parameter or lock.
- **Impact:** Invalid keys/models, quota/rate limits, and provider outages do not trigger a demonstrated fallback. **Inference:** real generation can keep using the strategy's original model despite a configured route, while overlapping requests can observe or restore the wrong outer model because the shared mutable field spans `await` work.
- **Root cause:** Routing is a pre-call name switch, not an execution policy with provider health, request-local selection, error classification, and bounded fallback.
- **Recommended fix:** Pass the selected model/provider as immutable request context; preflight provider configuration; classify auth/quota/transient/model-not-found failures; execute a bounded policy-defined fallback chain; surface degraded state; and test concurrent mixed-category requests.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_model_router.py tests/test_assistant_concurrency.py`; GUI provider/model tests; injected 401/404/429/5xx/local-offline scenarios.

### F-19 — Planning, relevance retrieval, and dynamic local skills are not wired into normal chat

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `rex/assistant.py:179-193,727-790`; `rex/actions/dispatcher.py:173-240`; `rex/context/builder.py:214-341`; `rex/planner.py`; `rex/commands/workflows.py:241`; `rex/skills/loader.py:72-115`
- **Observed evidence:** Core Assistant constructs a skill registry/router but never calls `load_skills_from_directory()`. The loader has no production call site. `Planner` is constructed by a workflow CLI command, not the Assistant/action dispatcher. Context injects profile, all formatted user facts, follow-up cues, and only the last four turns; it does not call `LongTermMemory.search()` or a ranked retrieval layer.
- **Impact:** Normal chat cannot reliably decompose/track/verify multi-step work, newly installed skills are not discovered dynamically, and relevant long-term memory is not selected by query relevance, freshness, confidence, or source.
- **Root cause:** Planner, long-term memory search, and skill loader exist as separate story artifacts without one assistant execution/retrieval contract.
- **Recommended fix:** Add a request-scoped intelligence stage: retrieve ranked user-authorized context, decide direct/tool/plan mode, generate an inspectable plan when needed, execute every step through the canonical lifecycle, verify completion, checkpoint/recover, and load signed/approved local skills into a versioned registry.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_us070_memory_search.py tests/test_planner.py tests/test_skill_loader.py tests/test_tool_execution_lifecycle.py`; multi-user retrieval and interrupted-plan recovery tests.

### F-20 — Assistant failure recovery is incomplete at the LLM/tool orchestration boundary

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** `rex/actions/dispatcher.py:226-240,290-314`; `rex/assistant.py:766-790`; `rex/model_router.py:269-341`
- **Observed evidence:** Auto-tool selection/execution and LLM generation have no bounded orchestration retry/replan/fallback wrapper. The only LLM exception handled is a `TypeError` used to try a legacy prompt signature; provider/network/quota errors escape. No production call connects cloud failure status to `cloud_limit_hit()`.
- **Impact:** A transient provider or tool failure can terminate a turn rather than degrade, retry safely, replan, or return a structured partial result with recovery guidance.
- **Root cause:** Component-level retries exist in places, but the assistant lacks a request-level failure policy and recovery state machine.
- **Recommended fix:** Add typed failure categories, retry budgets by idempotency/risk, provider fallback, safe replan, partial-result preservation, user-visible degraded status, and recovery checkpoints. Never retry mutations unless idempotency and verification permit it.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_model_router.py tests/test_tool_execution_lifecycle.py tests/test_assistant_recovery.py`

## 7. OpenClaw and dynamic plugins/skills

### F-21 — OpenClaw HTTP boundary preserves core independence

- **Severity:** Low (positive control; preserve)
- **Evidence class:** Verified fact; live gateway not tested
- **Exact location:** `rex/openclaw/http_client.py:42-145`; `rex/openclaw/tool_bridge.py:100-188`; `rex/assistant.py:174-177`; `REX_Unified_Build_Spec_UPDATED.md:102-119`
- **Observed evidence:** OpenClaw is accessed over an authenticated HTTP client with timeouts/retries. Tool dispatch is flag/config dependent; 404 and connection/auth errors fall back to local execution, while policy denial remains denial. No OpenClaw Python package is required for the core assistant.
- **Impact:** Rex retains its own identity, policy, memory, tool lifecycle, and local tools when OpenClaw is absent—the correct architectural direction.
- **Root cause:** The migration replaced package coupling with adapters and feature flags.
- **Recommended fix:** Preserve this optional boundary while adding discovery and verification; do not move Rex identity, memory, policy, or final success claims into OpenClaw.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_openclaw_http_client.py tests/test_openclaw_tool_bridge_http.py tests/test_openclaw_feature_flag.py tests/test_openclaw_contracts_audit.py`

### F-22 — Dynamic OpenClaw/plugin/skill discovery, approval, sync, and GUI control are absent

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md:5-52`; `rex/skills/loader.py:72-115`; `rex/assistant.py:179-187`; `gui/src/main/integrationInventory.ts:127-131`; `gui/src/main/integrationStatus.ts:161-165`; `gui/src/pages/SettingsPage.tsx`
- **Observed evidence:** The authoritative OpenClaw checklist remains entirely unchecked. No production code calls the local dynamic skill loader. There is no OpenClaw plugin/skill discovery or registry-sync call, allowlist/approval UI, installed-plugin/skill display, capability diff, or OpenClaw settings page. The inventory card points to unrelated AI settings.
- **Impact:** OpenClaw can be manually targeted as a remote tool endpoint, but it is not the dynamic, governed capability ecosystem required by the build spec.
- **Root cause:** Static HTTP dispatch was completed without the discovery/governance/control plane.
- **Recommended fix:** Build a versioned discovery adapter that imports only metadata first, presents capability/risk/permission diffs for approval, stores per-user/household grants, syncs health/version changes, and routes approved calls through Rex policy/verification/audit. Add one truthful Electron page.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_openclaw_* tests/test_skill_loader.py`; `cd gui; npm.cmd test; npm.cmd run typecheck`; live-free fake-gateway discovery/change/removal tests.

### F-23 — OpenClaw registration and health can claim more readiness than exists

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** `rex/openclaw/agent.py:122-144`; `rex/openclaw/tool_registry.py:28-30`; `gui/src/main/integrationStatus.ts:161-165`
- **Observed evidence:** Agent registration logs that gateway registration is not available and returns `None`. The default OpenClaw tool health check always returns `(True, "OK")`. Electron status is configuration-derived, not proof of agent registration, discovered capability health, or verified external mutation support.
- **Impact:** A configured endpoint or default health callback can be interpreted as operational readiness without proof that Rex is registered, a tool is callable, or its result is independently verifiable.
- **Root cause:** Configuration presence, transport reachability, registration, tool health, and verified write capability are not distinct states.
- **Recommended fix:** Use explicit states: disabled, configured, reachable, authenticated, discovered, approved, read-capable, write-capable, write-tested, verifier-capable, degraded. Default unknown health to unknown/unavailable, never healthy.
- **Validation commands:** `py -3.11 -m pytest -q tests/test_openclaw_agent_basic.py tests/test_openclaw_tool_bridge_http.py tests/test_openclaw_policy_gated_tools_e2e.py`; Electron integration-status tests.

## 8. Packaging, distribution, updates, CI, security, and release gates

### F-24 — Desktop managed-runtime packaging and CI are substantial but locally unconfirmed here

- **Severity:** Medium (verification boundary; preserve controls)
- **Evidence class:** Verified workflow/source fact; GitHub/installer execution not verified in this audit
- **Exact location:** `gui/package.json:10-65`; `.github/workflows/ci.yml:1-536`; `.github/workflows/windows-electron-artifact.yml:56-115`; `scripts/verify_electron_package_contents.py`; `scripts/test_installed_electron_artifact.ps1`
- **Observed evidence:** CI defines lint, formatting, security audit, typecheck, GUI lint/typecheck/test/build, Node audit, Python tests/coverage, pip-audit, and secret scan. Windows builds a wheel, managed runtime, NSIS installer, verifies forbidden packaged resources, and exercises the installed artifact without machine Python/Node. The current checkout lacks `gui/node_modules` and a usable Python 3.11 pytest environment, so GUI/Python suites and installer build were not rerun.
- **Impact:** Static workflow design is stronger than many areas, but current hosted CI and artifact success cannot be claimed from this audit.
- **Root cause:** Local dependencies/artifacts were intentionally not installed for this usage-conserving analysis, and hosted checks were not accessed.
- **Recommended fix:** Preserve all gates, add an auditable release evidence manifest, and require the exact commit's successful CI, signed artifact hash, package-content report, and installed smoke report before release approval.
- **Validation commands:** `py -3.11 -m pytest -q`; `cd gui; npm.cmd ci; npm.cmd test; npm.cmd run typecheck; npm.cmd run build; npm.cmd run dist`; `python scripts/verify_electron_package_contents.py gui/dist/win-unpacked/resources`; installed artifact smoke script.

### F-25 — Signing, updater, publishing, and version identity are not release-grade

- **Severity:** High
- **Evidence class:** Verified fact
- **Exact location:** `.github/workflows/windows-electron-artifact.yml:66-99`; `.github/workflows/release-please.yml:11-22`; `release-please-config.json`; `.release-please-manifest.json`; `pyproject.toml:7`; `gui/package.json:3,15,59-65`; `gui/src/main/index.ts:16`
- **Observed evidence:** Windows CI explicitly passes with an unsigned artifact when certificate secrets are absent. Release Please is configured only for the Python package and does not attach/publish the Windows artifact. Electron Builder runs with `--publish never`. No `electron-updater`/`autoUpdater` implementation exists. Python is version 1.4.1 while Electron is 1.0.0; workflow installer paths are hard-coded to 1.0.0. The Electron application ID also differs between runtime and Builder configuration.
- **Impact:** An unsigned or stale-version installer can pass the release dependency, there is no verified update channel/rollback path, and users cannot reliably associate the app, installer, backend, and release metadata.
- **Root cause:** Artifact construction was completed separately from release identity, mandatory signing, publishing, and signed update delivery.
- **Recommended fix:** Define one product version source, synchronize package/app/backend manifests, require Authenticode for release (allow unsigned only for non-release CI), publish hashes/SBOM/provenance, attach the installer to the release, and implement signed update metadata with staged rollout and rollback.
- **Validation commands:** release manifest/version consistency script; `Get-AuthenticodeSignature 'gui/dist/AskRex Setup <version>.exe'`; `cd gui; npm.cmd run dist`; offline signed-update metadata verification and rollback test.

### F-26 — Mobile has a critical tracked credential and no native private-release/security pipeline

- **Severity:** Critical
- **Evidence class:** Verified fact; credential value intentionally redacted
- **Exact location:** mobile `scripts/reset-project.js:8`; `.github/workflows/ci.yml:1-21`; `app.json:2-45`; repository root (no `eas.json`, native signing configuration, or private distribution workflow)
- **Observed evidence:** A credential-like GitHub personal-access token pattern is embedded in a tracked HTTPS URL at `scripts/reset-project.js:8`. Mobile CI runs tests, lint, TypeScript, and Expo web export only. It has no secret scan, dependency audit, native iOS/Android build, signing, entitlement, private distribution, update, SBOM/provenance, or release approval gate. `app.json` lacks final iOS bundle identifier/Android package identity.
- **Impact:** The token may permit unauthorized repository/account access and must be treated as compromised. Even after rotation, the current CI would not prevent recurrence. The app cannot be privately distributed as a verified native release from this repository.
- **Root cause:** A scaffold/reset clone URL captured a live credential, while mobile CI and release engineering remained prototype-oriented.
- **Recommended fix:** Immediately revoke/rotate the token without printing it; determine exposure and purge it from history if confirmed; replace the URL with credential-free configuration; add secret scanning and protected push rules; add dependency audit; define native identifiers/signing; create TestFlight/internal Android distribution with explicit device/user access; and require native build/install/smoke/security evidence.
- **Validation commands:** provider-side token revocation confirmation; `git log --all -- scripts/reset-project.js` followed by an approved historical secret scan that redacts matches; repository secret scan; `npm.cmd audit --audit-level=high`; mobile CI native build/signature/install tests.

## 9. Branding, logo, tokens, and navigation parity

### F-27 — Mobile package identity and logo do not match AskRex desktop

- **Severity:** High
- **Evidence class:** Verified fact and direct visual inspection
- **Exact location:** `docs/BRANDING.md:1-15`; mobile `app.json:3-8`; `package.json:2-4`; `app/+not-found.tsx:1-6`; `app/settings.tsx:883-888`; `assets/images/logo.png`; desktop `gui/src/renderer/src/assets/brand-icon.png`, `brand-wordmark-light.png`
- **Observed evidence:** Canonical product name is AskRex Assistant. Mobile app/package/slug remain `onspace-app`, scheme is `onspaceapp`, the not-found source says “Powered by OnSpace.AI,” and the footer says AskRex Mobile v0.1.0 while manifests say 1.0.0. Direct visual inspection shows the mobile neon geometric R logo is not the desktop dinosaur/R AskRex icon/wordmark; file hashes differ.
- **Impact:** Users, OS install surfaces, deep links, screenshots, and support diagnostics can show unrelated product identity and conflicting versions. This fails brand parity and weakens trust in private distribution.
- **Root cause:** The mobile application began from an OnSpace scaffold and was reskinned incompletely without consuming the canonical brand assets/version source.
- **Recommended fix:** Replace app/package/slug/scheme/display metadata with approved AskRex identities, import canonical logo/icon/splash assets through a shared brand package or generated artifact, remove OnSpace references, and derive displayed version/build from the runtime manifest.
- **Validation commands:** mobile `npm.cmd test; npm.cmd run lint; npx.cmd tsc --noEmit; npx.cmd expo config --type public`; text scan for `onspace|OnSpace`; visual snapshot comparison of login, tabs, settings, icon, and splash.

### F-28 — Design tokens, terminology, and navigation are independently defined

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** desktop `gui/src/styles/tokens.css:1-11`; `gui/src/layouts/AppLayout.tsx:133-336`; mobile `constants/theme.ts:1-47`; `app/(tabs)/_layout.tsx:62-127`; `constants/Colors.ts`
- **Observed evidence:** Desktop and mobile use different base surfaces, primary blues, text colors, spacing/radius systems, and independent source files. Desktop navigation uses Tasks, Calendar, Reminders, Notifications, Integrations, and Home Assistant; mobile uses Inbox, Automate, History, and Status even when corresponding capabilities are false. Mobile also retains an Expo-template `constants/Colors.ts`, creating a second color source.
- **Impact:** Interaction patterns and terminology drift, parity reviews are manual, and inactive prototype tabs look like peer features to real Chat/Voice.
- **Root cause:** No shared design-token/terminology/navigation contract is generated for both clients.
- **Recommended fix:** Create a platform-neutral token and product-language source, generate TypeScript/CSS outputs, define mobile-appropriate navigation from capability state, and remove dead token sources. Parity does not require identical layouts, but names, states, icons, and risk/verification language must match.
- **Validation commands:** token-generation drift check; `cd gui; npm.cmd test; npm.cmd run build`; mobile `npm.cmd test; npx.cmd tsc --noEmit`; visual regression snapshots for shared states.

## 10. Misleading, stubbed, hidden, dead, or undocumented behavior

### F-29 — Hidden routes and dead/legacy settings obscure supported product behavior

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** `gui/src/renderer/src/App.tsx:120,126`; `gui/src/layouts/AppLayout.tsx:133-336,361-368`; mobile `constants/Colors.ts`; `constants/config.ts:25-42`; Electron `gui/src/main/integrationInventory.ts:50-146`
- **Observed evidence:** History and Quick Actions have page routes/title mappings but no primary navigation item. Mobile keeps hard-coded James/Cole display roles/scopes and an Expo template color module alongside the real theme. Electron's integration and capability lists disagree about what exists. Several real modules are chat-only and undocumented in the UI.
- **Impact:** Maintainers cannot distinguish supported, hidden, legacy, demo, and dead code; users cannot reliably discover behavior or know which permission/state source is authoritative.
- **Root cause:** Prototype and legacy artifacts were retained without an explicit production/demo/deprecated classification enforced by tests.
- **Recommended fix:** Maintain a single surface manifest with lifecycle state (`supported`, `beta`, `disabled`, `developer-demo`, `deprecated`); drive navigation/capability reporting from it; delete dead template data; document intentional chat-only capabilities and their safety settings.
- **Validation commands:** static surface-manifest parity test; desktop/mobile unit tests and typechecks; scans for hard-coded production user fixtures and orphan routes.

### F-30 — Current audit/release checks can miss deliberately realistic stubs

- **Severity:** Medium
- **Evidence class:** Verified fact
- **Exact location:** `scripts/security_audit.py`; `.github/workflows/ci.yml:55-70,512-536`; mobile services cited in F-06; Electron handlers cited in F-04
- **Observed evidence:** The desktop release security audit passed while production-registered Electron handlers still return false success/fabricated data. Mobile tests passed while production services intentionally return mock operations. The mobile repo has no equivalent secret/truthfulness scan and contains the critical credential in F-26.
- **Impact:** Green automated checks can be misread as product readiness even though those checks do not assert that every visible action/state is server-authoritative and verified.
- **Root cause:** Static marker scans detect obvious TODO/stub patterns, but there is no surface-to-capability truthfulness contract or fixture-ban for production bundles.
- **Recommended fix:** Add release tests that enumerate every visible capability/action and require one of: real verified implementation, explicit disabled state, or developer-demo gate. Ban production imports/returns of mock datasets and false `ok/sent/success` handlers. Apply secret/truthfulness gates to both repos.
- **Validation commands:** desktop `python scripts/security_audit.py --release-gate`; mobile secret scan; new capability-truth matrix tests in both repos; Electron/mobile E2E tests with all unsupported server capabilities false.

## Validation performed in this audit

Verified:

- `npm.cmd test` in the mobile repo — **114 passed, 0 failed**.
- `npx.cmd tsc --noEmit` in the mobile repo — **passed**.
- `npm.cmd run lint` in the mobile repo — **passed with 0 errors and 20 warnings**.
- `git diff --no-index --exit-code tests/mobile_api/contract_vectors.json <mobile>/tests/contract/contract_vectors.json` — **byte-identical**.
- `python scripts/security_audit.py --release-gate` in desktop — **passed**; 1,238 files scanned, no actionable incomplete markers, merge markers, or desktop-repo exposed secrets.
- Targeted source searches and line reads for all cited locations.
- Per-command safe-directory mobile `git status --short` before/after validations — **no mobile changes**.
- Tracked-file and redacted pattern check for mobile `scripts/reset-project.js:8` — **credential-like GitHub token pattern confirmed; value not emitted**.
- Direct visual inspection and SHA-256 comparison of current desktop and mobile logo assets — **not the same asset/design**.

Not verified:

- Focused desktop Python tests: `py -3.11` has no pytest; system Python 3.14 pytest is incomplete because `packaging` is missing. No dependencies were installed for this analysis-only audit.
- Electron tests/typecheck/build/installer: `gui/node_modules` is absent.
- Current GitHub-hosted CI result, Windows runner result, Authenticode signature, release artifact publication, auto-update, or rollback.
- Live OpenAI/Ollama/OpenClaw/Home Assistant/email/calendar/SMS/phone/search/MQTT/push/n8n/ComfyUI/Plex/WooCommerce/WordPress behavior.
- Physical microphone/speaker/wake-word/barge-in/latency behavior, LAN/WAN mobile transport, physical iPhone secure storage/biometrics/push, or private mobile installation.

## Ten implementation batches in dependency order

### Batch 1 — Contain the mobile credential and establish two-repo security gates

**Scope:** Revoke/rotate the exposed token; assess/purge history; remove credentialed URL; add mobile secret scanning, dependency audit, protected push, and a regression fixture that uses a redacted URL. Extend truthfulness scanning to production mock imports and false success handlers.

**Acceptance gate:** Provider confirms old token unusable; current and historical scans are reviewed without exposing the value; both repos block new secrets; mobile `npm test`, lint, typecheck, secret scan, and high-severity audit pass.

### Batch 2 — Canonicalize data roots, ownership, and OS credential storage

**Scope:** Define one Windows application data root; set it for every managed bridge; explicitly partition private versus household data; migrate relative `data/`, install-relative config, and `~/.rex`; move secrets to an OS vault with user/account-bound references.

**Acceptance gate:** Two Rex users and two Windows users cannot read each other's private memory/history/credentials; migration is dry-run-first, idempotent, conflict-safe, and backed up; packaged Electron writes only to the approved data root; identity/isolation suites pass.

### Batch 3 — Build device-bound desktop/mobile pairing and secure transport

**Scope:** Add one-time QR/code enrollment, device key proof, desktop approval/rejection UI, scoped/expiring grants, per-device revoke, strong-auth timestamps, TLS/pinning policy, and audit events. Extend—not replace—the current token/session transport.

**Acceptance gate:** Password alone cannot create an action-capable mobile device; code replay/expiry/wrong-desktop/key mismatch fail closed; non-loopback plaintext is rejected; grants constrain actions; revoke terminates HTTP/SSE/WS and pending work; pairing tests pass on LAN and a physical phone.

### Batch 4 — Make every visible surface truthful

**Scope:** Remove mobile operational mocks and Electron fake SMS/notifications/Test Voice/email reply behavior from production. Gate navigation/actions by server/canonical capability state; show explicit unavailable/read-only/degraded/not-implemented states.

**Acceptance gate:** With every optional provider disabled and every scaffold capability false, neither client displays operational fixtures or enabled mutations; no action reports sent/done/verified without canonical evidence; surface-truth matrix tests pass.

### Batch 5 — Unify the canonical capability registry with Electron configuration

**Scope:** Project canonical tools, integrations, IPC surfaces, risk, permissions, health, verification, and settings destinations into one manifest; add a Capabilities page; configure or truthfully disable every real capability; resolve hidden routes and provider/model validation.

**Acceptance gate:** Registry-to-UI parity test has no orphan capability or navigation target; every real capability is discoverable and has a valid settings/status destination or explicit disabled reason; fresh-install AI configuration executes a validated provider/model pair.

### Batch 6 — Complete assistant routing, retrieval, planning, and recovery

**Scope:** Make model/provider selection request-local and concurrency-safe; implement typed failure fallback; wire relevance/freshness/confidence memory retrieval; connect planning to normal chat; add checkpointed recovery and idempotency-aware retry.

**Acceptance gate:** Concurrent mixed-model/user tests show no state leakage; injected provider/auth/quota/network failures follow the configured fallback policy; multi-step plans expose step/verification status and resume safely; retrieval is user-isolated and relevance-tested.

### Batch 7 — Complete the optional OpenClaw ecosystem control plane

**Scope:** Add discovery, metadata normalization, version/health sync, permission/risk diff, allowlist approval, user/household grants, verifier mapping, revocation, and Electron status/settings—while retaining local core fallbacks.

**Acceptance gate:** With OpenClaw absent, core Chat/Voice/local tools pass unchanged. With a fake gateway, install/change/remove events update only approved capabilities; denied/unverified mutations cannot claim success; all external calls are redacted and audited.

### Batch 8 — Close production voice behavior on packaged hardware

**Scope:** Replace fake Test Voice; implement wake-mode barge-in; validate device routing/recovery; define cold/warm latency budgets; exercise sleep/resume, restart, echo suppression, no-pause command, repeated-turn, and disconnected-device cases.

**Acceptance gate:** Signed/packaged build passes the physical microphone/speaker matrix; wake-to-spoken-response and barge-in pass repeated runs; structured timing meets approved thresholds; failures identify the exact stage and recover without app restart.

### Batch 9 — Unify brand, design tokens, terminology, navigation, and versions

**Scope:** Remove OnSpace identity; adopt canonical assets; generate CSS/TypeScript tokens from one source; align state/risk/verification terminology; make navigation capability-aware; establish one version/build source across Python, Electron, mobile, and display metadata.

**Acceptance gate:** Automated scans find no banned/scaffold brand; app icons/splash/login/settings use approved assets; generated token outputs are drift-free; version assertions match every artifact; desktop/mobile visual parity review passes at target sizes and accessibility contrast.

### Batch 10 — Establish signed desktop and private native mobile release trains

**Scope:** Require Authenticode for release, publish installer/hash/SBOM/provenance, add signed staged updates/rollback, create native iOS/Android identifiers/signing/private distribution, and combine software, hardware, live-provider, and recovery evidence into one release manifest.

**Acceptance gate:** Exact release commit has green two-repo CI; desktop installer and update metadata signatures verify; clean-machine install/update/rollback pass; private mobile build installs only through the approved channel and passes pairing/revoke/background/resume tests; release manifest clearly distinguishes mock/local/hardware/live-provider evidence and contains no waived Critical/High findings.
