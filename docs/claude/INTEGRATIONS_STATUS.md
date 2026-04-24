# Claude Reference: Integrations Status

This file is the active readiness snapshot for repo integration claims. Use these labels in README and planning docs unless a fresh code audit proves otherwise.

- **REAL** - backend exists, is tested enough for normal local use, and uses real data/service calls when configured.
- **PARTIAL** - backend exists but has meaningful gaps such as read-only support, credential-gated fallback, limited hardening, or incomplete UI coverage.
- **STUB** - scaffold or UI exists but external service behavior is not implemented end-to-end.
- **NOT STARTED** - roadmap only.

## Snapshot

| Integration | Status | Evidence / notes |
|---|---|---|
| Text chat | REAL | `rex` / `python -m rex` routes through `rex.cli:main` and `rex.assistant.Assistant`. |
| Voice loop | PARTIAL | `python rex_loop.py` wires wake word, Whisper STT, LLM, and TTS. Requires optional ML/audio dependencies and Python 3.11. |
| Python/Flask API and experimental web dashboard | PARTIAL | `rex-gui` serves local Flask APIs through `rex.gui_app`. The `/ui/` browser dashboard exists but is incomplete in live testing and should not be described as the primary GUI. |
| Electron desktop GUI | PARTIAL | Electron/React shell exists under `gui/` with routes and bridge handlers. It is the current primary GUI and depends on root-level Python bridge scripts and build artifacts in `gui/dist-electron`. |
| User auth/data isolation | REAL | `rex/auth.py`, `rex/permissions.py`, per-user profile data under `Memory/<user_id>/`, user preferences/avatar APIs, and SQLite-backed chat history exist. |
| Email | PARTIAL | Real IMAP/SMTP backend exists (`rex/integrations/email/backends/imap_smtp.py`) with stub fallback when credentials are absent. OAuth providers remain incomplete, and current Outlook-connected GUI status should not be described as full live mailbox sync. |
| Calendar | PARTIAL | ICS read-only backend exists (`rex/integrations/calendar/backends/ics_feed.py`) with stub fallback. Calendar write support and Google/CalDAV OAuth are not complete, and current Outlook-connected GUI status should not be described as full live calendar sync. |
| SMS / messaging | PARTIAL | Twilio SMS backend and stubs exist (`rex/integrations/messaging/backends/twilio_sms.py`, `rex/integrations/sms_service.py`). Requires Twilio credentials for real delivery. |
| Notifications | PARTIAL | `rex.notification` supports priority routing, quiet hours, digest, escalation, email/SMS/HA TTS channels, and CLI commands. Electron notification UI/IPC exists. Legacy Flask dashboard notification API routes are not the current surface. |
| Home Assistant TTS | PARTIAL | HA TTS notification client and `rex ha tts test` exist; requires config and hardening for production use. |
| Home Assistant device control | PARTIAL | `rex.gui_app` exposes HA test/save/state/device command endpoints. Device alias approval and command safety are still limited and credential-gated. |
| Web search | PARTIAL | `plugins/web_search.py` implements provider selection for configured providers. The tool registry health currently treats Brave/SerpAPI credentials as readiness signals. |
| Weather | PARTIAL | `weather_now` tool calls OpenWeatherMap through `OPENWEATHERMAP_API_KEY`; no key means no real weather results. |
| GitHub | PARTIAL | `rex gh` commands and `rex/github_service.py` exist; requires token and has limited surface area. |
| VS Code / code operations | PARTIAL | `rex code` commands and `rex/vscode_service.py` exist; intended as local developer tooling. |
| Browser automation | PARTIAL | `rex browser` commands and OpenClaw browser bridge exist; Playwright dependency and session handling are environment-sensitive. |
| OS automation / file ops | PARTIAL | `rex os` and `rex.tools.*` modules exist with confirmation/allowlist patterns; treat as high-risk and environment-specific. |
| Windows computer control | PARTIAL | `rex-agent`, `rex.computers.*`, allowlist/approval policy, and `rex pc` commands exist. Service hardening remains limited. |
| Shopping list | REAL | `rex shopping` CLI, `rex_shopping_list_bridge.py`, Electron page, and shopping PWA blueprint exist. |
| Reminders | REAL | `rex reminders` CLI, reminder service, and Electron bridge/page exist. |
| Tasks | REAL | Task bridge and Electron Tasks page exist; broader autonomous task execution remains alpha. |
| Knowledge base | PARTIAL | `rex kb` CLI and `rex/knowledge_base.py` exist for local ingestion/search; external sync is not implied. |
| Memory | PARTIAL | File profiles, key-value facts, working memory, long-term memory, and SQLite chat history exist, but data model boundaries are still evolving. |
| Voice identity / speaker recognition | PARTIAL | `rex/voice_identity/`, CLI enrollment/status/calibration, and Electron bridge scripts exist. Heavy speaker libraries are optional. |
| Planner / workflows / autonomy | PARTIAL | `rex plan`, `rex workflows`, `rex approvals`, `rex executor resume`, `rex/workflow_runner.py`, and `rex/autonomy/` exist. Treat as alpha and policy-gated, not fully autonomous production execution. |
| Scheduler | PARTIAL | `rex scheduler` commands and scheduler model exist; production callback coverage varies by feature. |
| WordPress | PARTIAL | `rex wp health` supports REST API health checks. It is monitoring-oriented, not a full WordPress content management client. |
| WooCommerce | PARTIAL | Order/product listing and approval-gated order status/coupon writes exist (`rex/woocommerce/`). Product writes and webhooks remain deferred. |
| OpenClaw gateway/client | PARTIAL | HTTP client/adapters and feature flags exist. Gateway-backed paths require an external OpenClaw gateway and `OPENCLAW_GATEWAY_TOKEN` where configured. |
| Rex tool server | REAL | `rex-tool-server` exposes `/rex/tools/{tool_name}` with API key auth, rate limiting, policy guard, and health endpoints. |
| TTS API | REAL | `rex-speak-api` exposes `/speak`, health endpoints, auth via `REX_SPEAK_API_KEY`, request size limits, and rate limiting. |
| Docker | PARTIAL | Dockerfile and docs exist; validate target deployment before claiming production readiness. |

## Known Caution Areas

- Python 3.11 is the only supported runtime. Python 3.12+ is rejected.
- Electron GUI claims should be verified with `npm.cmd run build` and the `gui/tmp_verify_*.cjs` harness pattern before release.
- Legacy Tkinter launchers (`run_gui.py`, `gui.py`) are deprecated.
- `flask_proxy.py` is a legacy proxy/API surface, not the primary GUI runtime.
- OpenClaw docs include historical migration details; current behavior is HTTP-based and feature-flagged.
- Archive docs and historical PRDs are not reliable as current-state sources.
