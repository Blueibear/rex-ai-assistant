# Cross-Documentation Truth Audit

**Scope:** US-054 production-readiness cross-doc audit.

**Verified code snapshot:** `AUDIT_COMMIT_PENDING` (replaced with the implementation commit SHA before this story is closed).

This audit reconciles current-state claims in [README](../README.md), [INSTALL](../INSTALL.md), [RUNNING](../RUNNING.md), [UI surfaces](UI_SURFACES.md), [surface classification](../SURFACE-CLASSIFICATION.md), [integration status](../INTEGRATIONS_STATUS.md), and [CLAUDE](../CLAUDE.md) against code and repository evidence. Historical files under `docs/archive/` are not current-state authorities.

## Audit Result

| Claim family | Verified current fact | Code / filesystem evidence | Current docs checked | Result |
|---|---|---|---|---|
| Install methods | End users use the packaged Windows Electron installer. `pip install .` is the developer/operator source install and does not install the Electron app. | `gui/package.json`, `pyproject.toml` | README, INSTALL, RUNNING, UI_SURFACES, SURFACE-CLASSIFICATION, CLAUDE | Reconciled |
| Console scripts | Exactly six `[project.scripts]` entries exist. | `pyproject.toml` | README, INSTALL, SURFACE-CLASSIFICATION, CLAUDE | Verified |
| Root Python inventory | Root-level `.py` count is 27; bridge wrappers are compatibility/test surfaces, not Electron runtime bridge paths. | Repository root + `gui/src/main/bridgeResolver.ts` | SURFACE-CLASSIFICATION, CLAUDE, UI_SURFACES | Verified |
| Voice mode default | Source `rex_loop.py` defaults to Hold-to-Talk/manual activation; `--mode wake-word` is explicit beta opt-in. Electron Hold-to-Talk is the supported end-user path. | `rex_loop.py` argparse default | README, INSTALL, RUNNING, UI_SURFACES, SURFACE-CLASSIFICATION, CLAUDE | Reconciled |
| OpenClaw status | Experimental/off by default. Enabled tools/voice require gateway URL + token. `/healthz` proves reachability only. Retryable gateway failures degrade to local execution with structured warning; 403 remains policy denial. | `rex/config.py`, `rex/openclaw/http_client.py`, `rex/openclaw/tool_bridge.py` | README, INSTALL, RUNNING, SURFACE-CLASSIFICATION, INTEGRATIONS_STATUS, CLAUDE, OpenClaw status | Reconciled |
| Docker tier | Developer-only/operator smoke path; lightweight liveness check is `python -m rex doctor --healthcheck`. | `Dockerfile` | README, SURFACE-CLASSIFICATION, CLAUDE, docker.md | Verified |
| Home Assistant verification | Mutation success is `verified` only after independent state observation. Other outcomes preserve uncertainty and verification evidence. | `rex/ha/mutation_service.py`, `rex/response/builder.py` | README, INTEGRATIONS_STATUS, CLAUDE, home_assistant.md | Verified |

## Install Methods

- **End-user artifact:** packaged Windows Electron installer. Its managed runtime includes Python 3.11, the AskRex wheel, canonical bridge scripts, voice dependencies, and FFmpeg; it does not require machine Node/Python.
- **Developer/operator source install:** Python 3.11 plus `pip install .`. This installs the Python package, six console scripts, bridge data files, and config example; it does not create the Electron installer.
- **Editable contributor install:** `pip install -e ".[dev,test]"` when development tooling is required.

## Console Scripts

Verified directly from `pyproject.toml`:

- `rex` -> `rex.cli:main`
- `rex-config` -> `rex.config:cli`
- `rex-speak-api` -> `rex_speak_api:main`
- `rex-agent` -> `rex.computers.agent_server:main`
- `rex-gui` -> `rex.gui_app:main`
- `rex-tool-server` -> `rex.openclaw.tool_server:main`

The mobile gateway is a `rex` CLI subcommand (`python -m rex mobile-api`), not a seventh console script.

## Root Python Inventory

Root-level `.py` count: `27`.

- `config.py`
- `conftest.py`
- `flask_proxy.py`
- `llm_client.py`
- `rex_chat_bridge.py`
- `rex_chat_stream_bridge.py`
- `rex_file_extract_bridge.py`
- `rex_loop.py`
- `rex_memories_bridge.py`
- `rex_reminders_bridge.py`
- `rex_shopping_list_bridge.py`
- `rex_speak_api.py`
- `rex_speaker_bridge.py`
- `rex_stt_bridge.py`
- `rex_tasks_bridge.py`
- `rex_voice_bridge.py`
- `rex_voice_enrollment_bridge.py`
- `rex_voice_sample_bridge.py`
- `rex_voice_upload_bridge.py`
- `rex_voices_bridge.py`
- `rex_wakeword_list_bridge.py`
- `rex_wakeword_sample_bridge.py`
- `rex_wakeword_train_bridge.py`
- `setup.py`
- `sitecustomize.py`
- `voice_loop.py`
- `wsgi.py`

The 17 `rex_*_bridge.py` root files are compatibility wrappers. Electron development resolves canonical scripts under `bridge/`; packaged Electron resolves `resources/bridge/`.

## Voice Default

Source voice default: `hold-to-talk`; `wake-word` is explicit beta opt-in.

Code evidence: `rex_loop.py` declares `choices=("hold-to-talk", "wake-word")` and `default="hold-to-talk"`. The lower-level builder retains a backwards-compatible internal default, but the user-facing source entry point supplies Hold-to-Talk explicitly.

## OpenClaw Status

OpenClaw defaults: tools `false`, voice backend `false`; enabled mode requires URL + token.

The surface is `experimental`, not shippable. The Electron GUI can test `/healthz` and report gateway reachability, but authentication and tool capability remain unproven by health alone. Tool dispatch uses bounded retries; exhausted connection/auth/429/5xx failures emit `openclaw.tool_fallback` and execute locally. Policy denial (403) never falls around Rex policy.

## Docker Tier

Docker tier: `developer-only`; healthcheck is `python -m rex doctor --healthcheck`.

The Docker image is an operator/development smoke path, not the end-user distribution artifact and not a supported production deployment tier.

## Home Assistant Verification

HA mutation success is `verified` only after independent state observation. Dispatch acceptance by itself never becomes a confirmed user-facing success. `switch`, `light`, `lock`, and `cover` use expected-state polling evidence; results preserve `expected`, `actual`, and `latency_ms`. Sensitive actions require action-bound confirmation before dispatch.

## Conflicts Resolved in This Story

1. INSTALL distinguishes the shippable packaged installer from the development-only source Electron launch command.
2. INSTALL and RUNNING state the source voice default is Hold-to-Talk and wake-word is beta opt-in.
3. INSTALL and RUNNING classify OpenClaw as experimental/off by default instead of generic developer/advanced.
4. CLAUDE qualifies `pip install .` as the developer/operator source install rather than an unqualified install path.
5. CLAUDE's root `rex_loop.py` description matches the Hold-to-Talk default.
6. CLAUDE's OpenClaw section records default-off/fail-closed configuration, health-evidence scope, bounded fallback behavior, and the policy-denial exception.

## Linked Current-State Authorities

- [README](../README.md)
- [INSTALL](../INSTALL.md)
- [RUNNING](../RUNNING.md)
- [UI surfaces](UI_SURFACES.md)
- [Surface classification](../SURFACE-CLASSIFICATION.md)
- [Integration status](../INTEGRATIONS_STATUS.md)
- [CLAUDE](../CLAUDE.md)
- [OpenClaw status](openclaw-migration-status.md)
- [Docker](docker.md)
- [Home Assistant](home_assistant.md)
