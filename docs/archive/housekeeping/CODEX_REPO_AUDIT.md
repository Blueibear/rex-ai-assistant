# Repository Audit

## 1. Executive Summary
This repository is not production-ready. It is an ambitious, heavily expanded prototype with several real subsystems and a broad test suite, but the repo as a whole is still prototype-grade, not beta-grade and not deploy-safe. The biggest problem is not lack of code. The biggest problem is that the repo says it can do more than its current planner, tool execution, packaging, and validation surfaces actually support.

Top 5 truths about the current state:
1. The codebase is much larger and more real than a hobby toy. `rex/` contains real service layers, a large CLI, workflow and policy plumbing, dashboard APIs, multiple backends, and meaningful tests.
2. The actual autonomous execution surface is much narrower than the docs imply. `rex/tool_router.py::execute_tool` only supports `time_now`, `weather_now`, and `web_search`, and the last two explicitly return "not implemented". Planner patterns in `rex/planner.py` still generate `send_email`, `calendar_create_event`, and `home_assistant_call_service`.
3. The test suite is strong enough to prove there is real engineering here. `.\.venv\Scripts\python.exe -m pytest -m "not slow and not audio and not gpu"` produced `1637 passed, 26 skipped, 2 failed`, and the two failures were both repo-integrity tests tripping over the pre-existing dirty tracked file `requirements-gpu-cu124.txt`.
4. Packaging, quality, and validation are not green. `ruff check rex` reported 339 errors, `black --check rex` would reformat 37 files, `mypy rex` found 272 errors in 53 files, and `scripts/validate_deployment.py` still expects an older torch/runtime model than the repo currently uses.
5. Documentation and status reporting are materially unreliable. `docs/autonomy.md`, `docs/tools.md`, `docs/ARCHITECTURE.md`, `README.windows.md`, `README.md`, `STABILIZATION_REPORT.md`, and the GitHub workflows do not agree with the executable reality.

## 2. What This Repo Actually Is
This repo is trying to be a local-first personal AI assistant platform with optional cloud integrations, multiple UI surfaces, and a policy-gated automation engine. The intended direction is visible in the architecture: a core assistant, a workflow planner/executor, persistent memory, optional voice input/output, web APIs, and service adapters for email, calendar, messaging, browser automation, OS automation, remote computers, Home Assistant, GitHub, WordPress, and WooCommerce.

In plain English for a non-developer: this is a smart-assistant toolkit that can chat, show a dashboard, keep local memory, run a few safe automated tasks, and connect to outside services if they are configured. Right now it behaves more like a large experimental platform than a finished assistant product.

Architecturally, the repo is split into:
- A canonical package in `rex/` that contains the real runtime code.
- Root-level Python files that are a mix of genuine compatibility wrappers (`config.py`, `llm_client.py`) and full alternate entry surfaces (`voice_loop.py`, `flask_proxy.py`, `rex_speak_api.py`, `gui.py`).
- A large test suite in `tests/`.
- A large documentation/report/planning surface in the repo root and `docs/`.
- Gitignored runtime state such as `.env`, `config/rex_config.json`, `data/workflows/`, `data/approvals/`, `data/notifications/`, and `Memory/`.

## 3. Verified Current Capabilities

### C-01. Package CLI and diagnostics
- Capability name: Package CLI entrypoint and diagnostics commands
- Evidence in code: `rex/__main__.py`, `rex/cli.py::main`, `pyproject.toml [project.scripts]`
- How it appears to work: `python -m rex --help`, `python -m rex version`, `python -m rex doctor`, and `python -m rex tools` all executed successfully in this environment.
- Confidence level: High
- Status: Working

### C-02. Runtime config loading plus legacy-env migration warnings
- Capability name: JSON runtime config plus secret env loading
- Evidence in code: `rex/config.py::load_config`, `rex/config_manager.py::load_config`, `rex/config_manager.py::get_legacy_env_warnings`, `utils/env_loader.py`
- How it appears to work: the CLI loads `config/rex_config.json`, reads `.env`, and emits warnings for ignored legacy env vars such as `OLLAMA_HOST`.
- Confidence level: High
- Status: Working

### C-03. Workflow planning and execution for the `time_now` tool
- Capability name: Rule-based planning and low-risk workflow execution
- Evidence in code: `rex/planner.py`, `rex/executor.py`, `rex/workflow_runner.py`, `rex/tool_router.py::execute_tool`
- How it appears to work: `.\.venv\Scripts\python.exe -m rex plan "what's the time in Dallas, TX" --execute` generated a one-step workflow, saved it under `data/workflows/`, validated it, and completed execution successfully.
- Confidence level: High
- Status: Working

### C-04. Tool registry status reporting
- Capability name: Tool inventory and health/credential reporting
- Evidence in code: `rex/tool_registry.py`, `rex/cli.py::cmd_tools`
- How it appears to work: `rex tools` lists registered tools and distinguishes `[READY]`, `[NO CREDS]`, and `[UNHEALTHY]`.
- Confidence level: High
- Status: Working

### C-05. Dashboard web UI, auth, settings API, scheduler API, chat API, and notification SSE surface
- Capability name: Dashboard HTTP surface
- Evidence in code: `rex/dashboard/routes.py`, `rex/dashboard/auth.py`, `rex/dashboard/sse.py`, `rex/dashboard/static/`, `rex/dashboard/templates/`, `tests/test_dashboard.py`
- How it appears to work: targeted tests for dashboard routes and auth passed (`tests/test_dashboard.py`), and the module includes real Flask routes, session management, redaction logic, and frontend assets.
- Confidence level: High
- Status: Likely Working

### C-06. Flask proxy API with identity and plugin-backed `/search`
- Capability name: Main Flask proxy API
- Evidence in code: `flask_proxy.py`, `tests/test_flask_proxy.py`
- How it appears to work: tests cover `/whoami` and `/search`; the proxy loads user memory, enforces identity checks, wires the dashboard blueprint, and optionally exposes plugin-backed search.
- Confidence level: Medium
- Status: Likely Working

### C-07. TTS HTTP API with auth and rate limiting
- Capability name: `rex_speak_api.py` Flask TTS service
- Evidence in code: `rex_speak_api.py`, `wsgi.py`, `tests/test_rex_speak_api.py`
- How it appears to work: the API has auth, CORS, rate limiting, HA bridge wiring, text chunking, and optional TTS/audio dependency guards; the corresponding tests passed in the broad pytest run.
- Confidence level: Medium
- Status: Likely Working

### C-08. Local data features: memory, knowledge base, scheduler, reminders, and cues
- Capability name: Local persistence and user-memory features
- Evidence in code: `rex/memory.py`, `rex/knowledge_base.py`, `rex/scheduler.py`, `rex/reminder_service.py`, `rex/cue_store.py`, tests such as `tests/test_memory.py`, `tests/test_knowledge_base.py`, `tests/test_scheduler.py`, `tests/test_followup_engine.py`
- How it appears to work: the CLI and service layers are real, and the pytest suite covers the data structures and command handlers.
- Confidence level: Medium
- Status: Likely Working

### C-09. Email service with stub mode and real IMAP/SMTP backend path
- Capability name: Email send/read abstractions
- Evidence in code: `rex/email_service.py`, `rex/email_backends/imap_smtp.py`, `rex/email_backends/account_router.py`, `tests/test_email_service_backend.py`, `tests/test_email_backend_imap_smtp.py`, `tests/test_cli_email_accounts.py`
- How it appears to work: the service supports offline stub mode by default and has a real backend path for IMAP/SMTP plus account routing; the implementation is covered by offline tests, not live credential tests.
- Confidence level: Medium
- Status: Likely Working

### C-10. Calendar service with stub mode and ICS read-only backend
- Capability name: Calendar read/write local mode plus ICS ingestion
- Evidence in code: `rex/calendar_service.py`, `rex/calendar_backends/ics_backend.py`, `rex/calendar_backends/factory.py`, `tests/test_calendar_ics_backend.py`, `tests/test_calendar_service.py`
- How it appears to work: stub event persistence is real, and the ICS backend supports local files and HTTPS sources with SSRF-style restrictions.
- Confidence level: Medium
- Status: Likely Working

### C-11. Messaging service, Twilio backend, inbound webhook store, and notification routing
- Capability name: SMS/messaging plus notification delivery framework
- Evidence in code: `rex/messaging_service.py`, `rex/messaging_backends/twilio_backend.py`, `rex/messaging_backends/inbound_webhook.py`, `rex/notification.py`, `rex/dashboard_store.py`, tests such as `tests/test_messaging_service.py`, `tests/test_inbound_webhook.py`, `tests/test_notification.py`
- How it appears to work: stub messaging is real, Twilio send and inbound webhook code exist, notifications persist to SQLite and support digests/escalation, and email delivery can use `EmailService.send()` when a real backend is active.
- Confidence level: Medium
- Status: Likely Working

### C-12. Browser automation service
- Capability name: Playwright-backed browser automation
- Evidence in code: `rex/browser_automation.py`, `tests/test_browser_automation.py`
- How it appears to work: the service manages sessions, screenshots, and scripted actions, with optional Playwright import guards and passing targeted tests.
- Confidence level: Medium
- Status: Likely Working

### C-13. OS automation service
- Capability name: Allowlisted shell and file operations with policy checks
- Evidence in code: `rex/os_automation.py`, `tests/test_os_automation.py`
- How it appears to work: the service allows safe commands and file operations under `data/`, logs actions, and enforces policy before execution.
- Confidence level: Medium
- Status: Likely Working

### C-14. GitHub and VS Code-style service layers
- Capability name: GitHub API helpers and code-file/test utilities
- Evidence in code: `rex/github_service.py`, `rex/vscode_service.py`, `tests/test_github_service.py`, `tests/test_vscode_service.py`
- How it appears to work: the services support repository/PR/issue calls, file reads, patch application, and pytest execution through service facades. Coverage is offline and mocked.
- Confidence level: Medium
- Status: Likely Working

### C-15. Remote Windows computer client and agent server
- Capability name: Remote command/status via HTTP agent
- Evidence in code: `rex/computers/client.py`, `rex/computers/service.py`, `rex/computers/agent_server.py`, `docs/computers.md`, `tests/test_windows_agent.py`, `tests/test_computers.py`
- How it appears to work: the repo contains both the client side and the Flask agent server side, including auth, allowlists, rate limiting, and CLI commands.
- Confidence level: High
- Status: Likely Working

### C-16. WordPress and WooCommerce integration layers
- Capability name: WordPress health monitoring and WooCommerce read/write clients
- Evidence in code: `rex/wordpress/client.py`, `rex/wordpress/service.py`, `rex/woocommerce/client.py`, `rex/woocommerce/service.py`, `tests/test_wordpress.py`, `tests/test_woocommerce.py`
- How it appears to work: WordPress health checks and WooCommerce read/write API calls are implemented behind service classes and covered by targeted tests.
- Confidence level: High
- Status: Likely Working

### C-17. Web search plugin outside the workflow tool path
- Capability name: Plugin-level web search with provider fallbacks
- Evidence in code: `plugins/web_search.py`, `tests/test_web_search_plugin.py`, `tests/test_flask_proxy.py`
- How it appears to work: the plugin can use SerpAPI, Brave, Google CSE, DuckDuckGo scraping, and Browserless, and `/search` in `flask_proxy.py` can call it.
- Confidence level: Medium
- Status: Likely Working

### C-18. Voice loop implementation and optional dependency guards
- Capability name: Wake-word -> STT -> LLM -> TTS pipeline code
- Evidence in code: `rex/voice_loop.py`, `voice_loop.py`, `rex_loop.py`, `rex/voice_loop_optimized.py`, tests such as `tests/test_voice_loop.py`, `tests/test_voice_loop_optional_imports.py`
- How it appears to work: there is substantial voice loop code, import guards, wake-sound generation, and tests, but this audit did not validate end-to-end audio hardware or model behavior.
- Confidence level: Medium
- Status: Unclear

## 4. Claimed or Implied Capabilities That Are Not Yet Reliable

### R-01. Multi-step autonomous workflows beyond `time_now`
`docs/autonomy.md` presents newsletter, calendar, home-control, report, and search workflows as executable. `rex/planner.py` does generate those tool names, but `rex/tool_router.py::execute_tool` only supports `time_now`, `weather_now`, and `web_search`, and the latter two explicitly return "not implemented". `send_email`, `calendar_create_event`, and `home_assistant_call_service` hit policy approval first and then fall through to "Unknown tool" when policy is bypassed.

### R-02. `web_search` as a first-class workflow tool
The plugin in `plugins/web_search.py` is real, and `/search` in `flask_proxy.py` can use it, but the workflow/tool path still treats `web_search` as not implemented. The docs do not distinguish between "plugin-backed search endpoint" and "tool-router backed autonomous tool."

### R-03. Scheduler-triggered workflow execution
`rex/scheduler.py` supports `workflow_id` fields, but the implementation explicitly says workflow triggering is stubbed and only logs "would trigger workflow". The scheduler is usable for callbacks, not for the workflow integration that the data model suggests.

### R-04. Audit replay as a real recovery/debug feature
`rex/replay.py` explicitly says it is a stub, reconstructs a `ToolCall`, and returns a placeholder result instead of replaying actual tool execution.

### R-05. Current Docker deployment guidance
`README.md` includes `docker run --env-file .env ...` examples, but the repo's Docker packaging is not trustworthy as-is. `.dockerignore` is too narrow, `Dockerfile` copies the full repo, and the dependency matrix in the Docker build does not match the rest of the repo.

### R-06. Windows quickstart as written
`README.windows.md` tells users to configure via environment variables and says `python rex_assistant.py` starts a wake-word assistant. In reality, `rex_assistant.py` starts a text chat loop, and the current config model is primarily `config/rex_config.json` plus secret-only `.env`.

### R-07. Production-ready / fully stabilized status reports
Files such as `STABILIZATION_REPORT.md` claim the codebase is production-ready and that multiple issues are false alarms. Those claims do not survive contact with current lint/type/deployment results or the planner/tool-router mismatch.

## 5. Architecture Review

### Project structure
- Canonical package: `rex/`
- Root-level entry surfaces and wrappers: `rex_assistant.py`, `rex_loop.py`, `voice_loop.py`, `flask_proxy.py`, `rex_speak_api.py`, `gui.py`, plus compatibility wrappers like `config.py` and `llm_client.py`
- Plugins: `plugins/`
- Tests: `tests/`
- Scripts: `scripts/`
- Docs and reports: `README.md`, `README.windows.md`, `CONFIGURATION.md`, `RUNNING.md`, many verification and stabilization reports, and `docs/`
- Runtime-local state: `.env`, `config/rex_config.json`, `data/`, `Memory/`, `logs/`, `transcripts/`, most of which are gitignored

### Major modules
- Core assistant/runtime: `rex/assistant.py`, `rex/app.py`, `rex/services.py`
- Config/credentials: `rex/config.py`, `rex/config_manager.py`, `rex/credentials.py`, `utils/env_loader.py`
- Planning/execution/policy: `rex/planner.py`, `rex/executor.py`, `rex/workflow.py`, `rex/workflow_runner.py`, `rex/policy.py`, `rex/policy_engine.py`, `rex/autonomy_modes.py`
- Tools: `rex/tool_registry.py`, `rex/tool_router.py`
- Voice: `rex/voice_loop.py`, `voice_loop.py`, `rex/voice_loop_optimized.py`, wake-word helpers
- Web/API: `flask_proxy.py`, `rex/dashboard/*`, `rex_speak_api.py`
- Integrations: email, calendar, messaging, notifications, browser, OS, GitHub, VS Code, computers, WordPress, WooCommerce, Home Assistant TTS

### Execution flow
- `python -m rex` enters through `rex/__main__.py` into `rex/cli.py::main`.
- Planner flow is `Planner` -> `Workflow` -> `Executor` / `WorkflowRunner` -> `tool_router.execute_tool`.
- The dashboard and main HTTP API are served from `flask_proxy.py`, which registers the dashboard blueprint from `rex/dashboard`.
- The TTS API is a separate Flask app in `rex_speak_api.py`.
- Voice startup is split across `rex_loop.py`, root `voice_loop.py`, and `rex/voice_loop.py`.

### Key dependencies
- Core: Flask, flask-cors, flask-limiter, requests, pydantic, python-dotenv, BeautifulSoup
- Optional/feature-specific: torch, torchvision, torchaudio, whisper, openwakeword, Coqui TTS, OpenAI, Ollama, Playwright, Twilio
- System assumptions: ffmpeg, git, optional CUDA, optional microphone/speakers

### Integrations
- Email: stub plus real IMAP/SMTP path
- Calendar: stub plus ICS read-only path
- Messaging: stub plus Twilio send/inbound path
- Notifications: dashboard store, email/SMS/HA-TTS routing
- Browser/OS/GitHub/VSCode/computers/WordPress/WooCommerce: real service classes with offline tests
- Search: plugin path exists separately from the tool router

### Configuration model
- Intended current model: `config/rex_config.json` for runtime settings, `.env` for secrets
- Profiles: `profiles/*.json`
- Credential lookup: `rex/credentials.py`
- Reality: the code mostly follows the JSON runtime config model, but large parts of the docs and auxiliary scripts still speak in env-var terms

### Startup/runtime assumptions
- Project root exists and is writable for runtime data under `data/`, `logs/`, `transcripts/`, `Memory/`
- `config/rex_config.json` may be auto-created by `rex/config_manager.py::load_config`
- Audio and ML features require optional dependencies and local hardware/model state
- Dashboard and proxy can allow local bypass auth in some configurations

### Test structure
- Broad pytest suite with 1665 collected tests in the main filtered run
- Strong targeted subsystem coverage for dashboard, browser, OS automation, GitHub, VS Code, Windows agent, WordPress, WooCommerce
- Offline-first tests for most integrations using mocks and temp files
- Two repository-integrity tests assume a clean worktree and fail if any tracked file is already dirty

### Deployment assumptions
- Dockerfile exists but is not trustworthy as a deployment artifact yet
- GitHub Actions CI targets `master`, while release automation targets `main`
- Windows README assumes env-driven setup that no longer matches the main config direction

### Architectural strengths
- Broad automated coverage
- Clear modular boundaries for many integrations
- Real workflow/policy/audit abstractions
- Optional dependency guards are generally present
- Several subsystems are already more real than the README's beta/stub framing suggests

### Architectural weaknesses
- Too many overlapping entry surfaces
- The planner/registry/router contract is broken
- Documentation is drifting from code
- Repo root mixes code, reports, logs, and operational artifacts
- Quality gates and deployment artifacts are not aligned with the current codebase

## 6. Weaknesses, Risks, and Failure Points

### Correctness

#### COR-001. Planner, registry, and router disagree on what is executable
- Severity: High
- Evidence:
  - `rex/planner.py` generates `send_email`, `calendar_create_event`, `home_assistant_call_service`, `weather_now`, `time_now`, and `web_search`.
  - `rex/tool_registry.py` registers `time_now`, `weather_now`, `web_search`, `send_email`, and `home_assistant`.
  - `rex/tool_router.py::execute_tool` hard-codes `supported_tools = {"time_now", "weather_now", "web_search"}` and returns "Unknown tool" for anything else.
  - Direct execution check:
    - `time_now` returned a real result.
    - `weather_now` and `web_search` returned "Tool ... is not implemented".
    - `send_email`, `calendar_create_event`, and `home_assistant_call_service` raised `ApprovalRequiredError`, then returned "Unknown tool ..." when called with `skip_policy_check=True`.
- Why it matters: the core automation promise is internally inconsistent. Plans can validate or be described in docs even though the executor cannot carry them out.
- Exact suggested fix: define one authoritative executable tool catalog, make `Planner` consult only that catalog, wire real handlers for any documented tools, and add integration tests that prove every planner-emitted tool runs through `execute_tool`.
- Scope: large

### Security

#### SEC-001. Docker build context can capture secrets and local state
- Severity: Critical
- Evidence:
  - `.dockerignore` only excludes a handful of paths and does not exclude `.env`, `.venv`, `venv/`, `config/`, `data/`, `tests/`, or local caches beyond `__pycache__/`.
  - `Dockerfile` does `COPY . .` in the runtime stage.
  - This repo currently has a local `.env` and gitignored runtime config/state present.
- Why it matters: a normal `docker build` from a developer workstation can embed secrets, local config, local workflow history, and virtualenv contents into the image or at least send them to the build context.
- Exact suggested fix: replace broad `COPY . .` with a whitelist of runtime files, expand `.dockerignore` to exclude all secrets, virtualenvs, caches, docs, tests, local data, and gitignored state, and document which runtime mounts are expected instead of baking workstation state into the image.
- Scope: medium

### Dependency management

#### DEP-001. Dependency and packaging artifacts disagree on the supported runtime matrix
- Severity: High
- Evidence:
  - `requirements-cpu.txt` is labeled CPU-only but pins `torch==2.7.1+cu118`.
  - `Dockerfile` installs `torch==2.7.1`, `torchvision==0.20.1`, and `torchaudio==2.5.1`, which is a mismatched family relative to other manifests.
  - `pyproject.toml` allows `torch>=2.6.0,<2.9.0`, `torchvision>=0.17.0,<0.23.0`, `torchaudio>=2.6.0,<2.9.0`.
  - `scripts/validate_deployment.py` still expects torch `2.5.x`.
- Why it matters: installation instructions, CI assumptions, and deployment scripts are not describing the same environment. That makes reproducible installs and debugging unnecessarily difficult.
- Exact suggested fix: declare one canonical supported matrix for base, CPU, cu118, and cu124 installs; sync `pyproject.toml`, requirements files, Pipfile notes, Dockerfile, doctor/validation scripts, and setup docs to that matrix; remove obviously wrong labels like the CUDA wheel in `requirements-cpu.txt`.
- Scope: medium

### Code quality

#### QLT-001. Core quality gates are not green
- Severity: High
- Evidence:
  - `.\.venv\Scripts\python.exe -m ruff check rex` -> `Found 339 errors.`
  - `.\.venv\Scripts\python.exe -m black --check rex` -> `37 files would be reformatted, 85 files would be left unchanged.`
  - `.\.venv\Scripts\python.exe -m mypy rex` -> `Found 272 errors in 53 files (checked 121 source files)`.
- Why it matters: the repo cannot honestly claim a stabilized or release-ready state while the package-level lint, format, and type-check baselines are this far from green.
- Exact suggested fix: create a staged cleanup plan: first fix import/order and obvious unused-symbol issues, then address structural type errors in core modules (`rex/voice_loop.py`, `rex/llm_client.py`, `rex/config.py`, `rex/service_supervisor.py`), and finally enforce the cleaned baseline in CI at least for touched files.
- Scope: large

### Testing

#### TST-001. Repo-integrity tests are brittle and fail on pre-existing dirtiness
- Severity: Medium
- Evidence:
  - `tests/test_repo_integrity.py` and `tests/test_repository_integrity.py` both call `git status --porcelain` and assume any tracked modification is test-induced.
  - `git status --short` before and after the audit showed only `M requirements-gpu-cu124.txt`.
  - Main pytest run result: `2 failed, 1637 passed, 26 skipped`; both failures were these integrity tests.
- Why it matters: these tests do not measure whether the test suite dirtied the repo. They measure whether the repo was already clean, which makes the signal noisy and breaks normal use on a dirty branch.
- Exact suggested fix: snapshot tracked-file status at test start and compare against that baseline, or restrict the check to files touched during the test session. Do not fail just because the working tree was already dirty.
- Scope: small

### Developer experience

#### OPS-001. `scripts/security_audit.py` generates large false-positive noise
- Severity: Medium
- Evidence:
  - `.\.venv\Scripts\python.exe scripts/security_audit.py` scanned `10398` files.
  - It reported `295 placeholder findings (242 code-critical, 51 doc-acceptable, 2 needs-review)`.
  - The reported "code-critical" findings included `.mypy_cache/...` files rather than actual repo source.
- Why it matters: a security audit script that classifies cache files as code-critical trains maintainers to ignore its output.
- Exact suggested fix: scan tracked files or a curated source-file allowlist, exclude caches and generated directories (`.mypy_cache`, `.pytest_cache`, `.ruff_cache`, build output, virtualenvs), and split "source findings" from "documentation findings" cleanly.
- Scope: small

#### DX-001. The repo root mixes product code, legacy wrappers, reports, and local operational files
- Severity: Low
- Evidence:
  - The root contains product code, wrappers, installers, scripts, verification reports, crash logs, helper demos, and compatibility modules all side-by-side.
  - Examples include `gui.py`, `flask_proxy.py`, `rex_speak_api.py`, `voice_loop.py`, `STABILIZATION_REPORT.md`, `gui_debug.log`, `FINAL_SUMMARY.txt`, and many verification reports.
- Why it matters: discoverability is poor, drift is easier to introduce, and it is harder to tell which surfaces are current, historical, or legacy.
- Exact suggested fix: move historical reports to `docs/reports/`, consolidate operational scripts under `scripts/`, keep only deliberate entrypoints at the root, and clearly separate compatibility wrappers from active runtime modules.
- Scope: medium

### Deployment/reliability

#### OPS-002. `scripts/validate_deployment.py` is stale and contradicts the current repo
- Severity: Medium
- Evidence:
  - `.\.venv\Scripts\python.exe scripts/validate_deployment.py` exited non-zero with `Score: 5/7 checks passed`.
  - It flagged missing `REX_ACTIVE_USER` as an env requirement even though the current config direction is JSON runtime config plus secret-only `.env`.
  - It warned `PyTorch 2.6.0+cu124 (expected 2.5.x)`, which no longer matches the rest of the repo.
- Why it matters: operators using the project's own validation script will be told the repo is wrong even when they are using the current dependency/config model.
- Exact suggested fix: rewrite the deployment validator around the actual supported configuration model and dependency matrix; make it verify `config/rex_config.json` and current torch ranges instead of stale env and version assumptions.
- Scope: small

### Documentation

#### DOC-001. Runtime configuration documentation is materially inconsistent with the code
- Severity: High
- Evidence:
  - `.env.example` says runtime config is in `config/rex_config.json` and `.env` is secrets only.
  - `README.md` still has large env-var configuration sections and describes the GUI as an editor for "all environment variables".
  - `README.md` and `README.windows.md` still instruct users to configure runtime behavior via env vars.
  - `rex/config_manager.py` emits warnings that legacy non-secret env vars are ignored at runtime.
- Why it matters: configuration is the first thing operators touch. Conflicting setup instructions guarantee misconfiguration and wasted debugging time.
- Exact suggested fix: pick one documented runtime model, then rewrite `README.md`, `README.windows.md`, `RUNNING.md`, GUI wording, and deployment docs around it. Keep `.env` documented as secrets-only unless a setting is genuinely still env-backed.
- Scope: medium

#### DOC-002. Windows quickstart and startup guidance are wrong
- Severity: High
- Evidence:
  - `README.windows.md` tells users to run `python rex_assistant.py` and then "Say your wake word".
  - Actual runtime check: piping `quit` into `python rex_assistant.py` produced `Rex assistant ready. Type 'exit' or 'quit' to stop.`, which is a text chat loop.
  - `README.windows.md` still says configure via environment variables and says web search "defaults to DuckDuckGo scraping", which is true for the plugin path but not for the workflow tool path.
- Why it matters: the Windows guide is an onboarding document. Right now it sends users into the wrong entrypoint with the wrong mental model.
- Exact suggested fix: rewrite the guide so it distinguishes text chat, full voice loop, dashboard, and TTS API; update config instructions to match the JSON runtime model; and explicitly explain the difference between plugin-backed search and tool-router-backed automation.
- Scope: small

#### DOC-003. Architecture, release, and status documents are stale or false
- Severity: Medium
- Evidence:
  - `docs/ARCHITECTURE.md` says `rex/voice_loop.py` is a compatibility wrapper to `rex/voice_loop_optimized.py`; the actual file contains a full implementation.
  - `.github/workflows/ci.yml` targets `master`, while `.github/workflows/release-please.yml` targets `main`.
  - `README.md` says merging to `main` triggers releases.
  - `STABILIZATION_REPORT.md` says the project is production-ready and claims dependency facts that no longer match the repo.
- Why it matters: stale "authoritative" documents are worse than missing documents because they actively mislead maintainers and reviewers.
- Exact suggested fix: mark old reports as historical, move them to an archive location, update `docs/ARCHITECTURE.md` to describe the current voice-loop reality, and make branch/release docs match the actual branch strategy or vice versa.
- Scope: medium

### Architecture

#### ARC-001. Voice loop and root-level entry surfaces are duplicated and drifting
- Severity: Medium
- Evidence:
  - `voice_loop.py` at the repo root is a full implementation, not a wrapper.
  - `rex/voice_loop.py` is also a full implementation.
  - `rex/voice_loop_optimized.py` and `docs/ARCHITECTURE.md` both present the optimized module as the canonical implementation.
  - `mypy rex` reports multiple voice-loop related type errors, and the doc story does not match the runtime story.
- Why it matters: duplicated "canonical" implementations guarantee future drift, especially in a voice pipeline that already depends on many optional imports and runtime assumptions.
- Exact suggested fix: choose one canonical voice-loop implementation, keep only thin wrappers around it if backward compatibility is required, and remove or deprecate the competing copies. Then update docs and tests to reference that single path.
- Scope: large

### Incomplete functionality

#### INC-001. Audit replay is explicitly stubbed
- Severity: Medium
- Evidence:
  - `rex/replay.py` says "STUB IMPLEMENTATION" and returns a placeholder result instead of replaying tools.
- Why it matters: if replay is exposed as part of the audit/debugging story, users will assume it re-executes behavior when it does not.
- Exact suggested fix: either implement replay through the existing tool router in dry-run mode with a clear side-effect policy, or clearly de-scope it from active documentation and user-facing claims.
- Scope: medium

#### INC-002. Scheduler workflow triggering is explicitly stubbed
- Severity: Medium
- Evidence:
  - `rex/scheduler.py` says workflow triggering is stubbed.
  - The execution path logs `would trigger workflow` when a job has `workflow_id` but no callback.
- Why it matters: the data model suggests workflows can be scheduled, but the scheduler does not currently execute them.
- Exact suggested fix: either wire scheduler jobs into `WorkflowRunner` or remove/disable the `workflow_id` scheduling story until that integration exists.
- Scope: medium

## 7. Contradictions and Drift
- Docs and code disagree on runtime configuration. `.env.example` and `CONFIGURATION.md` center JSON runtime config, while `README.md`, `README.windows.md`, and GUI wording still talk like env vars are the main runtime settings surface.
- The autonomy docs over-promise the executor. `docs/autonomy.md` describes calendar, home-control, newsletter, report, and search workflows as executable; `rex/tool_router.py::execute_tool` proves they are not.
- The tools docs overstate `web_search` as a usable registered tool. `docs/tools.md` treats it like a normal built-in tool, but the current tool-router path returns "Tool web_search is not implemented".
- The architecture docs say `rex/voice_loop.py` is a wrapper around `rex/voice_loop_optimized.py`. The file itself contains full implementation logic, and the repo root also contains another full `voice_loop.py`.
- The Windows quickstart says `python rex_assistant.py` starts a wake-word assistant. It actually starts a text chat loop.
- README claims some integrations are stub-only when the code is actually further along. Email has real IMAP/SMTP backend code, calendar has ICS read-only backend code, messaging has Twilio code, notifications can call real email delivery when configured, and the Windows agent server is implemented.
- README claims releases happen from `main`, while CI runs on `master` and release automation listens to `main`.
- `requirements-cpu.txt` is labeled CPU-only but pins a CUDA wheel. The filename and the content do not agree.
- `STABILIZATION_REPORT.md` says the codebase is production-ready and that several concerns are false alarms. Current lint/type/deployment evidence does not support that conclusion.

## 8. Unknowns and Validation Gaps

### U-01. Live external integrations
- What remains uncertain: real email, ICS-over-network, Twilio, Home Assistant, GitHub, WordPress, and WooCommerce behavior against live services
- Why: this audit had code and offline tests, but no live credentials or live endpoints
- Exact validation step: run credentialed smoke tests per integration in a disposable environment and capture real request/response traces without secrets

### U-02. Full voice pipeline on real hardware
- What remains uncertain: wake-word reliability, STT latency/accuracy, TTS playback quality, and long-running microphone stability
- Why: the code and tests exist, but this audit did not exercise microphone/speaker hardware or model files end-to-end
- Exact validation step: run `rex_loop.py` or the canonical voice-loop entrypoint on a machine with the intended microphone, speaker, and model setup; record startup, wake detection, STT, LLM, and TTS success/failure over repeated interactions

### U-03. Docker image build and runtime behavior
- What remains uncertain: whether the current Dockerfile builds successfully, starts correctly, and behaves safely with real runtime mounts
- Why: this audit inspected the Docker artifacts statically but did not build or run the image
- Exact validation step: build the image from a clean clone with a scrubbed environment, inspect the image contents, and run a containerized smoke test for the chosen entrypoint

### U-04. Tk GUI behavior beyond importability
- What remains uncertain: whether `gui.py` is stable and consistent with the current config model
- Why: the GUI imported successfully, but there is no equivalent targeted runtime validation like the dashboard has
- Exact validation step: launch `python run_gui.py` on Windows, exercise the dashboard and settings tabs, and confirm they read/write the intended config surfaces without stale env-var assumptions

### U-05. Long-running service supervision
- What remains uncertain: long-lived stability of `rex/app.py`, event bus, scheduler, notifier, and workflow runner under sustained usage
- Why: the suite is strong on unit and functional tests, but this audit did not run a soak test
- Exact validation step: run the supervised app with representative jobs and notification traffic for multiple hours, capturing logs, memory usage, and failure recovery

### U-06. Windows service wrapper
- What remains uncertain: real behavior of `rex/windows_service.py`
- Why: it is present, but the current environment did not validate pywin32-backed Windows service installation or operation
- Exact validation step: install the service wrapper on a Windows machine with pywin32, start/stop the service, and verify expected process supervision behavior

## 9. Production Readiness Assessment

### Local development use
Ready with caveats. The repo installs in the current virtualenv, the main filtered pytest run is overwhelmingly green, the CLI works, and targeted subsystem tests pass. A developer can work on this repo effectively.

### Personal daily use
Conditionally usable for narrow, manually supervised workflows. Text chat, dashboard/API work, and some local services are plausible. Broad autonomous behavior is not trustworthy enough to rely on without knowing exactly which code path you are using.

### Limited internal use
Not ready as a unified assistant platform. A small team could use selected subsystems for testing or demos, but not the repo's full advertised surface. The planner/tool mismatch and docs/config drift are too large.

### Beta testing
Not ready as a full-product beta. Some individual subsystems are beta-like, but the repo-level experience is still too internally inconsistent.

### Production use
Not ready. The repo fails basic truthfulness and packaging tests for production: the executor surface is narrower than documented, quality gates are not green, deployment artifacts drift, and Docker packaging is unsafe as written.

## 10. Prioritized Fix Plan

### Critical blockers
- P0 / SEC-001 - Lock down Docker packaging
  - Why this should happen now: current container builds can absorb secrets and local state.
  - Dependencies or prerequisites: none
  - Expected outcome: a buildable image that does not leak workstation data and has a predictable runtime footprint.

### High-value stability fixes
- P1 / COR-001 - Make the exposed automation surface truthful
  - Why this should happen now: it is the core product promise and the current planner/registry/router mismatch undermines trust immediately.
  - Dependencies or prerequisites: decide the canonical executable tool set.
  - Expected outcome: every documented planner-emitted tool either executes successfully or is removed from the exposed surface.
- P1 / DEP-001 - Unify dependency, packaging, and validation matrices
  - Why this should happen now: install and deployment guidance currently disagree.
  - Dependencies or prerequisites: decide the supported torch/platform matrix.
  - Expected outcome: requirements files, Dockerfile, pyproject extras, and validation scripts all describe the same environment.
- P1 / QLT-001 - Get package-level quality gates under control
  - Why this should happen now: the repo cannot credibly move toward beta while package lint/format/type baselines are this far off.
  - Dependencies or prerequisites: none beyond time and staging discipline.
  - Expected outcome: a manageable or green `ruff`, `black`, and `mypy` baseline for `rex/`.
- P1 / DOC-001 and DOC-002 - Rewrite the configuration and Windows onboarding docs
  - Why this should happen now: users will hit these docs before they hit the code.
  - Dependencies or prerequisites: settle the canonical config model and entrypoints.
  - Expected outcome: setup instructions match the actual runtime.

### Important quality improvements
- P2 / DOC-003 - Demote or correct stale architecture, release, and status documents
  - Why this should happen now: stale "authoritative" docs are causing active confusion.
  - Dependencies or prerequisites: decide the real branch strategy and canonical voice-loop path.
  - Expected outcome: maintainers can trust the repo docs again.
- P2 / TST-001 - Fix the repo-integrity tests
  - Why this should happen now: they currently produce false failures on dirty branches.
  - Dependencies or prerequisites: none
  - Expected outcome: integrity tests detect test-caused dirtiness rather than pre-existing worktree state.
- P2 / OPS-001 and OPS-002 - Repair the project's own validation scripts
  - Why this should happen now: current audit/deployment scripts are noisy or stale.
  - Dependencies or prerequisites: dependency/config matrix decisions from DEP-001.
  - Expected outcome: support scripts provide useful signal instead of misleading maintainers.
- P2 / ARC-001 - Consolidate the voice-loop and root-entry architecture
  - Why this should happen now: duplicate implementations are already drifting.
  - Dependencies or prerequisites: decide the canonical implementation.
  - Expected outcome: one clear voice path, fewer duplicate bugs, simpler docs.
- P2 / INC-002 - Either implement scheduler workflow execution or stop implying it exists
  - Why this should happen now: the current data model promises behavior that is still stubbed.
  - Dependencies or prerequisites: the workflow runner integration contract.
  - Expected outcome: scheduled workflows are either real or honestly absent.

### Nice-to-have improvements
- P3 / INC-001 - Implement or remove replay as an active feature
  - Why this should happen now: low urgency compared with truth-surface and deployment work.
  - Dependencies or prerequisites: clear dry-run semantics for tool replay.
  - Expected outcome: replay becomes either real or clearly archived/de-scoped.
- P3 / DX-001 - Clean up repo layout and historical artifacts
  - Why this should happen now: it improves maintainability but does not unblock correctness.
  - Dependencies or prerequisites: none
  - Expected outcome: clearer repo navigation and lower risk of future drift.

## 11. Strategic Roadmap

### Phase 1: Truth and Containment
- Objective: stop lying about what the repo can do and stop unsafe packaging.
- Tasks:
  - Fix SEC-001 by tightening `.dockerignore` and `Dockerfile`.
  - Fix DOC-001, DOC-002, and DOC-003 so setup, architecture, and release docs match reality.
  - Resolve branch naming drift between `.github/workflows/ci.yml`, `.github/workflows/release-please.yml`, and `README.md`.
  - Mark historical reports like `STABILIZATION_REPORT.md` as historical or archive them.
- Dependencies: none
- Exit criteria: packaging no longer leaks local state, and the top-level docs no longer contradict the code.
- Risks: maintainers may resist removing optimistic claims without equivalent feature completion.

### Phase 2: Execution Surface Completion
- Objective: make the planner/tooling story internally consistent.
- Tasks:
  - Fix COR-001 by choosing the authoritative executable tool set.
  - Wire real handlers for any tools that stay exposed, especially `send_email`, `web_search`, `calendar_create_event`, and Home Assistant calls, or remove those planner patterns and registry entries.
  - Fix INC-002 by either implementing scheduled workflow execution or disabling the feature surface that implies it.
- Dependencies: Phase 1 truth-surface cleanup
- Exit criteria: every documented automation path has a real executable backend and passing integration coverage, or it is clearly de-scoped.
- Risks: this phase can expand quickly if the team tries to implement every implied feature instead of narrowing the supported surface first.

### Phase 3: Quality and Validation Hygiene
- Objective: make the repo operationally defensible.
- Tasks:
  - Fix QLT-001 with staged lint/format/type cleanup.
  - Fix TST-001 so repo-integrity tests stop failing on dirty branches.
  - Fix OPS-001 and OPS-002 so security/deployment scripts provide useful signal.
  - Align the dependency matrix under DEP-001.
- Dependencies: the execution surface from Phase 2 should be stable enough to codify.
- Exit criteria: `pytest`, package-level `ruff`, package-level `black --check`, `mypy`, and project validation scripts are green or at least intentionally gated with an explicit baseline policy.
- Risks: cleanup work may expose deeper architectural problems in `rex/voice_loop.py`, `rex/llm_client.py`, and related modules.

### Phase 4: Release Preparation and Real-World Validation
- Objective: validate the chosen feature set under real conditions.
- Tasks:
  - Run live credentialed smoke tests for the integrations the project intends to support.
  - Build and run the Docker image from a clean environment.
  - Exercise the Tk GUI and long-running service supervision paths.
  - Decide the future of INC-001 replay and clean up DX-001 repo layout.
- Dependencies: Phases 1 through 3
- Exit criteria: the project can support a narrow, explicitly documented beta scope with live-tested integrations and sane deployment guidance.
- Risks: live integrations may fail in ways offline tests did not predict, forcing more backend hardening than expected.

## 12. Final Blunt Assessment
This repo is closest to a large prototype with several beta-quality subsystems inside it.

What is holding it back most is not "missing code" in the abstract. It is the gap between the repo's claims and the repo's executable truth: the planner and docs promise a broad automation platform, while the actual tool-router path really only delivers a small subset; the packaging and validation artifacts do not agree with each other; and the documentation layer is too stale to trust.

The smartest next move is to stop expanding the feature surface, make the exposed execution path truthful and internally consistent, fix the container/dependency/validation drift, and only then decide which narrow scope is actually ready for beta.
