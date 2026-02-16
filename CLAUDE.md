# CLAUDE.md

## Project Overview
Rex AI Assistant is a local-first, voice-activated AI companion. It supports wake word detection, speech-to-text, LLM chat, and text-to-speech, with optional integrations for search, messaging, email, calendar, and Home Assistant.

Primary goals: reliability, security by default, and smooth day-to-day usage on Windows 10/11, macOS, and Linux.

## Tech Stack and Conventions
### Language and runtime
- Language: Python 3.9 to 3.13 (3.10+ preferred)
- Packaging: pyproject.toml with setuptools backend, install via `pip install .`
- Entry points:
  - `rex` -> `rex.cli:main`
  - `rex-config` -> `rex.config:cli`
  - `rex-speak-api` -> `rex_speak_api:main`

### Core components
- API: Flask (Flask-CORS, Flask-Limiter)
- Config: Pydantic v2, python-dotenv
- STT: OpenAI Whisper (offline)
- Wake word: openWakeWord
- TTS: Coqui XTTS (voice cloning supported); optional edge-tts or pyttsx3
- LLM: local Transformers, OpenAI API, or Ollama
- Search: plugin-based (SerpAPI, Brave, Google CSE, DuckDuckGo)

### Style and quality
- Prefer clear, testable functions over clever code.
- Keep changes small and reviewable.
- Add logging for non-trivial behavior (use the existing logging utilities).
- Update or add tests when behavior changes.
- Avoid introducing new heavy dependencies unless clearly justified.

### Security
- Never commit secrets. Secrets belong in `.env` only.
- Runtime settings belong in `config/rex_config.json`. Secrets belong in `.env`.
- Prefer least-privilege defaults for any networked feature (TTS API, plugins, integrations).
- Treat all external inputs as untrusted (web content, email, SMS, plugin results).

## Repository Structure
Top-level directories and what they contain:
- `rex/` - main package (CLI, services, workflows, integrations)
- `scripts/` - operational scripts (health checks, utilities, tooling)
- `plugins/` - optional plugins (web search and other extensible features)
- `config/` - app configuration files (not secrets)
- `Memory/` - per-user memory profiles and stored context
- `tests/` - pytest suite
- `docs/` - additional documentation

Notable top-level modules and entrypoints:
- `rex_loop.py` - full voice loop (wake word -> STT -> LLM -> TTS)
- `voice_loop.py` - core voice loop helpers
- `wakeword_listener.py` - wake word listener utilities
- `rex_speak_api.py` - Flask TTS API with auth and rate limiting
- `run_gui.py` / `gui.py` - desktop GUI

## Commands
### Install
- Create venv:
  - Windows PowerShell: `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
  - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
- Base install: `python -m pip install --upgrade pip setuptools wheel` then `pip install .`
- Optional stacks:
  - CPU ML: `pip install -r requirements-cpu.txt`
  - GPU CUDA 12.4 (Windows): `pip install -r requirements-gpu-cu124.txt`
  - Dev tools: `pip install -r requirements-dev.txt`

### Run
- Health check: `python scripts/doctor.py`
- Text mode: `python -m rex`
- Voice mode: `python rex_loop.py`
- GUI: `python run_gui.py`
- TTS API: `python rex_speak_api.py` (or `rex-speak-api` if installed as a script)

### Test and lint
- Tests: `pytest`
- Targeted tests: `pytest tests/<file>.py`
- Format: `black .`
- Lint: `ruff check .`
- Types: `mypy .` (only if configured; keep type hints consistent)

## Rules
1. Read before writing.
   - Inspect existing modules and patterns before adding new ones.
   - Do not invent filenames or APIs that do not exist in this repo.

2. Respect the config split.
   - Secrets in `.env` only.
   - Runtime config in `config/rex_config.json`.

3. Windows compatibility matters.
   - Do not rely on `simpleaudio` on Windows.
   - Avoid dependencies that are known to fail on Windows unless optional and guarded.

4. Keep integrations optional.
   - Email, calendar, SMS, MQTT, Home Assistant, and web search must degrade gracefully if not configured.

5. Do not add network exposure by default.
   - Anything that binds to a port must be authenticated and rate limited.
   - Prefer localhost binding unless explicitly configured otherwise.

6. After changes, run at least:
   - `python check_imports.py`
   - `pytest` (or the most relevant subset)

## Common Anti-Patterns to Avoid
- Hardcoding paths that only work on one machine.
- Creating new config knobs in `.env` for non-secret runtime settings.
- Adding new dependencies for convenience when stdlib or existing deps already cover it.
- Writing code that assumes optional services exist (MQTT broker, OpenAI key, Ollama daemon).

## Working Agreements for Claude Code
- If requirements are ambiguous, propose a safe default and explain it briefly.
- Provide the full updated file when you modify code, not a partial diff.
- Keep outputs paste-ready and avoid non-printing Unicode characters.

## Code output rules (must follow)

- Do not provide truncated code. If you change a file, output the entire updated file unless the request explicitly asks for a diff.
- Do not use placeholders like "..." or "rest of file" or "omitted for brevity" in code.
- Do not claim something is implemented unless the code shown fully implements it.
- If a change spans multiple files, list every file you changed and output each file in full.
- If you cannot complete an implementation due to missing information, state exactly what is missing and provide the best safe partial implementation with clear TODO markers.
- Every PR must include a minimal verification step that can be run locally (command + expected outcome).


## Workflow feedback loop (must follow)

If the reviewer (Codex or a human) changes Claude’s output due to a recurring mistake or wrong assumption,
update CLAUDE.md in the same PR with a new or refined rule that would have prevented the issue.
Keep the rule short and specific.


## Maintenance rules for CLAUDE.md (must follow)

When you change the repo, you must keep this file accurate.

Update CLAUDE.md in the same PR when any of the following change:
- New commands, scripts, or make targets are added or existing ones change
- Project structure changes (new top-level folders, new services, renamed modules)
- Dependencies or runtime requirements change (Python version, system deps, Docker images)
- New environment variables, secrets, or config files are introduced
- New integrations are added (Home Assistant, IFTTT, Cloudflare Tunnel, etc.)
- Any “Rules” or “Conventions” in this file become outdated

Do not update CLAUDE.md for:
- Pure refactors that do not change behavior, commands, or structure
- Comment-only or formatting-only changes

PR requirements:
- Every PR must include a short “Verification” section describing how you tested the change
- If CLAUDE.md was updated, mention it explicitly in the PR description

