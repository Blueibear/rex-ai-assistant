# AskRex Assistant

<p align="center">
  <img src="https://github.com/Blueibear/AskRex-Assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://www.buymeacoffee.com/Blueibear" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 28px !important;width: 120px !important;" ></a>
</p>

AskRex Assistant is a local-first, voice-activated AI companion that runs entirely on your machine — no cloud subscription required. It combines wake word detection, offline speech recognition via OpenAI Whisper, LLM-powered responses through Transformers, OpenAI, or Ollama, and text-to-speech synthesis, making it a practical choice for hobbyists, home-automation enthusiasts, and developers who want a private, customisable assistant.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Current Limitations](#current-limitations)
- [OpenClaw Integration](#openclaw-integration)
- [Docker](#docker)
- [Memory & Personalization](#memory--personalization)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Full Documentation](docs/INDEX.md)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Blueibear/AskRex-Assistant.git
   cd AskRex-Assistant
   ```

2. **Use Python 3.11 and create a virtual environment:**

   Windows (PowerShell):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Rex:**

   Windows full install:
   ```powershell
   .\install.ps1
   ```

   macOS / Linux full install:
   ```bash
   bash install.sh
   ```

   Windows GPU + TTS path:
   ```powershell
   pip install --upgrade pip setuptools wheel
   pip install -r requirements-gpu-cu124.txt
   ```

4. **Configure your model provider**. LM Studio remains optional. Ollama and OpenAI-compatible local servers also work. If using LM Studio, start the local server on `http://localhost:1234` and set your model in `config/rex_config.json`:
   ```json
   { "openai": { "base_url": "http://localhost:1234/v1", "model": "your-model-name" } }
   ```

5. **Run Rex and verify** — Rex prints `Rex assistant ready` and responds to your first message:
   ```bash
   rex
   python -m rex doctor
   ```

> **Python 3.11 is required. Python 3.12 and above are not supported.** The current dependency stack is validated on Python 3.11 only. Fresh installs on Python 3.12, 3.13, and 3.14 are rejected; the ML/TTS dependency path is known to fail on those versions.

> **Advanced / Developer Install** — for GPU setups, custom extras, Docker, or development workflows, see [docs/advanced-install.md](docs/advanced-install.md).
>
> **Want one consolidated guide?** See [docs/INSTRUCTION_MANUAL.md](docs/INSTRUCTION_MANUAL.md) for a single manual that combines install, configuration, usage, operations, and troubleshooting.

## Features

> **Alpha software** — core voice pipeline works today; integrations and advanced features vary by maturity (see labels below).

- 🔊 **Wake word detection** via openWakeWord (customizable trigger phrases) `[Works today]`
- 🗣️ **Speech-to-text** using OpenAI Whisper (runs offline) `[Works today]`
- 🤖 **LLM responses** via Transformers (local), OpenAI API, or Ollama `[Works today]`
- 🔉 **Text-to-speech** with Coqui XTTS, edge-tts, or pyttsx3 (voice cloning supported) `[Works today]`
- 🌐 **Web search plugins** for SerpAPI, Brave, Google CSE, and DuckDuckGo `[Requires configuration]`
- 🧠 **Per-user memory** profiles with conversation history and preferences `[In progress — not production ready]`
- 📧 **Email and calendar** integration with triage and scheduling `[Requires configuration — IMAP/SMTP credentials needed]`
- 📱 **Multi-channel messaging** via SMS `[Requires configuration — Twilio credentials needed]`
- 🔔 **Smart notifications** with priority routing, digest mode, quiet hours, and auto-escalation; dashboard channel persists to local SQLite store with real API endpoints `[Works today]`
- 🤖 **Autonomous workflows** with planner and workflow runner for multi-step task automation `[In progress — not production ready]`
- 🎯 **Smart planning** converts natural language goals into structured workflows `[In progress — not production ready]`
- ⚙️ **Configurable autonomy modes** (OFF/SUGGEST/AUTO) for fine-grained control `[Works today]`
- 🔐 **Flask TTS API** with authentication and rate limiting `[Works today]`
- ✅ **CI/CD** with GitHub Actions and Release Please automation `[Works today]`
- 🐳 **Docker support** for containerized deployment `[Works today]`

## Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | macOS 11+, Windows 10/11, or Ubuntu 20.04+ |
| **Python** | 3.11 (Python 3.12 and above are not supported) |
| **FFmpeg** | Must be installed and available on PATH |
| **Hardware** | Microphone and speakers for voice mode |
| **GPU** (optional) | NVIDIA GPU with CUDA 11.8+ for acceleration |

> **Note for Windows users**: The `simpleaudio` package (used for audio playback) has build issues on Windows and is automatically disabled. Audio playback functionality will be limited on Windows, but all core features work correctly.

## Configuration

Rex uses a dual-config system:

- **Secrets** (API keys, tokens) → `.env`
  Copy `.env.example` to `.env` and fill in the values you need. The file documents every supported secret with inline comments.
- **Runtime settings** (models, audio, wake word, feature flags) → `config/rex_config.json`
  Edit this file directly or use `rex-config` to manage it.

See [CONFIGURATION.md](CONFIGURATION.md) for the full reference including configuration precedence, migration from legacy env vars, and all available fields. For a complete list of supported environment variables see [docs/environment-variables.md](docs/environment-variables.md).

## Usage

Rex supports text chat, voice mode, GUI configuration, audio device mnb setup, TTS API, tool registry, GitHub integration, health checks, and autonomous workflows. See [docs/usage.md](docs/usage.md) for full usage instructions.

## Current Limitations

Integration readiness varies. For a complete, up-to-date classification of every integration
(REAL / PARTIAL / STUB / NOT STARTED) with evidence notes, see
[docs/claude/INTEGRATIONS_STATUS.md](docs/claude/INTEGRATIONS_STATUS.md).

Summary of integrations that require credentials or have known gaps:

- **Email** — PARTIAL: real IMAP/SMTP backend; falls back to stub without credentials.
- **Calendar** — PARTIAL: ICS read-only; no write support; CalDAV/Google OAuth not implemented.
- **SMS / Messaging** — PARTIAL: real Twilio backend; falls back to stub without credentials.
- **Notifications** — REAL: priority routing, digest, SQLite dashboard, SSE push all active.
- **Voice Identity** — PARTIAL: enrollment scaffolding present; not universally production-ready.
- **Autonomous Workflows** — STUB: scaffolding only; roadmap item.
- **WordPress / WooCommerce** — PARTIAL: read-only REST API access; write actions deferred.

## OpenClaw Integration

Rex can route LLM calls through the [OpenClaw](https://github.com/openclaw) gateway over HTTP, gaining access to any model provider OpenClaw supports (Ollama, OpenAI, Anthropic, etc.). Rex also exposes its tools (time, weather, email, SMS, calendar, HA, Plex, WooCommerce) as HTTP endpoints so any OpenClaw channel (WhatsApp, Telegram, Discord) can invoke them.

**Current status:** Phase 8 (HTTP integration) is complete. All integration is HTTP-based with proper error handling, retries, and auth. Feature flags in `config/rex_config.json` under `openclaw` control which code paths use the gateway. See [docs/openclaw-agent-setup.md](docs/openclaw-agent-setup.md) for setup instructions.

## Docker

Build and run Rex in a container. See [docs/docker.md](docs/docker.md) for full Docker instructions.

## Memory & Personalization

Each user has a dedicated profile in `Memory/<username>/` with preferences, history, and voice settings. See [docs/memory.md](docs/memory.md) for the full memory system documentation.

## Development

For development workflows including pip extras, GPU setup, code quality tools, and running tests, see [docs/advanced-install.md](docs/advanced-install.md). The full voice stack, including Windows GPU + TTS, is currently supported on Python 3.11 only.

### Available Test Markers

- `unit` — Fast unit tests
- `integration` — Tests requiring external services
- `slow` — Tests that take significant time
- `audio` — Tests requiring audio hardware
- `gpu` — Tests requiring GPU acceleration
- `network` — Tests requiring network access

## Troubleshooting

For help with common errors (missing API keys, FFmpeg, PyTorch, audio devices, CUDA), see [docs/troubleshooting.md](docs/troubleshooting.md).

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run linting and tests: `ruff check . && black . && pytest`
5. Commit with conventional commits: `git commit -m "feat: add amazing feature"`
6. Push to your fork: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

Released under the **MIT License**.

Copyright © 2025 James Ramsey

See [LICENSE](LICENSE) for full details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Coqui TTS](https://github.com/coqui-ai/TTS) for text-to-speech
- [openWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for LLM support
- [Flask](https://flask.palletsprojects.com/) for API framework
- [Release Please](https://github.com/googleapis/release-please) for automated releases

---

**Need help?** Check the [Troubleshooting](#troubleshooting) section or file an issue at https://github.com/Blueibear/AskRex-Assistant/issues
