# Rex AI Assistant

<p align="center">
  <img src="https://github.com/Blueibear/askrex-assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://www.buymeacoffee.com/Blueibear" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 28px !important;width: 120px !important;" ></a>
</p>

Rex AI Assistant is a local-first, voice-activated AI companion that runs entirely on your machine — no cloud subscription required. It combines wake word detection, offline speech recognition via OpenAI Whisper, LLM-powered responses through Transformers, OpenAI, or Ollama, and text-to-speech synthesis, making it a practical choice for hobbyists, home-automation enthusiasts, and developers who want a private, customisable assistant.

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

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Blueibear/askrex-assistant.git
   cd askrex-assistant
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

   macOS / Linux full install:
   ```bash
   bash install.sh
   ```

   Windows full install:
   ```powershell
   .\install.ps1
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

> Python support is intentionally strict: the current dependency stack is validated on Python 3.11. Fresh installs on Python 3.13/3.14 are known to fail in the ML/TTS path, so unsupported versions now fail fast.

> **Advanced / Developer Install** — for GPU setups, custom extras, Docker, or development workflows, see [docs/advanced-install.md](docs/advanced-install.md).
>
> **Want one consolidated guide?** See [docs/INSTRUCTION_MANUAL.md](docs/INSTRUCTION_MANUAL.md) for a single manual that combines install, configuration, usage, operations, and troubleshooting.

## Features

- 🔊 **Wake word detection** via openWakeWord (customizable trigger phrases)
- 🗣️ **Speech-to-text** using OpenAI Whisper (runs offline)
- 🤖 **LLM responses** via Transformers (local), OpenAI API, or Ollama
- 🔉 **Text-to-speech** with Coqui XTTS, edge-tts, or pyttsx3 (voice cloning supported)
- 🌐 **Web search plugins** for SerpAPI, Brave, Google CSE, and DuckDuckGo
- 🧠 **Per-user memory** profiles with conversation history and preferences
- 📧 **Email and calendar** integration with triage and scheduling *(beta — stub/mock data only)*
- 📱 **Multi-channel messaging** via SMS *(beta — stub scaffolding, real delivery requires Twilio credentials)*
- 🔔 **Smart notifications** with priority routing, digest mode, quiet hours, and auto-escalation *(beta — stub scaffolding)*
- 🤖 **Autonomous workflows** with planner and workflow runner for multi-step task automation
- 🎯 **Smart planning** converts natural language goals into structured workflows
- ⚙️ **Configurable autonomy modes** (OFF/SUGGEST/AUTO) for fine-grained control
- 🔐 **Flask TTS API** with authentication and rate limiting
- ✅ **CI/CD** with GitHub Actions and Release Please automation
- 🐳 **Docker support** for containerized deployment

## Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | macOS 11+, Windows 10/11, or Ubuntu 20.04+ |
| **Python** | 3.11 |
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

The following integrations are **beta / stub scaffolding** and do not yet connect to live services:

| Feature | Status | Details |
|---------|--------|---------|
| **Email** | Beta (real backend available) | Supports real IMAP4-SSL read + SMTP send when configured; defaults to stub/mock for offline dev. Multi-account support included. |
| **Calendar** | Beta (ICS read-only + stub fallback) | Supports reading events from local `.ics` files or HTTPS ICS feeds; defaults to stub/mock for offline dev. CalDAV/Google OAuth planned. |
| **SMS / Messaging** | Beta (real backend available) | Real SMS delivery and inbound webhook receiver via Twilio when configured (opt-in); defaults to stub/mock for offline dev. Multi-account support and inbound message routing included. |
| **Notifications** | Beta (dashboard + email channels real) | Priority routing and digest logic exist; dashboard channel persists to local SQLite store with API endpoints; email channel uses real SMTP when configured. |
| **Identity** | Beta (session-scoped fallback) | When voice/speaker recognition is unavailable, use `rex identify` or `rex whoami` to set/view the active user for the session. |
| **WordPress** | Beta (read-only) | Health check (`rex wp health`) via WP REST API. Supports `none`, `application_password`, and `basic` auth methods. Write actions deferred to Cycle 6.3. |
| **WooCommerce** | Beta (read-only) | Orders list and products list (`rex wc orders list`, `rex wc products list`) via WC REST API v3. Client-side low-stock filter supported. Write actions deferred to Cycle 6.3. |

All stub commands are fully usable for development and testing. Contributions to add real backends are welcome.

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

**Need help?** Check the [Troubleshooting](#troubleshooting) section or file an issue at https://github.com/Blueibear/askrex-assistant/issues
