# AskRex Assistant

<p align="center">
  <img src="https://github.com/Blueibear/askrex-assistant/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://www.buymeacoffee.com/Blueibear" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 28px !important;width: 120px !important;" ></a>
</p>

AskRex Assistant is a local-first, voice-activated AI companion that runs entirely on your machine — no cloud subscription required. It combines wake word detection, offline speech recognition via OpenAI Whisper, LLM-powered responses through Transformers, OpenAI, or Ollama, and text-to-speech synthesis, making it a practical choice for hobbyists, home-automation enthusiasts, and developers who want a private, customisable assistant.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Configuration](#configuration-environment-variables)
- [Usage](#usage)
- [Current Limitations](#current-limitations)
- [OpenClaw Migration](#openclaw-migration)
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

2. **Run the installer for your platform:**

   macOS / Linux:
   ```bash
   bash install.sh
   ```

   Windows (PowerShell):
   ```powershell
   .\install.ps1
   ```

3. **Configure LM Studio** — download and open [LM Studio](https://lmstudio.ai), load a model, and start the local server on `http://localhost:1234`. Then set your model in `config/rex_config.json`:
   ```json
   { "openai": { "base_url": "http://localhost:1234/v1", "model": "your-model-name" } }
   ```

4. **Run Rex:**
   ```bash
   rex
   ```

5. **Verify it works** — Rex prints `Rex assistant ready` and responds to your first message. Run the health check to confirm all components are operational:
   ```bash
   python scripts/doctor.py
   ```

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
| **Python** | 3.9 through 3.13 (3.10+ recommended) |
| **FFmpeg** | Must be installed and available on PATH |
| **Hardware** | Microphone and speakers for voice mode |
| **GPU** (optional) | NVIDIA GPU with CUDA 11.8+ for acceleration |

> **Note for Windows users**: The `simpleaudio` package (used for audio playback) has build issues on Windows and is automatically disabled. Audio playback functionality will be limited on Windows, but all core features work correctly.

## Configuration (Environment Variables)

Rex uses a dual-config system: secrets in `.env`, runtime settings in `config/rex_config.json`. See [docs/environment-variables.md](docs/environment-variables.md) for a full reference of all supported variables.

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

## OpenClaw Migration

Rex is undergoing a phased migration to run as an opinionated application layer on top of [OpenClaw](https://github.com/openclaw), an agent engine that provides channels, sessions, browser control, dashboard UI, skills/plugins, and multi-agent orchestration.

The goal is to stop rebuilding generic agent infrastructure inside Rex and instead focus on what makes Rex unique: persona, voice identity, wakeword/voice loop, Home Assistant integration, WordPress/WooCommerce/Plex integrations, business workflows, and policy/approval logic.

**Current status:** Phase 7 (retirement) is in progress. Two redundant modules (`plugin_loader.py`, `executor.py`) have been retired so far. Remaining retirements are blocked pending OpenClaw installation. See [PRD-openclaw-pivot-for-rex.md](PRD-openclaw-pivot-for-rex.md) for the full migration plan and [progress-openclaw-pivot.txt](progress-openclaw-pivot.txt) for iteration history.

**Migration adapters** live in `rex/openclaw/` and include bridges for tools, events, browser automation, workflows, voice, and identity. Feature flags in `config/rex_config.json` under the `openclaw` key control which code paths use the new OpenClaw backend.

## Docker

Build and run Rex in a container. See [docs/docker.md](docs/docker.md) for full Docker instructions.

## Memory & Personalization

Each user has a dedicated profile in `Memory/<username>/` with preferences, history, and voice settings. See [docs/memory.md](docs/memory.md) for the full memory system documentation.

## Development

For development workflows including pip extras, GPU setup, code quality tools, and running tests, see [docs/advanced-install.md](docs/advanced-install.md).

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
