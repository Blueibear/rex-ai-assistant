# Rex AI Assistant - Architecture Documentation

## Overview

Rex is a voice-activated AI assistant that combines wake word detection, speech recognition, LLM processing, and text-to-speech synthesis. This document describes the architecture, design decisions, and code organization.

## Directory Structure

```
rex-ai-assistant/
├── rex/                    # Canonical package implementation
│   ├── __init__.py        # Package exports
│   ├── assistant.py       # Core Assistant class
│   ├── config.py          # Configuration management
│   ├── llm_client.py      # LLM integration
│   ├── memory.py          # Working & long-term memory (see docs/memory.md)
│   ├── memory_utils.py    # User memory & profiles
│   ├── knowledge_base.py  # Document storage & search (see docs/knowledge_base.md)
│   ├── plugin_loader.py   # Plugin system
│   ├── voice_loop.py      # Voice loop (wrapper to optimized)
│   ├── voice_loop_optimized.py  # CANONICAL voice loop implementation
│   ├── assistant_errors.py # Exception hierarchy
│   ├── logging_utils.py   # Logging configuration
│   └── wakeword/          # Wake word detection
│       ├── listener.py
│       └── utils.py
├── plugins/               # Pluggable extensions
│   ├── __init__.py       # Plugin package marker
│   └── web_search.py     # Web search plugin
├── utils/                # Utilities
│   └── env_loader.py     # Environment variable loader
├── scripts/              # Utility scripts
│   └── doctor.py         # Health check script
├── tests/                # Test suite
├── Memory/               # User profiles & conversation history
├── *.py (root level)     # Compatibility wrappers & entry points
└── docs/                 # Documentation
    └── ARCHITECTURE.md   # This file
```

## Design Principles

### 1. Canonical Source: `rex/` Package

**All core functionality lives in the `rex/` package.** This is the single source of truth for:
- Configuration (`rex/config.py`)
- Error handling (`rex/assistant_errors.py`)
- LLM integration (`rex/llm_client.py`)
- Memory management (`rex/memory_utils.py`)
- Plugin system (`rex/plugin_loader.py`)
- Voice loop (`rex/voice_loop_optimized.py`)

### 2. Root-Level Compatibility Wrappers

Root-level Python files serve as **backward compatibility wrappers** for legacy code:

```python
# Example: config.py (root level)
"""Wrapper for backward compatibility."""
from rex.config import *  # noqa: F401, F403
```

**Why this structure?**
- **Gradual migration**: Existing code can continue importing from root
- **Clean package**: New code imports from `rex.*`
- **Minimal duplication**: Wrappers are tiny (1-3 lines)

**Which files are wrappers?**
- `config.py`
- `assistant_errors.py`
- `llm_client.py`
- `logging_utils.py`
- `memory_utils.py`
- `plugin_loader.py`

**Which root files are standalone?**
- `rex_assistant.py` - Main CLI entry point
- `rex_loop.py` - Voice loop runner
- `rex_speak_api.py` - Flask TTS API server
- `flask_proxy.py` - User authentication proxy
- `audio_config.py` - Audio device configuration
- `install.py` - Interactive installer

## Core Components

### Configuration (`rex/config.py`)

**Responsibility:** Environment variable loading, validation, and configuration management.

```python
from rex.config import AppConfig, load_config

config = load_config()  # Loads from environment
```

**Key features:**
- Pydantic-based validation
- Type-safe configuration
- Environment variable mapping
- Singleton pattern for global config

### Voice Loop (`rex/voice_loop_optimized.py`)

**IMPORTANT:** `rex/voice_loop.py` is a **compatibility wrapper** that re-exports from `voice_loop_optimized.py`.

**Canonical implementation:** `rex/voice_loop_optimized.py`

**Architecture:**
1. **AsyncMicrophone** - Audio input with VAD
2. **WakeAcknowledgement** - Wake word feedback sound
3. **SpeechToText** - Whisper integration
4. **TextToSpeech** - Multi-provider TTS (XTTS, Edge, Piper, Windows)
5. **VoiceLoop** - Main orchestrator

**Why "optimized"?**
- Voice Activity Detection (VAD) for faster recording
- Automatic Whisper model downgrade (base → tiny for 3x speedup)
- Better default TTS provider (Edge-TTS instead of slow XTTS)
- Concurrent STT + LLM processing where possible

### Plugin System (`rex/plugin_loader.py`)

**Architecture:**
```
plugins/
├── __init__.py           # Package marker
├── web_search.py         # Example plugin
└── [future_plugin].py

Each plugin:
1. Defines register() function
2. Returns Plugin(name, description, execute)
3. Loaded dynamically by plugin_loader
```

**Plugin Protocol:**
```python
class Plugin(Protocol):
    name: str
    description: str

    def execute(self, context: dict, **kwargs) -> Any:
        ...
```

**Loading:**
```python
from rex.plugin_loader import load_plugins

plugins = load_plugins()
result = plugins['web_search'].execute({}, query="Python")
```

### Memory System (`rex/memory_utils.py`)

**Structure:**
```
Memory/
├── users.json            # Email → user_key mapping
└── <user_key>/
    ├── core.json        # Profile (name, preferences, voice)
    └── history.json     # Conversation history
```

**Security:**
- Path traversal protection via `_sanitize_user_key()`
- Validation with `_validate_path_within()`
- User isolation

## Entry Points

### 1. CLI Assistant (`python -m rex`)

**Flow:**
```
python -m rex
  ↓
rex/__main__.py
  ↓
rex_assistant.py:main()
  ↓
rex.assistant.Assistant
```

**Mode:** Text-based chat (no audio)

### 2. Voice Loop (`python rex_loop.py`)

**Flow:**
```
python rex_loop.py
  ↓
build_voice_loop()
  ↓
rex.voice_loop_optimized.VoiceLoop
  ↓
Continuous wake word → STT → LLM → TTS loop
```

**Mode:** Full voice interaction with wake word

### 3. TTS API (`python -m rex-speak-api`)

**Flow:**
```
python rex_speak_api.py
  ↓
Flask app on port 5000
  ↓
POST /speak with API key
  ↓
TTS generation → audio file
```

**Mode:** HTTP API for text-to-speech

## Design Decisions

### Why Both `voice_loop.py` and `voice_loop_optimized.py`?

**History:**
- Original implementation: `voice_loop.py` (incomplete)
- Optimized implementation: `voice_loop_optimized.py` (complete, faster)
- Solution: Make `voice_loop.py` a wrapper to `voice_loop_optimized.py`

**Current state:**
```python
# rex/voice_loop.py
from .voice_loop_optimized import (
    VoiceLoop,
    build_voice_loop,
    # ... all exports
)
```

**Future:** Could merge into single `voice_loop.py` after deprecation period.

### Why Separate `utils/env_loader.py`?

**Problem:** Direct `python-dotenv` usage violates import order (PEP 8).

**Solution:** Bootstrap module that auto-loads on first import.

```python
# Any module can simply:
import utils.env_loader  # Auto-loads .env

# Instead of:
from utils.env_loader import load as _load_env
_load_env()  # Manual call
```

**Benefits:**
- PEP 8 compliant imports
- Single responsibility
- No manual initialization needed

### Why `rex/` Package vs Root Files?

**Evolution:**
1. **Phase 1** (legacy): All code in root directory
2. **Phase 2** (current): Core in `rex/`, wrappers in root
3. **Phase 3** (future): Deprecate root wrappers, `rex/` only

**Migration strategy:**
```python
# Old code (still works)
from config import load_config

# New code (recommended)
from rex.config import load_config
```

**Timeline:**
- ✅ Phase 2 complete (current state)
- 📅 Phase 3 planned (post v1.0)

## Plugin Development Guide

### Creating a New Plugin

1. **Create plugin file:**
```python
# plugins/my_plugin.py
from rex.plugins import Plugin

def register() -> Plugin:
    """Register plugin with the system."""
    return Plugin(
        name="my_plugin",
        description="Does something useful",
        execute=my_execute_function,
    )

def my_execute_function(context: dict, **kwargs):
    """Plugin logic here."""
    return {"result": "success"}
```

2. **Plugin is auto-discovered** by `rex/plugin_loader.py`

3. **Use in assistant:**
```python
plugins = load_plugins()
result = plugins['my_plugin'].execute({}, arg1="value")
```

### Plugin Best Practices

- ✅ Handle missing dependencies gracefully
- ✅ Validate inputs
- ✅ Return structured data (dict)
- ✅ Log errors, don't crash
- ✅ Document in docstrings
- ❌ Don't modify global state
- ❌ Don't import at module level if dependency might be missing

## Testing Architecture

### Test Organization

```
tests/
├── test_config.py          # Configuration tests
├── test_llm_client.py      # LLM integration tests
├── test_memory_utils.py    # Memory system tests
├── test_plugin_loader.py   # Plugin loading tests
├── test_voice_loop.py      # Voice loop tests
├── test_rex_speak_api.py   # API tests
└── ...
```

### Test Markers

Defined in `pyproject.toml`:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Tests with external services
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.audio` - Requires audio hardware
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.network` - Requires network access

**Run specific tests:**
```bash
pytest -m unit                    # Fast unit tests only
pytest -m "not slow and not audio"  # Skip slow/audio tests
```

## Security Architecture

### API Authentication (`rex_speak_api.py`)

**Layers:**
1. **API Key Validation** - HMAC constant-time comparison
2. **Rate Limiting** - Per-user or per-IP limits
3. **Trusted Proxy Handling** - Only trust X-Forwarded-For from configured proxies

**Environment variables:**
```bash
REX_SPEAK_API_KEY=your-secret-key
REX_TRUSTED_PROXIES=127.0.0.1,::1  # Comma-separated IPs
REX_SPEAK_RATE_LIMIT=30            # Requests per window
REX_SPEAK_RATE_WINDOW=60           # Window in seconds
```

### Memory Isolation

**Path traversal protection:**
```python
# Prevents: ../../etc/passwd
user_key = _sanitize_user_key(raw_input)
path = _validate_path_within(user_folder, file)
```

### Dependency Security

**Tracked in:**
- `SECURITY_ADVISORY.md` - Known vulnerabilities
- `requirements.txt` - Minimum safe versions
- `pyproject.toml` - Security-critical dependencies

**Process:**
1. Dependabot alerts
2. Review CVE severity
3. Update minimum versions
4. Document in SECURITY_ADVISORY.md

## Deployment

### Docker

**Multi-stage build:**
1. **Stage 1 (deps):** Install dependencies
2. **Stage 2 (runtime):** Copy files, create non-root user

**Security features:**
- Non-root user (`rex:rex`)
- Minimal base image (slim)
- No unnecessary packages
- Health checks

### Environment Configuration

**Required:**
- `REX_SPEAK_API_KEY` (for API mode)

**Optional:**
- `REX_ACTIVE_USER` - Default user profile
- `REX_WHISPER_MODEL` - Whisper model size
- `REX_TTS_PROVIDER` - TTS backend
- `REX_LOG_LEVEL` - Logging verbosity

See `.env.example` for full list.

## Migration Guide

### From Root Imports to `rex/` Package

**Step 1:** Update imports
```python
# Before
from config import load_config
from assistant_errors import ConfigurationError

# After
from rex.config import load_config
from rex.assistant_errors import ConfigurationError
```

**Step 2:** No other changes needed! The API is identical.

### From `voice_loop.py` to `voice_loop_optimized.py`

**No changes needed!** `voice_loop.py` is already a wrapper:

```python
from rex.voice_loop import VoiceLoop  # Gets optimized version
```

## Future Roadmap

### Short-term
- ✅ Complete test coverage (>80%)
- ✅ Enforce type checking in CI
- 📅 Add more plugins (calendar, email, etc.)
- 📅 Improve wake word accuracy

### Long-term
- 📅 Deprecate root-level wrappers
- 📅 Switch to pure `rex/` package imports
- 📅 Plugin marketplace/discovery
- 📅 Multi-language support
- 📅 Streaming TTS for lower latency

## References

- **Main README:** `../README.md`
- **Security Advisory:** `../SECURITY_ADVISORY.md`
- **Environment Config:** `../.env.example`
- **Plugin Example:** `../plugins/web_search.py`
- **Test Examples:** `../tests/`

## Contributing

When adding new features:
1. ✅ Put code in `rex/` package (not root)
2. ✅ Add comprehensive tests
3. ✅ Update this documentation
4. ✅ Use type hints
5. ✅ Follow PEP 8
6. ✅ Add docstrings

**Questions?** Open an issue on GitHub.
