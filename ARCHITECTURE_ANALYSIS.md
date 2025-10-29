# Architecture Analysis & Action Plan

**Date:** 2025-10-29
**Branch:** `claude/session-011CUYfrKv9big1HN5RekfuX`
**Status:** Analysis Complete

---

## Executive Summary

Analysis of the Rex AI Assistant codebase identified several architectural concerns categorized by priority. This document provides current state assessment, recommended actions, and implementation status.

---

## P0 Issues (Block Install/Run) - Assessment

### 1. Broken Dockerfile ❌ NOT FOUND
**Claim:** Incomplete RUN command with literal `...` and truncated pip install

**Analysis:**
- Current Dockerfile (lines 1-44) is well-formed and complete
- All RUN commands are properly terminated
- Uses proper CPU-only PyTorch installation
- No literal `...` or truncated commands found

**Status:** ✅ **NO ACTION REQUIRED** - Dockerfile is correct

### 2. Contradictory CPU Requirements ❌ NOT FOUND
**Claim:** requirements-cpu.txt has `torch==2.7.1+cu118` (CUDA tags in CPU file)

**Analysis:**
- `requirements-cpu.txt` correctly specifies `torch==2.6.0` (no CUDA suffix)
- `requirements-gpu.txt` properly has `torch==2.6.0+cu118` with CUDA tags
- Files are correctly separated for CPU vs GPU

**Status:** ✅ **NO ACTION REQUIRED** - Requirements files are correct

### 3. Package Layout Duplication ⚠️ CONFIRMED
**Claim:** Two module trees (root `*.py` + `rex/` package) cause import shadowing

**Analysis:**
```
Root level: 23 Python modules (config.py, llm_client.py, memory_utils.py, etc.)
rex/ package: 9 Python files, some are shim re-exports
```

**Examples of shims:**
```python
# rex/llm_client.py
from llm_client import *  # Re-exports root module

# rex/memory.py
from memory_utils import *  # Re-exports root module
```

**Risks:**
- Import shadowing when installed as package
- Non-hermetic installs (depends on PYTHONPATH)
- Confusion about canonical import path
- `setup.py` had to add `py_modules` workaround

**Recommended Fix (P2 - future refactoring):**
1. Move all logic to `rex/` package (canonical source)
2. Delete root-level duplicates or make them thin CLI entry points only
3. Use proper relative imports within `rex/`
4. Update all imports in tests and examples

**Status:** 📋 **DOCUMENTED** - Non-blocking, recommend future refactoring

### 4. Config Key Drift ✅ MINIMAL
**Claim:** Inconsistent names/aliases (e.g., `max_memory_items` truncated reference)

**Analysis:**
Checked consistency across config systems:

**rex/config.py (pydantic-settings):**
```python
max_memory_items: int = Field(..., validation_alias="REX_MEMORY_MAX_ITEMS")
```

**config.py (dataclass):**
```python
memory_max_turns: int = 50  # Different name!
```

**rex/assistant.py:**
```python
self._history_limit = history_limit or self._settings.max_memory_items  # Uses pydantic version
```

**Issue:** Two config systems with different field names:
- `rex.config.Settings.max_memory_items` (used by rex.assistant)
- `config.AppConfig.memory_max_turns` (used by root scripts)

**Status:** ⚠️ **DOCUMENTED** - Minor drift between two config systems (see recommendation)

### 5. Missing Installation Steps ✅ COVERED
**Claim:** Docker references elided Torch versions; CI passes partially while real envs fail

**Analysis:**
- Dockerfile explicitly pins: `torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0`
- CI uses `requirements-cpu.txt` with same versions
- No version elision found

**Status:** ✅ **NO ACTION REQUIRED** - Versions are explicit

---

## P1 Issues (High Priority) - Actionable

### 1. Import Style ⚠️ CONFIRMED
**Issue:** `rex/memory.py` re-exports via absolute import from `memory_utils`

**Example:**
```python
# rex/memory.py
from memory_utils import *  # Absolute import to root module
```

**Recommended Fix:**
Move `memory_utils.py` logic into `rex/memory.py` directly, or use `from ..memory_utils import *` if keeping root modules temporarily.

**Status:** 📋 **Documented** for future refactoring

### 2. Dependency Policy ✅ ACTION ITEM
**Issue:** All dependencies in single list; heavy deps (TTS, Whisper, etc.) installed even for minimal use

**Recommended Fix:**
Add optional extras to `pyproject.toml`:
```toml
[project.optional-dependencies]
stt = ["openai-whisper>=20230124"]
tts = ["TTS>=0.18.0"]
wakeword = ["openwakeword>=0.6.0"]
search = ["beautifulsoup4>=4.12.0"]
audio = ["sounddevice>=0.4.6", "soundfile>=0.12.0", "simpleaudio>=1.0.4"]
full = ["rex-ai-assistant[stt,tts,wakeword,search,audio]"]
```

**Benefits:**
- Faster installs for specific use cases
- Smaller Docker images
- Better developer experience

**Status:** 🔨 **IMPLEMENTING**

### 3. Config Surface ✅ ACTION ITEM
**Issue:** `REQUIRED_ENV_KEYS = {"REX_WAKEWORD"}` or `("OPENAI_API_KEY",)` contradicts "local-first"

**Current State:**
```python
# rex/config.py
REQUIRED_ENV_KEYS = {"REX_WAKEWORD"}  # Not enforced

# config.py
REQUIRED_ENV_KEYS = {"REX_WAKEWORD"}  # Not enforced
```

**Recommended Fix:**
Make OpenAI key required only when `llm_backend="openai"`:
```python
def validate_config(settings: Settings) -> list[str]:
    errors = []
    if settings.llm_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY required when llm_backend=openai")
    # ... other conditional validations
    return errors
```

**Status:** 🔨 **IMPLEMENTING**

### 4. CI Gaps 📋 DOCUMENTED
**Recommendations:**
- Add Docker build & smoke test to CI
- Test Python 3.10, 3.11, 3.12 matrix
- Add no-audio E2E test with mocks

**Status:** 📋 **Documented** for future CI improvements

### 5. Security Defaults ✅ ACTION ITEM
**Current State:**
- Rate limits configured but not strict defaults
- API key auth optional
- CORS allowlist implemented (✅)

**Recommended Improvements:**
- Enable rate limiting by default in production
- Require API key for rex-speak-api in production
- Add request timeout enforcement
- Document security best practices

**Status:** 🔨 **IMPLEMENTING**

### 6. Tests - E2E Mocked Pipeline 📋 FUTURE
**Recommendation:** Add end-to-end test:
```python
async def test_full_pipeline_mocked():
    # WAV input → STT stub → LLM stub → TTS no-op
    # Assert transcript persisted, timing reasonable
```

**Status:** 📋 **Documented** for future test improvements

---

## P2 Issues (Medium Priority) - Documented

### Structured Logging
- Add JSON logging option via `REX_LOG_FORMAT=json`
- Include context IDs per interaction
- Add `--debug-audio` flag to dump WAVs

### Plugin API
- Formalize `PluginSpec` with typed interface
- Document capability flags (network, disk, timeout)
- Example plugin with best practices

### Memory Schema
- Define JSON schema for `Memory/<user>/core.json`
- Add size caps and rotation
- Consider SQLite for conversation history with timestamp index

### CLI Polish
- `rex config show/set/validate`
- Safe .env editing helpers

### Docs Consolidation
- Merge README/INSTALL into quickstart
- Ensure CPU-only path works out of the box

---

## Concrete Actions - Implementation Plan

### Phase 1: Dependency & Config Improvements (CURRENT)
1. ✅ Add optional extras to pyproject.toml (stt, tts, wakeword, search, audio, full)
2. ✅ Add conditional config validation (backend-specific requirements)
3. ✅ Add startup validation function that prints config table
4. ✅ Document security best practices in SECURITY_DEFAULTS.md

### Phase 2: Package Refactoring (FUTURE - Breaking Changes)
1. Consolidate to single canonical package structure
2. Move root *.py modules into rex/ or remove
3. Update all imports to use `from rex import ...`
4. Remove shim re-exports

### Phase 3: Testing & CI Improvements (FUTURE)
1. Add E2E mocked pipeline test
2. Add Docker build to CI
3. Add Python version matrix
4. Add smoke tests

---

## Appendix: File Structure

### Current Layout
```
rex-ai-assistant/
├── *.py (23 root modules)        # Entry points + legacy
├── rex/                           # Core package
│   ├── __init__.py
│   ├── assistant.py              # Main orchestrator
│   ├── config.py                 # Pydantic settings (preferred)
│   ├── voice_loop.py
│   ├── llm_client.py             # Shim → root llm_client.py
│   ├── memory.py                 # Shim → root memory_utils.py
│   └── plugins/
├── config.py                      # Dataclass config (legacy)
├── llm_client.py                  # Root implementation
├── memory_utils.py                # Root implementation
└── tests/
```

### Recommended Future Layout
```
rex-ai-assistant/
├── rex/                           # Single canonical package
│   ├── __init__.py
│   ├── assistant.py
│   ├── config.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py             # Moved from root
│   ├── memory/
│   │   ├── __init__.py
│   │   └── utils.py              # Moved from root
│   ├── voice_loop.py
│   └── plugins/
├── rex_assistant.py               # Thin CLI wrapper only
├── rex_speak_api.py               # Thin API entry only
└── tests/
```

---

## Summary

**P0 Issues:** 1/5 confirmed (package layout), 4/5 not found in current code
**P1 Issues:** 3/6 actionable now, 3/6 documented for future
**Recommendation:** Implement Phase 1 improvements (optional extras, config validation, startup checks) without breaking current functionality. Defer package restructuring to major version bump.
