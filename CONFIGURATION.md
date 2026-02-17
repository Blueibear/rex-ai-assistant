# Rex AI Assistant Configuration Guide

Rex AI Assistant uses a dual-configuration system that separates secrets from runtime settings for better security and ease of use.

## Dependency notes
Core runtime dependencies are covered by the main requirements files. Optional features such as the web search plugin or the Home Assistant bridge rely on extra packages like requests, BeautifulSoup, and Flask. If those extras are not installed, Rex can still import and run core features, but using the optional feature will raise a clear error.

## Configuration Files

### config/rex_config.json (Runtime Settings)
Non-secret configuration including audio devices, model settings, wake word configuration, etc.

**Location**: `config/rex_config.json`

**Format**: JSON

**Example**:
```json
{
  "active_profile": "default",
  "profiles_dir": "profiles",
  "audio": {
    "input_device_index": 0,
    "output_device_index": null,
    "sample_rate": 16000
  },
  "wake_word": {
    "wakeword": "rex",
    "threshold": 0.5
  },
  "models": {
    "llm_provider": "transformers",
    "llm_model": "sshleifer/tiny-gpt2"
  }
}
```

**Editing**:
- Primary method: Use the Settings tab in the GUI
- Alternative: Edit `config/rex_config.json` directly with a text editor
- CLI: Use `python -m audio_config` for audio device selection

### profiles directory (Profile Settings)
Profiles define higher level behavior, including enabled capabilities and overrides to runtime settings.

**Location**: `profiles/`

**Active profile selector**: `config/rex_config.json` uses `active_profile` and defaults to `default`.

**Profile format**:
```json
{
  "profile_version": 1,
  "name": "default",
  "description": "Paid stable profile",
  "capabilities": ["local_commands", "ha_router"],
  "overrides": {
    "runtime": {
      "log_level": "INFO"
    },
    "models": {
      "tts_provider": "xtts"
    }
  }
}
```

**Profiles shipped with the repo**:
- `profiles/default.json` is the paid stable profile
- `profiles/james.json` is a personal profile with expanded household capabilities

**Important**:
- Secrets stay in `.env` only
- Profiles should not contain API keys or tokens

### .env (Secrets Only)
API keys, tokens, and other sensitive information.

**Location**: `.env` (root directory)

**Format**: Key-value pairs

**Example**:
```bash
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=...
HA_TOKEN=...
```

**Editing**:
- GUI: Settings tab, Secrets section
- Alternative: Edit `.env` directly with a text editor
- IMPORTANT: Never commit .env to version control

## Configuration Structure

### Audio Settings (rex_config.json)
```json
{
  "audio": {
    "input_device_index": null,    // Audio input device (null = system default)
    "output_device_index": null,   // Audio output device (null = system default)
    "sample_rate": 16000           // Sample rate in Hz (16000 recommended)
  }
}
```

**How to configure**:
1. Open GUI
2. Go to Dashboard tab
3. Select device from "Input Device" dropdown
4. Device is automatically saved to `config/rex_config.json`

**CLI method**:
```bash
# List available devices
python -m audio_config --list

# Set input device
python -m audio_config --set-input 0

# Show current configuration
python -m audio_config --show
```

### Wake Word Settings (rex_config.json)
```json
{
  "wake_word": {
    "backend": "openwakeword",                  // Backend: openwakeword, custom_onnx, custom_embedding
    "wakeword": "rex",                          // Legacy wake phrase (keyword is preferred)
    "keyword": null,                            // Preferred wake word keyword
    "model_path": null,                         // Path to custom ONNX model
    "embedding_path": null,                     // Path to custom embedding .pt
    "fallback_to_builtin": true,                // Fall back to built-in keyword when custom fails
    "fallback_keyword": "hey jarvis",           // Preferred built-in fallback
    "threshold": 0.5,                           // Detection threshold (0.0-1.0)
    "window": 1.0,                              // Detection window (seconds)
    "poll_interval": 0.01,                      // Poll interval (seconds)
    "wake_sound_path": "assets/wake_acknowledgment.wav"
  }
}
```

#### Built-in OpenWakeWord keywords and fallback behavior
When backend is openwakeword, Rex validates the requested keyword against the installed OpenWakeWord models.
If the keyword is missing or invalid, Rex automatically falls back to a working keyword, preferring "hey jarvis" if it is available, otherwise the first available model.

#### Custom wake words
Store custom models in `models/wakewords/` and point the configuration at the file path. The GUI Wake Word panel lets you select built-in keywords and any files in that folder.

**Custom ONNX model (recommended)**
```json
{
  "wake_word": {
    "backend": "custom_onnx",
    "model_path": "models/wakewords/hey_rex.onnx",
    "fallback_to_builtin": true,
    "fallback_keyword": "hey jarvis"
  }
}
```

**Custom embedding model**
```json
{
  "wake_word": {
    "backend": "custom_embedding",
    "embedding_path": "models/wakewords/hey_rex.pt",
    "fallback_to_builtin": true,
    "fallback_keyword": "hey jarvis"
  }
}
```

If fallback_to_builtin is false and the custom file is missing or invalid, Rex raises a clear error on startup.

#### Training workflow (OpenWakeWord)
1. Train or export your wake word with OpenWakeWord tooling to produce an ONNX model.
2. Place the ONNX file in `models/wakewords/`.
3. Set `wake_word.backend` to `custom_onnx` and update `wake_word.model_path`.
4. Validate the file before starting Rex:
   ```bash
   python scripts/validate_wakeword_model.py --backend custom_onnx --model-path models/wakewords/hey_rex.onnx
   ```

#### Custom embedding workflow (optional)
If you have a .pt embedding file, set `wake_word.backend` to `custom_embedding` and point `embedding_path` at the file. Rex uses cosine similarity on a lightweight embedding of the audio frame.
Use the same validation script:
```bash
python scripts/validate_wakeword_model.py --backend custom_embedding --embedding-path models/wakewords/hey_rex.pt
```

#### Legacy wakeword alias
`wake_word.wakeword` is treated as a legacy alias for `wake_word.keyword`. Rex will keep reading it for compatibility, but new configuration should prefer `keyword` only.

Legacy environment variables are ignored for wake word configuration. Always update `config/rex_config.json` instead.

### Model Settings (rex_config.json)
```json
{
  "models": {
    "llm_provider": "transformers",    // LLM: transformers, openai, ollama
    "llm_backend": null,               // Alias for llm_provider
    "llm_model": "sshleifer/tiny-gpt2",
    "llm_max_tokens": 120,
    "llm_temperature": 0.7,
    "llm_top_p": 0.9,
    "llm_top_k": 50,
    "llm_seed": 42,
    "stt_model": "base",               // Whisper model: tiny, base, small, medium, large
    "stt_device": "cpu",               // Device: cpu, cuda
    "tts_provider": "xtts",            // TTS: xtts, edge, piper, pyttsx3
    "tts_model": null,
    "tts_voice": null,
    "windows_tts_voice_index": null
  }
}
```

## Tool Requests and Results
Rex supports a minimal tool routing flow for local tools. The model can emit a single line tool request:

```
TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}
```

Rex executes the tool and sends a tool result line back to the model:

```
TOOL_RESULT: {"tool":"time_now","args":{"location":"Dallas, TX"},"result":{"local_time":"YYYY-MM-DD HH:MM","timezone":"America/Chicago"}}
```

### Supported Tools
- time_now is implemented
- weather_now is stubbed for future use
- web_search is stubbed for future use

### Extending Tools
To add new tools, update the tool router module to parse the new tool name and return a structured result. Add tests for the new tool and keep results in the TOOL_RESULT format.

### Runtime Settings (rex_config.json)
```json
{
  "runtime": {
    "log_level": "INFO",                // DEBUG, INFO, WARNING, ERROR, CRITICAL
    "file_logging_enabled": false,
    "transcripts_enabled": true,
    "transcripts_dir": "transcripts",
    "active_user": null,                // Active user profile
    "user_id": "default",
    "memory_max_turns": 50,             // Conversation history length
    "command_duration": 5.0,            // Command recording duration (seconds)
    "capture_seconds": 5.0,
    "detection_frame_seconds": 1.0,
    "conversation_export": true,
    "speak_language": "en"
  }
}
```

### Secrets (.env)
```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
OLLAMA_API_KEY=...

# Search API Keys
BRAVE_API_KEY=...
SERPAPI_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...

# Home Assistant
HA_TOKEN=...
HA_SECRET=...

# Other Services
REX_SPEAK_API_KEY=...
BROWSERLESS_API_KEY=...
REX_PROXY_TOKEN=...
FLASK_LIMITER_STORAGE_URI=...
REX_SPEAK_STORAGE_URI=...
```

## Configuration Precedence

Rex resolves settings in this order (highest priority first):

1. **Profile overrides** — the active profile's `overrides` section in `profiles/<name>.json`
2. **config/rex_config.json** — user-edited runtime settings
3. **Built-in defaults** — `DEFAULT_CONFIG` in `rex/config_manager.py`

Secrets (API keys, tokens) are always read from environment variables loaded via `.env`. They are never stored in `rex_config.json`.

Legacy non-secret environment variables (e.g. `OPENAI_BASE_URL`, `REX_LLM_PROVIDER`) are **ignored at runtime**. If any are set, Rex logs a warning and recommends running the migration command.

## Migration from Legacy Configuration

If you have an existing Rex installation with non-secret settings in `.env`, use the migration command to move them into `config/rex_config.json`.

### Migration command

```bash
rex-config migrate-legacy-env
```

Options:
- `--config-path PATH` — target config file (default: `config/rex_config.json`)
- `--dry-run` — show what would be migrated without writing changes

The migration command:
- Only migrates non-secret settings (API keys and tokens stay in `.env`)
- Never overwrites config values that already differ from the defaults
- Prints a summary of every setting it migrated

### What gets migrated:
- Audio device settings (`REX_INPUT_DEVICE`, `REX_SAMPLE_RATE`, etc.)
- Wake word settings (`REX_WAKEWORD`, `REX_WAKEWORD_THRESHOLD`, etc.)
- Model settings (`REX_LLM_PROVIDER`, `REX_WHISPER_MODEL`, etc.)
- Runtime settings (`REX_LOG_LEVEL`, `REX_MEMORY_MAX_TURNS`, etc.)
- OpenAI runtime settings (`OPENAI_BASE_URL`, `OPENAI_MODEL`)

### What stays in .env:
- API keys (`OPENAI_API_KEY`, `BRAVE_API_KEY`, etc.)
- Tokens (`HA_TOKEN`, `REX_PROXY_TOKEN`, etc.)
- Secret URLs and credentials

### Migration process:
1. Run `rex-config migrate-legacy-env` (or add `--dry-run` to preview)
2. Non-secret settings are written to `config/rex_config.json`
3. Existing non-default config values are preserved (no-overwrite rule)
4. Legacy environment variables are ignored at runtime after migration
5. Remove migrated settings from `.env` to clean up (optional)

### Manual migration:
If you prefer to migrate manually, copy settings from `.env` to `config/rex_config.json`:

Example:
```bash
# Old .env
REX_WAKEWORD=jarvis
REX_LLM_PROVIDER=openai
REX_SAMPLE_RATE=16000
```

Becomes in `config/rex_config.json`:
```json
{
  "wake_word": {
    "wakeword": "jarvis"
  },
  "models": {
    "llm_provider": "openai"
  },
  "audio": {
    "sample_rate": 16000
  }
}
```

## Best Practices

### Security
1. **Never commit .env to version control**
   - .env is in .gitignore by default
   - Contains sensitive API keys

2. **Backup .env separately**
   - Store in password manager or secure location
   - Do not store in public repositories

3. **Rotate API keys if exposed**
   - If .env is accidentally committed, rotate all keys immediately
   - Remove from git history using tools like git-filter-branch

### Configuration Management
1. **Use GUI for most changes**
   - Settings tab provides validation and help
   - Changes are applied immediately

2. **Edit JSON directly for batch changes**
   - Valid JSON required
   - Invalid JSON is backed up and replaced with defaults

3. **Keep backups**
   - GUI creates automatic backups in `backups/` directory
   - Manual backup: `cp config/rex_config.json config/rex_config.backup.json`

4. **Version control rex_config.json (optional)**
   - Safe to commit (contains no secrets)
   - Good for tracking configuration changes
   - Useful for team setups

### Audio Device Selection
1. **Use GUI for device selection**
   - Shows all available devices
   - Test button verifies device works
   - Saves automatically

2. **Device indices can change**
   - Plugging/unplugging USB devices changes indices
   - Re-select device if audio stops working
   - Use WASAPI devices on Windows (more reliable)

3. **Troubleshooting**
   - "No input devices found": Check microphone is plugged in
   - "Device failed": Try different device or restart audio service
   - WDM-KS devices: Not supported, use WASAPI or DirectSound instead

## Configuration Schema

See `config/rex_config.schema.json` for complete JSON Schema documentation.

## Troubleshooting

### Invalid Configuration File
**Symptom**: GUI shows error on startup or defaults to all settings

**Solution**:
1. Check `config/` directory for `.invalid.<timestamp>.json` file
2. This is your corrupt config, backed up automatically
3. Fix JSON syntax errors or delete to start fresh
4. Valid JSON is required (use JSONLint.com to validate)

### Legacy Environment Variables Ignored
**Symptom**: Changes to .env don't affect runtime behavior

**Solution**:
1. Check logs for "Legacy environment variables detected" warning
2. Edit `config/rex_config.json` instead of .env
3. Use GUI Settings tab for guided editing
4. Remove legacy settings from .env (optional cleanup)

### Audio Device Not Persisting
**Symptom**: Selected audio device resets on restart

**Solution**:
1. Verify `config/rex_config.json` exists and is writable
2. Check file permissions
3. Look for errors in System Log in GUI
4. Manually edit config file if needed

### Settings Not Taking Effect
**Symptom**: Changed settings don't work after save

**Solution**:
1. Some settings require restart (marked with ⚠ in GUI)
2. Click "Restart App" button in Settings tab
3. For CLI: Stop and restart rex_loop.py

## Test Coverage Gate

The current coverage floor is 25 percent. This is a temporary baseline that will be raised in steps as coverage improves, for example 25, 35, 45, 55.

Run tests locally on Windows with:

```bash
python -m pytest
```

## Reference

### Configuration Locations
- Runtime config: `config/rex_config.json`
- Config schema: `config/rex_config.schema.json`
- Example config: `config/rex_config.example.json`
- Secrets: `.env`
- Secrets template: `.env.example`
- Backups: `backups/`

### Related Documentation
- [README.md](README.md) - General project documentation
- [.env.example](.env.example) - Secrets template with descriptions
- [config/rex_config.example.json](config/rex_config.example.json) - Configuration example

### Support
- GitHub Issues: https://github.com/Blueibear/rex-ai-assistant/issues
- Check logs: `logs/rex.log`
- Enable debug logging: Set `"log_level": "DEBUG"` in rex_config.json
