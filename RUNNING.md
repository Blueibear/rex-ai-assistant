# How to Run Rex AI Assistant

This guide provides exact commands for running Rex on Windows.

## Quick Start (Windows PowerShell)

### 1. Create and Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install all requirements
pip install -r requirements.txt
```

### 3. Configure Environment

```powershell
# Copy the example environment file
Copy-Item .env.example .env

# Edit .env with your preferred text editor
notepad .env
```

**Key settings to configure:**
- `REX_ACTIVE_USER` - Your user profile name
- `REX_LLM_PROVIDER` - Choose: transformers, openai, or ollama
- `REX_LLM_MODEL` - Model to use (depends on provider)
- `OPENAI_API_KEY` - If using OpenAI (required for OpenAI provider)
- `REX_WHISPER_MODEL` - Choose: tiny, base, small, medium, large

### 4. Launch the GUI

```powershell
# Option A: Using the dedicated GUI entrypoint (recommended)
python run_gui.py

# Option B: Using gui.py directly
python gui.py
```

The GUI will open with two tabs:
- **Dashboard**: Start/stop the assistant, view conversation history
- **Settings**: Configure all environment variables with a visual editor

### 5. Launch Voice Assistant (CLI Mode)

```powershell
# Start the full voice loop with wake word detection
python rex_loop.py

# Override the active user
python rex_loop.py --user james

# Enable specific plugins only
python rex_loop.py --enable-plugin web_search
```

## Common Issues

### Issue: Import Error

**Error:** `ImportError: cannot import name 'AsyncRexAssistant'`

**Solution:** Make sure you pulled the latest code. The import has been fixed in `gui.py`.

### Issue: MQTT Warnings

**Error:** Warnings about `asyncio-mqtt` or `aiomqtt`

**Solution:** MQTT is optional. The app will work without it. To install:
```powershell
pip install aiomqtt
```

### Issue: FFmpeg/torio Warnings

**Error:** Warnings about FFmpeg extensions not loading

**Solution:** These are non-critical. FFmpeg is only needed for specific audio codec features. To reduce noise, you can set logging level to WARNING in your .env:
```
REX_LOG_LEVEL=WARNING
```

### Issue: jieba pkg_resources Warning

**Error:** `DeprecationWarning: pkg_resources is deprecated`

**Solution:** This is a known issue with jieba. It's non-fatal. To suppress:
```powershell
# Set environment variable to hide deprecation warnings
$env:PYTHONWARNINGS="ignore::DeprecationWarning"
python run_gui.py
```

Or add to your PowerShell profile for persistence.

### Issue: Home Assistant AttributeError

**Error:** `AttributeError: 'AppConfig' object has no attribute 'ha_base_url'`

**Solution:** This was fixed in recent commits. Home Assistant integration is now optional. Rex will work without HA configured. If you want to use Home Assistant integration, add to your .env:
```
HA_BASE_URL=http://homeassistant.local:8123
HA_TOKEN=your_long_lived_access_token
```

## Verification Script

Run the import checker to verify all modules are syntactically correct:

```powershell
python check_imports.py
```

## Updating Dependencies

```powershell
# Update all packages to latest compatible versions
pip install --upgrade -r requirements.txt
```

## Running Tests

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_voice_loop.py

# Run with verbose output
pytest -v
```

## Troubleshooting

### Virtual Environment Not Activating

If `.\.venv\Scripts\Activate.ps1` doesn't work:

1. Check execution policy:
   ```powershell
   Get-ExecutionPolicy
   ```

2. If it's "Restricted", change it:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Try again:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

### Missing Dependencies

If you see `ModuleNotFoundError` for numpy, torch, tkinter, etc.:

```powershell
# Reinstall all requirements
pip install --force-reinstall -r requirements.txt
```

### Performance Issues

For faster startup and better performance:

1. Use smaller models:
   ```
   REX_WHISPER_MODEL=tiny
   REX_LLM_MODEL=distilgpt2
   ```

2. Use CPU-optimized PyTorch (already in requirements.txt)

3. Disable debug logging:
   ```
   REX_DEBUG_LOGGING=false
   ```

## Development Mode

To run Rex from source for development:

```powershell
# Install in editable mode
pip install -e .

# Run from anywhere
rex --help
```

## Additional Resources

- **README.md** - Project overview and features
- **.env.example** - Full list of configuration options
- **gui_settings_tab.py** - Settings editor implementation
- **tests/** - Unit tests and examples
