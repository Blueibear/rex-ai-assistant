# Rex Doctor

`rex doctor` is a diagnostic command that checks your AskRex Assistant environment for common issues and misconfigurations.

## Purpose

Running `rex doctor` helps you:

- Verify your Python version is supported
- Check that configuration files exist and are valid
- Confirm required API keys are set in environment variables
- Ensure external dependencies (like `ffmpeg`) are installed
- Detect basic security issues with file permissions
- Verify core Python packages are installed

## Usage

### Basic Usage

```bash
rex doctor
```

Or via the Python module:

```bash
python -m rex doctor
```

### Verbose Mode

For more detailed output, including individual dependency checks:

```bash
rex doctor -v
```

### Help

```bash
rex doctor --help
```

## What It Checks

### 1. Python Version

Checks that your Python version is supported:

- **OK**: Python 3.11
- **Error**: Any other Python version

### 2. Package Installation

Verifies that the `rex` package is properly installed:

```bash
pip install -e .
```

### 3. Configuration File

Checks for `config/rex_config.json`:

- **OK**: File exists and contains valid JSON
- **Warning**: File doesn't exist but `rex_config.example.json` does
- **Error**: File exists but contains invalid JSON

### 4. Environment File

Checks for `.env` file:

- **OK**: File exists
- **Warning**: File doesn't exist (API keys won't be loaded)

### 5. API Keys

Checks for common API key environment variables:

- `OPENAI_API_KEY`
- `OLLAMA_API_KEY`
- `BRAVE_API_KEY`
- `SERPAPI_KEY`
- `GOOGLE_API_KEY`
- `HA_TOKEN`

At least one API key is typically needed for Rex to function.

### 6. External Dependencies

Checks that required binaries are on your PATH:

- **ffmpeg**: Required for audio processing
- **git**: Required for version control

### 7. GPU Availability

Checks if CUDA GPU is available for ML inference:

- **OK**: CUDA GPU detected
- **Info**: No GPU available (CPU mode will be used)

### 8. Config Permissions

Checks for potential security issues:

- Warns if `.env` file is world-readable (others can read your API keys)

### 9. Core Dependencies (Verbose Mode)

In verbose mode (`-v`), also checks that core Python packages are installed:

- torch (PyTorch)
- transformers
- whisper
- TTS
- openwakeword
- flask
- pydantic
- dotenv

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | All checks passed (or only warnings) |
| 1    | One or more critical errors found |

## Output Format

```
Rex Doctor - Environment Diagnostics
========================================

Project root: /path/to/rex-ai-assistant

[OK]     Python Version: Python 3.11.0
[OK]     Package Installation: rex package installed (contracts v0.1.0)
[OK]     Config File: Found and readable: rex_config.json
[OK]     Environment File: Found: .env
[INFO]   API Keys: 2 API key(s) configured
[OK]     Config Permissions: Configuration permissions look reasonable
[OK]     Binary: ffmpeg: Found: /usr/bin/ffmpeg
[OK]     Binary: git: Found: /usr/bin/git
[INFO]   GPU Availability: No CUDA GPU available (CPU mode)

----------------------------------------
PASSED with warnings: 0 warning(s)
Rex should work, but consider addressing the warnings above.
```

## Troubleshooting

### "rex_config.json not found"

Copy the example configuration file:

```bash
cp config/rex_config.example.json config/rex_config.json
```

Then customize it for your setup.

### ".env not found"

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### "No API keys configured"

Add at least one API key to your `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### "ffmpeg not found"

Install ffmpeg:

- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

### "Python X.X is not supported"

Use Python 3.11. The current Rex dependency stack is validated on Python 3.11, and fresh installs on Python 3.13/3.14 are known to fail in the ML/TTS path.

### ".env file is world-readable"

Restrict permissions on your `.env` file:

```bash
chmod 600 .env
```

## Programmatic Usage

You can also use the doctor module programmatically:

```python
from rex.doctor import run_diagnostics

# Run all checks
exit_code = run_diagnostics(verbose=True)

# Individual checks
from rex.doctor import (
    check_python_version,
    check_config_file,
    check_environment_variables,
)

result = check_python_version()
print(f"{result.status}: {result.message}")
```

## See Also

- [Installation Guide](DEPLOYMENT_CHECKLIST.md)
- [Configuration Reference](contracts.md)
- [Architecture Overview](ARCHITECTURE.md)
