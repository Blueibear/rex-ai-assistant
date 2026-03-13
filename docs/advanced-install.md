# Advanced / Developer Install

This document covers installation options beyond the single-command `install.sh` / `install.ps1` scripts. Use it if you need GPU support, a custom extras selection, development tooling, or a Docker-based workflow.

---

## Manual Install (macOS / Linux)

```bash
# Clone repository
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Copy environment template
cp .env.example .env
# Edit .env with your preferred editor to set API keys and options

# Install base dependencies (no ML stack)
pip install --upgrade pip setuptools wheel
pip install .

# Optional: install CPU-only ML + audio stack
pip install -r requirements-cpu.txt

# Run health check
python scripts/doctor.py

# Start text-based chat mode
python -m rex

# Or start full voice assistant with wake word
python rex_loop.py
```

---

## Manual Install (Windows PowerShell)

```powershell
# Clone repository
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Copy environment template
Copy-Item .env.example .env
# Edit .env with your preferred editor to set API keys and options

# Install base dependencies (no ML stack)
pip install --upgrade pip setuptools wheel
pip install .

# Optional: install CPU-only ML + audio stack
pip install -r requirements-cpu.txt

# Run health check
python scripts/doctor.py

# Start text-based chat mode
python -m rex

# Or start full voice assistant with wake word
python rex_loop.py
```

---

## Pip Extras Reference

Install from source or a built wheel with optional extras:

```bash
pip install .                    # base only (no ML stack)
pip install -e ".[dev]"          # dev tooling (pytest, ruff, black, mypy)
pip install -e ".[ml,audio]"     # ML + audio stack
pip install -e ".[full]"         # full install (ml + audio + sms)
```

`requirements.txt` serves as a pointer with guidance; use the split requirements files for CPU/GPU installs to avoid CUDA-only wheels in CI.

---

## Interactive Installer

Rex includes a Python-based interactive installer for additional setup options:

```bash
# Basic installation
python install.py

# Include ML models (Whisper, XTTS)
python install.py --with-ml

# Include development tools (pytest, ruff, black, mypy)
python install.py --with-dev

# Auto-install ffmpeg (Linux/macOS only)
python install.py --auto-install-ffmpeg

# Test audio devices
python install.py --mic-test
```

---

## Full vs. Lean Install Scripts

Use these scripts for supervised deployments with the service supervisor enabled:

```bash
# Full install with optional extras (sms + devtools) and systemd service setup
./install_full.sh

# Lean node install with minimal dependencies and a trimmed service list
./install_lean.sh
```

Optional environment overrides:

```bash
REX_SERVICE_PORT=8765 REX_SKIP_SERVICE=1 ./install_full.sh
REX_SERVICES=event_bus,workflow_runner,memory_store,credential_manager ./install_lean.sh
```

---

## GPU Acceleration (Optional)

### CUDA 12.4 (Recommended for Windows 11)

For NVIDIA GPUs with CUDA 12.4, use the CUDA 12.4 requirements file:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu-cu124.txt
```

This installs:
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`

**Verify GPU is detected:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### CPU-Only Installation (no CUDA)

For development, CI, or systems without GPU:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cpu.txt
```

### Alternative: CUDA 11.8

For systems with CUDA 11.8:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-gpu.txt
```

---

## Development Setup

```bash
# Activate development dependencies
pip install -e .[dev]

# Lint with Ruff
ruff check .

# Format with Black
black .

# Type check with Mypy
mypy .

# Run all linting/formatting
ruff check . && black --check . && mypy .
```

### Running Tests

```bash
# Install test dependencies
pip install -e .[test]

# Run all tests
pytest

# Run with coverage
pytest --cov=rex --cov-report=html

# Run only unit tests (skip slow/audio/GPU tests)
pytest -m "not slow and not audio and not gpu"

# Run specific test file
pytest tests/test_config.py

# Verbose output
pytest -v
```
