# Rex AI Assistant - Installation Guide

This guide covers all installation methods and platform-specific setup for the Rex AI Assistant.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
  - [CPU-Only (Recommended for Development)](#cpu-only-recommended-for-development)
  - [GPU-Accelerated (Production)](#gpu-accelerated-production)
  - [Docker](#docker)
- [Feature Matrix](#feature-matrix)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows (WSL2 recommended)
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 5GB free space (for models)
- **Audio**: Working microphone and speakers

### Optional Requirements
- **GPU**: NVIDIA GPU with CUDA 11.8+ (for GPU acceleration)
- **VRAM**: 4GB+ for GPU-accelerated speech models

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU-only version (fastest for development)
pip install -r requirements-cpu.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys and preferences

# Run Rex
python -m rex
# Or: python rex_assistant.py
```

## Installation Methods

### CPU-Only (Recommended for Development)

CPU-only installation is faster, lighter, and sufficient for most development and testing scenarios.

```bash
# Install CPU-only PyTorch and dependencies
pip install -r requirements-cpu.txt

# Or install from pyproject.toml
pip install -e .
```

**Pros:**
- Fast installation (~500MB download)
- Lower memory usage
- No GPU drivers required
- Works on all platforms

**Cons:**
- Slower speech recognition and synthesis
- Not recommended for real-time production use

### GPU-Accelerated (Production)

GPU acceleration provides significantly faster speech processing for production deployments.

```bash
# Install CUDA 11.8 compatible PyTorch
pip install -r requirements-gpu.txt

# Or install from pyproject.toml with GPU extras
pip install -e .[gpu-cu118]

# For CUDA 12.1
pip install -e .[gpu-cu121]
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8 or 12.1
- 4GB+ VRAM

**Pros:**
- 5-10x faster speech recognition
- Real-time TTS generation
- Better for production workloads

**Cons:**
- Larger installation (~3GB)
- Requires GPU drivers and CUDA

### Docker

Docker provides a consistent, isolated environment for running Rex.

```bash
# Build the image (CPU-only by default)
docker build -t rex-assistant .

# Run interactively
docker run -it --rm \
  -v $(pwd)/Memory:/app/Memory \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  rex-assistant

# Run as daemon
docker run -d \
  --name rex \
  -v rex-memory:/app/Memory \
  -v rex-models:/app/models \
  --env-file .env \
  rex-assistant
```

**GPU Support in Docker:**
```bash
# Build GPU version
docker build -t rex-assistant:gpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  .

# Run with GPU
docker run -it --rm --gpus all \
  -v $(pwd)/Memory:/app/Memory \
  --env-file .env \
  rex-assistant:gpu
```

## Feature Matrix

| Feature | CPU-Only | GPU (CUDA) | OpenAI Backend |
|---------|----------|------------|----------------|
| **Wake Word Detection** | ✅ Fast | ✅ Fast | ✅ Fast |
| **Speech Recognition** | ⚠️ Slow (5-15s) | ✅ Fast (<1s) | ✅ Fast (<1s) |
| **Language Model** | ✅ Local | ✅ Local | ✅ Cloud |
| **Text-to-Speech** | ⚠️ Slow (5-10s) | ✅ Fast (<1s) | ✅ Fast (<1s) |
| **Offline Mode** | ✅ Yes | ✅ Yes | ❌ No |
| **Privacy** | ✅ Fully Local | ✅ Fully Local | ⚠️ Data sent to OpenAI |
| **Cost** | 💰 Free | 💰 Free | 💰💰 Pay per use |
| **Memory Usage** | ~2GB | ~4GB | ~1GB |
| **Installation Size** | ~500MB | ~3GB | ~200MB |

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    python3-venv

# Continue with Quick Start
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install ffmpeg portaudio python@3.11

# Continue with Quick Start
```

### Windows

**Option 1: WSL2 (Recommended)**
1. Install WSL2 and Ubuntu: https://docs.microsoft.com/en-us/windows/wsl/install
2. Follow Linux instructions above

**Option 2: Native Windows**
1. Install Python 3.11 from https://www.python.org/downloads/
2. Install Git from https://git-scm.com/download/win
3. See `README.windows.md` for detailed instructions

## Configuration

### Environment Variables

All configuration is done via environment variables. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

**Key Variables:**

```bash
# Required for API functionality
REX_SPEAK_API_KEY=your-secret-key-here

# Optional: Use OpenAI for better quality
OPENAI_API_KEY=sk-your-openai-key
REX_LLM_BACKEND=openai
REX_LLM_MODEL=gpt-3.5-turbo

# Audio configuration
REX_WAKEWORD=rex
REX_WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
REX_WHISPER_DEVICE=cpu  # Or cuda for GPU

# CORS configuration (if running API)
REX_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000
```

See `.env.example` for the complete list of configuration options.

## Running Tests

```bash
# Install test dependencies
pip install -e .[dev,test]

# Run all unit tests
pytest -m unit

# Run all tests (including slow/integration tests)
pytest

# Run with coverage
pytest --cov=rex --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## Troubleshooting

### Audio Issues

**Problem:** No audio input detected
```bash
# List available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set device index in .env
REX_INPUT_DEVICE=0  # Replace with your device index
```

**Problem:** Audio output not working
```bash
# Test audio output
python -c "import simpleaudio; simpleaudio.WaveObject([0x00]*1000, 1, 1, 16000).play().wait_done()"
```

### Memory Issues

**Problem:** Out of memory during model loading

**Solution:** Use smaller models or CPU-only mode:
```bash
REX_WHISPER_MODEL=tiny  # Smallest model
REX_WHISPER_DEVICE=cpu
REX_TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC  # Lighter TTS model
```

### Import Errors

**Problem:** `ModuleNotFoundError` for torch/transformers

**Solution:** Reinstall dependencies with clean cache:
```bash
pip cache purge
pip uninstall torch torchvision torchaudio -y
pip install -r requirements-cpu.txt --no-cache-dir
```

### CUDA Errors

**Problem:** CUDA version mismatch

**Solution:** Check CUDA version and install matching PyTorch:
```bash
nvidia-smi  # Check CUDA version
# Install matching PyTorch from https://pytorch.org/get-started/locally/
```

### Doctor Script

Run the built-in diagnostics:
```bash
python scripts/environment_doctor.py
```

This will check:
- Python version
- Dependency versions
- Audio device availability
- Model download status
- Environment configuration

## Development Mode

For active development with hot-reloading:

```bash
# Install in editable mode with dev dependencies
pip install -e .[dev]

# Enable debug logging
export REX_DEBUG_LOGGING=true
export REX_LOG_LEVEL=DEBUG

# Run tests on file changes
pytest-watch
```

## Uninstallation

```bash
# Remove pip package
pip uninstall rex-ai-assistant

# Remove virtual environment
rm -rf venv/

# Remove downloaded models and state (optional)
rm -rf models/ Memory/ transcripts/ logs/
```

## Getting Help

- **Documentation**: See `README.md` for usage guide
- **Issues**: Report bugs at https://github.com/Blueibear/rex-ai-assistant/issues
- **Discussions**: Ask questions in GitHub Discussions
- **Windows Users**: See `README.windows.md` for platform-specific tips

## Next Steps

After installation:

1. Configure your `.env` file with API keys
2. Test audio devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
3. Run the environment doctor: `python scripts/environment_doctor.py`
4. Start Rex: `python -m rex`
5. Say your wake word (default: "rex") and start chatting!
