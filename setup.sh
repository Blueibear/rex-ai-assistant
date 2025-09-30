#!/bin/bash
set -euo pipefail

echo "Starting Rex AI Assistant environment setup (Python $(python3 --version))..."

# 1. System dependencies
apt-get update
apt-get install -y \
    ffmpeg \
    espeak-ng \
    libespeak-ng-dev \
    libsndfile1-dev \
    portaudio19-dev \
    build-essential \
    python3-dev \
    curl \
    git

# 2. Upgrade pip tooling
pip install --upgrade pip setuptools wheel

# 3. Install the CPU PyTorch stack pinned to supported versions
pip install \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Optionally preload heavyweight models (skipped on failure)
python3 -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')" || true
python3 -c "from transformers import pipeline; pipeline('text-generation', model='distilgpt2')" || true
python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)" || true

cat <<'MSG'
Setup complete.
If you plan to use GPU acceleration, enable it by setting REX_GPU=true and ensure CUDA drivers are installed.
MSG
