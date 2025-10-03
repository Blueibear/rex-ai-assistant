#!/bin/bash
set -e

echo "🔧 Starting Rex AI Assistant environment setup (Python $(python3 --version))..."

# 1. System dependencies
apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libespeak-ng-dev \
    libsndfile1-dev \
    build-essential \
    python3-dev \
    curl \
    git

# 2. Upgrade pip tools
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install Python dependencies from requirements.txt
pip install -r requirements.txt || true

# 5. Install Coqui TTS (compatible with Python 3.10)
pip install TTS==0.17.3

# 6. Download wake-word ONNX model if missing
mkdir -p data/models
if [ ! -f data/models/rex.onnx ]; then
    echo "⬇️  Downloading default wake-word model..."
    curl -L -o data/models/rex.onnx https://huggingface.co/spaces/Blueibear/rex-onnx/resolve/main/hey_jarvis.onnx
fi

# 7. Preload Models (optional for faster warmups)
echo "🧠 Preloading Whisper model..."
python3 -c "import whisper; whisper.load_model('base')" || true

echo "🤖 Preloading transformer model..."
python3 -c "from transformers import pipeline; pipeline('text-generation', model='distilgpt2')" || true

echo "🔊 Preloading Coqui TTS model..."
python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')" || true

echo "✅ Rex AI Assistant setup complete!"
