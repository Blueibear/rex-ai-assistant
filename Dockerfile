# Use the Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies for audio and TTS/ASR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        portaudio19-dev \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for better layer caching
COPY pyproject.toml ./

# Install Python dependencies (CPU-only by default)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Copy source code and install the package
COPY . .
RUN pip install --no-cache-dir -e .

# Environment variables (set sensible defaults for containers)
ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=INFO \
    REX_FILE_LOGGING_ENABLED=false \
    REX_WHISPER_DEVICE=cpu

# Create directories for state
RUN mkdir -p /app/Memory /app/logs /app/transcripts /app/models

# Mountable volumes for persistent state
VOLUME ["/app/Memory", "/app/models", "/app/transcripts"]

# Default command: use the module entrypoint
CMD ["python", "-m", "rex"]
