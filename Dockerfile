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
        portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire source code into image
COPY . .

# Environment variables (set sensible defaults)
ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=info

# Mountable volume for memory or logs
VOLUME ["/app/Memory"]

# Default command
CMD ["python", "rex_assistant.py"]
