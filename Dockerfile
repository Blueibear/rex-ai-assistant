FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies required for audio libraries
COPY requirements.txt ./
RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        portaudio19-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Environment variables with sensible defaults for the assistant
ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=info

# Optional: mountable volume for logs or user memory
VOLUME ["/app/Memory"]

CMD ["python", "rex_assistant.py"]
