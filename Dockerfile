python:3.11-slim

# Set work directory
WORKDIR /app

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

ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=info

VOLUME ["/app/Memory"]

CMD ["python", "rex_assistant.py"]
