# Use the Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system-level dependencies required for audio + voice functionality
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source code into container
COPY . .

# Set default environment variables
ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=info

# Optional: mountable volume for user memory or logs
VOLUME ["/app/Memory"]

# Set default entry point
CMD ["python", "rex_assistant.py"]

