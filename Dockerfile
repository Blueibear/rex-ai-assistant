FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libasound2 \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=info

# Optional: Mountable volume for logs or user memory
VOLUME ["/app/Memory"]

# Entry point
CMD ["python", "rex_assistant.py"]
