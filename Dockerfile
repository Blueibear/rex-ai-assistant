# ============================================
# Stage 1: Dependencies - Install Python deps
# ============================================
FROM python:3.11.11-slim AS deps

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

# Copy dependency files
COPY pyproject.toml setup.py* ./
COPY rex/ ./rex/

# Install Python dependencies (CPU-only by default)
# Using torch 2.7.1 to avoid known DoS vulnerabilities in 2.6.0
RUN pip install --no-cache-dir --upgrade pip>=25.3 setuptools>=78.1.1 wheel && \
    pip install --no-cache-dir torch==2.7.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

# ============================================
# Stage 2: Runtime - Minimal runtime image
# ============================================
FROM python:3.11.11-slim AS runner

WORKDIR /app

# Install only runtime system dependencies (smaller than deps stage)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libasound2 \
        portaudio19-runtime \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python environment from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash rex && \
    mkdir -p /app/Memory /app/logs /app/transcripts /app/models && \
    chown -R rex:rex /app

# Switch to non-root user
USER rex

# Environment variables (set sensible defaults for containers)
ENV PYTHONUNBUFFERED=1 \
    REX_WAKEWORD=rex \
    REX_DEVICE=cpu \
    REX_LOG_LEVEL=INFO \
    REX_FILE_LOGGING_ENABLED=false \
    REX_WHISPER_DEVICE=cpu

# Mountable volumes for persistent state
VOLUME ["/app/Memory", "/app/models", "/app/transcripts"]

# Expose ports (if using Flask APIs)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command: use the module entrypoint
CMD ["python", "-m", "rex"]
