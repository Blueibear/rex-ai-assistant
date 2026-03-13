# Rex AI Assistant — Docker

## Build and Run

```bash
# Build image
docker build -t rex-ai-assistant .

# Run with environment file
docker run --rm --env-file .env -it rex-ai-assistant

# Run with volume mounts for persistent data
docker run --rm --env-file .env \
  -v $(pwd)/Memory:/app/Memory \
  -v $(pwd)/transcripts:/app/transcripts \
  -v $(pwd)/models:/app/models \
  -it rex-ai-assistant

# Run TTS API server (expose port 5000)
docker run --rm --env-file .env -p 5000:5000 \
  -it rex-ai-assistant python rex_speak_api.py
```

**Note:** Docker image uses CPU-only PyTorch by default. For GPU support, modify the `Dockerfile` to install CUDA-enabled PyTorch wheels.
