# AskRex Assistant Docker

The Dockerfile uses Python 3.11 and CPU PyTorch by default. The default command
is `python -m rex`.

## Build and Run

```bash
docker build -t rex-ai-assistant .
docker run --rm --env-file .env -it rex-ai-assistant
```

Persist runtime state:

```bash
docker run --rm --env-file .env \
  -v "$(pwd)/Memory:/app/Memory" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -it rex-ai-assistant
```

Run the TTS API on its current default port:

```bash
docker run --rm --env-file .env -p 5005:5005 \
  -it rex-ai-assistant rex-speak-api
```

Run the tool server:

```bash
docker run --rm --env-file .env -p 18790:18790 \
  -it rex-ai-assistant rex-tool-server
```

The image is CPU-oriented. For GPU support, update the Dockerfile to install
the CUDA-enabled PyTorch wheels that match the host and driver stack.
