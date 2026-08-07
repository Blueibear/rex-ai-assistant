# AskRex Assistant Docker

> **Support status: developer-only.** Docker is an operator/development path for
> smoke testing and service experiments. It is not an end-user release artifact
> and is not the supported production deployment path. The supported end-user
> artifact is the Windows Electron Voice installer.

The Dockerfile uses Python 3.11 and CPU PyTorch by default. The default command
is `python -m rex`. Its Docker `HEALTHCHECK` runs `python -m rex doctor --healthcheck`,
a lightweight liveness probe that fails when the core Python/package/CLI runtime
cannot load. Use plain `python -m rex doctor` for broader readiness diagnostics.

## Build and Run

```bash
docker build -t askrex-assistant:smoke .
docker run --rm askrex-assistant:smoke python -m rex doctor
```

For an interactive developer container with explicitly supplied legacy plaintext
credentials, opt in to that unpackaged fallback and pass the environment file:

```bash
docker run --rm --env-file .env \
  -e REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK=1 \
  -it askrex-assistant:smoke
```

Persist developer runtime state only when needed:

```bash
docker run --rm --env-file .env \
  -e REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK=1 \
  -v "$(pwd)/Memory:/app/Memory" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -it askrex-assistant:smoke
```

The image is CPU-oriented. GPU container support is not a supported AskRex
release path; developers experimenting with it must supply CUDA/PyTorch wheels
that match their own host and driver stack.
