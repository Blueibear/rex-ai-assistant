"""Windows Agent HTTP server (Cycle 5.3).

Minimal, secure HTTP server that runs on a Windows (or any) machine and exposes
the endpoints that the Rex ``AgentClient`` expects:

- ``GET /health``  → ``{"status": "ok"}``
- ``GET /status``  → host info (hostname, OS, user, time)
- ``POST /run``    → run an allowlisted command via subprocess, return JSON result

Security model
--------------
- Binds to ``127.0.0.1`` by default.  Change ``REX_AGENT_HOST`` to expose on
  a network interface — only do this behind a TLS-terminating reverse proxy.
- Every request requires the ``X-Auth-Token`` header.  The token is loaded from
  the environment variable named by ``REX_AGENT_TOKEN_ENV`` (default:
  ``REX_AGENT_TOKEN``).  The token is **never** logged.
- Commands are only executed if they appear in the server-side allowlist
  (``REX_AGENT_ALLOWLIST``).  This is defence-in-depth on top of the
  client-side allowlist in ``rex/computers/service.py``.
- In-memory fixed-window rate limiter (stdlib only, no flask-limiter dep).
- ``subprocess`` is invoked with ``shell=False``.
- Command output is truncated to ``REX_AGENT_MAX_OUTPUT`` bytes (default 64 KiB).
- Execution timeout is enforced via ``subprocess.communicate(timeout=...)``.

Environment variables
---------------------
``REX_AGENT_TOKEN``         Auth token (required — server refuses to start without it).
``REX_AGENT_TOKEN_ENV``     Name of the env var that holds the token (default:
                             ``REX_AGENT_TOKEN``).  Useful when you want the token
                             in a differently-named var.
``REX_AGENT_HOST``          Bind host (default: ``127.0.0.1``).
``REX_AGENT_PORT``          Bind port (default: ``7777``).
``REX_AGENT_ALLOWLIST``     Comma-separated command names allowed for execution
                             (default: ``whoami``).
``REX_AGENT_RATE_LIMIT``    Max requests per client IP per minute (default: ``60``).
``REX_AGENT_TIMEOUT``       Subprocess execution timeout in seconds (default: ``30``).
``REX_AGENT_MAX_OUTPUT``    Max bytes of stdout/stderr returned (default: ``65536``).
"""

from __future__ import annotations

import hmac
import logging
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from getpass import getuser
from typing import Any

from flask import Flask, Response, jsonify, request
from rex.exception_handler import wrap_entrypoint
from rex.graceful_shutdown import get_shutdown_handler
from rex.health import check_config, create_health_blueprint
from rex.request_logging import install_request_logging
from rex.startup_validation import check_startup_env

logger = logging.getLogger("rex.agent_server")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

AUTH_HEADER = "X-Auth-Token"

_DEFAULT_TOKEN_ENV = "REX_AGENT_TOKEN"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7777
_DEFAULT_ALLOWLIST = "whoami"
_DEFAULT_RATE_LIMIT = 60  # requests per minute
_DEFAULT_TIMEOUT = 30  # seconds
_DEFAULT_MAX_OUTPUT = 65536  # bytes


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if raw.strip().isdigit():
        return int(raw.strip())
    return default


def _load_token() -> str:
    """Load the auth token from the environment.

    The env var name can itself be overridden via ``REX_AGENT_TOKEN_ENV``.
    """
    token_env = os.getenv("REX_AGENT_TOKEN_ENV") or _DEFAULT_TOKEN_ENV
    return os.getenv(token_env, "")


def _load_allowlist() -> frozenset[str]:
    raw = os.getenv("REX_AGENT_ALLOWLIST", _DEFAULT_ALLOWLIST)
    return frozenset(cmd.strip() for cmd in raw.split(",") if cmd.strip())


# ---------------------------------------------------------------------------
# In-memory fixed-window rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Fixed-window per-IP rate limiter backed by in-memory deques.

    Thread-safety: Flask development server is single-threaded but the
    production WSGI server may be multi-threaded.  The deque append/popleft
    operations are GIL-protected on CPython, which is sufficient here.
    """

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str) -> bool:
        """Return ``True`` if the request is within the rate limit."""
        if self._limit <= 0:
            return True  # rate limiting disabled
        now = time.monotonic()
        bucket = self._buckets[key]
        # Evict timestamps outside the current window.
        while bucket and now - bucket[0] > self._window:
            bucket.popleft()
        if len(bucket) >= self._limit:
            return False
        bucket.append(now)
        return True


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    token: str | None = None,
    allowlist: frozenset[str] | None = None,
    rate_limit: int | None = None,
    cmd_timeout: int | None = None,
    max_output: int | None = None,
) -> Flask:
    """Create and return the configured Flask application.

    Parameters may be supplied directly (useful for testing) or read from
    environment variables when ``None``.

    Args:
        token:       Auth token.  Uses ``_load_token()`` if ``None``.
        allowlist:   Set of allowed command names.  Uses ``_load_allowlist()``
                     if ``None``.
        rate_limit:  Max requests per IP per minute.  Uses env var if ``None``.
        cmd_timeout: Subprocess timeout in seconds.  Uses env var if ``None``.
        max_output:  Max output bytes.  Uses env var if ``None``.

    Returns:
        Configured :class:`flask.Flask` application instance.
    """
    _token: str = token if token is not None else _load_token()
    _allowlist: frozenset[str] = allowlist if allowlist is not None else _load_allowlist()
    _rate_limit: int = (
        rate_limit
        if rate_limit is not None
        else _parse_int_env("REX_AGENT_RATE_LIMIT", _DEFAULT_RATE_LIMIT)
    )
    _cmd_timeout: int = (
        cmd_timeout
        if cmd_timeout is not None
        else _parse_int_env("REX_AGENT_TIMEOUT", _DEFAULT_TIMEOUT)
    )
    _max_output: int = (
        max_output
        if max_output is not None
        else _parse_int_env("REX_AGENT_MAX_OUTPUT", _DEFAULT_MAX_OUTPUT)
    )

    limiter = _RateLimiter(limit=_rate_limit, window_seconds=60)

    app = Flask(__name__, static_folder=None)
    install_request_logging(app)
    app.register_blueprint(create_health_blueprint(checks=[check_config]))

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth() -> Response | None:
        """Return a 401 response if the request is not authenticated.

        Returns ``None`` if authentication passes.
        """
        provided = request.headers.get(AUTH_HEADER, "")
        if not _token:
            # Server is misconfigured — refuse all requests.
            logger.error("Agent token is not configured; refusing request")
            return jsonify({"error": "Server misconfigured: token not set"}), 401  # type: ignore[return-value]
        if not provided:
            return jsonify({"error": "Missing auth token"}), 401  # type: ignore[return-value]
        if not hmac.compare_digest(provided, _token):
            logger.warning("Rejected request from %s: invalid token", request.remote_addr)
            return jsonify({"error": "Invalid auth token"}), 401  # type: ignore[return-value]
        return None

    # ------------------------------------------------------------------
    # Rate-limit helper
    # ------------------------------------------------------------------

    def _check_rate_limit() -> Response | None:
        """Return a 429 response if the client IP is over the rate limit."""
        key = request.remote_addr or "unknown"
        if not limiter.is_allowed(key):
            logger.warning("Rate limit exceeded for %s", key)
            return jsonify({"error": "Too many requests"}), 429  # type: ignore[return-value]
        return None

    # ------------------------------------------------------------------
    # Global request guards
    # ------------------------------------------------------------------

    @app.before_request
    def _reject_during_shutdown() -> Response | tuple[Response, int] | None:
        """Return 503 when a SIGTERM-triggered shutdown is in progress."""
        if get_shutdown_handler().is_shutting_down:
            return jsonify({"error": "Server is shutting down"}), 503
        return None

    @app.before_request
    def _guard_request() -> Response | tuple[Response, int] | None:
        """Enforce auth and rate limits for all agent API routes.

        Applying guards at ``before_request`` ensures protections also apply to
        automatically generated methods such as ``OPTIONS``.
        """
        if request.path not in {"/health", "/status", "/run"}:
            return None

        auth_err = _check_auth()
        if auth_err is not None:
            return auth_err

        rate_err = _check_rate_limit()
        if rate_err is not None:
            return rate_err

        return None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        return jsonify({"status": "ok"}), 200

    @app.route("/status", methods=["GET"])
    def status() -> tuple[Response, int]:
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            current_user = getuser()
        except Exception:  # noqa: BLE001
            current_user = "unknown"

        return (
            jsonify(
                {
                    "hostname": socket.gethostname(),
                    "os": platform.platform(),
                    "user": current_user,
                    "time": now,
                }
            ),
            200,
        )

    @app.route("/run", methods=["POST"])
    def run() -> tuple[Response, int]:
        payload: Any = request.get_json(silent=True)
        if not payload or not isinstance(payload, dict):
            return jsonify({"error": "Request body must be JSON"}), 400

        command = payload.get("command")
        if not command or not isinstance(command, str):
            return jsonify({"error": "'command' field is required and must be a string"}), 400

        args_raw = payload.get("args", [])
        if not isinstance(args_raw, list):
            return jsonify({"error": "'args' must be a list"}), 400
        args: list[str] = [str(a) for a in args_raw]

        cwd: str | None = payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            return jsonify({"error": "'cwd' must be a string or null"}), 400

        # Server-side allowlist check (defence in depth).
        if command not in _allowlist:
            allowed_display = ", ".join(sorted(_allowlist)) or "(none)"
            logger.warning(
                "DENIED run attempt: remote=%s command=%r allowlist=[%s]",
                request.remote_addr,
                command,
                allowed_display,
            )
            return jsonify({"error": f"Command {command!r} is not allowlisted"}), 403

        # Execute the command.
        argv = [command] + args
        start_ns = time.monotonic_ns()
        exit_code = -1
        stdout_str = ""
        stderr_str = ""

        try:
            proc = subprocess.run(  # noqa: S603
                argv,
                shell=False,
                capture_output=True,
                timeout=_cmd_timeout,
                cwd=cwd,
            )
            exit_code = proc.returncode
            stdout_bytes = proc.stdout or b""
            stderr_bytes = proc.stderr or b""
            # Truncate to safe limit.
            if len(stdout_bytes) > _max_output:
                stdout_bytes = stdout_bytes[:_max_output]
            if len(stderr_bytes) > _max_output:
                stderr_bytes = stderr_bytes[:_max_output]
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)
            logger.warning(
                "TIMEOUT run: remote=%s command=%r duration_ms=%d",
                request.remote_addr,
                command,
                duration_ms,
            )
            return (
                jsonify(
                    {
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": f"Command timed out after {_cmd_timeout}s",
                        "duration_ms": duration_ms,
                    }
                ),
                200,
            )
        except FileNotFoundError:
            duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)
            logger.warning(
                "NOT FOUND run: remote=%s command=%r",
                request.remote_addr,
                command,
            )
            return (
                jsonify(
                    {
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": f"Command not found: {command!r}",
                        "duration_ms": duration_ms,
                    }
                ),
                200,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)
            logger.error(
                "ERROR run: remote=%s command=%r error=%s",
                request.remote_addr,
                command,
                exc,
            )
            return (
                jsonify(
                    {
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": f"Execution error: {exc}",
                        "duration_ms": duration_ms,
                    }
                ),
                200,
            )

        duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)
        logger.info(
            "RUN: remote=%s command=%r exit_code=%d duration_ms=%d",
            request.remote_addr,
            command,
            exit_code,
            duration_ms,
        )
        return (
            jsonify(
                {
                    "exit_code": exit_code,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "duration_ms": duration_ms,
                }
            ),
            200,
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance and entry point
# ---------------------------------------------------------------------------


def _build_default_app() -> Flask:
    """Build the app with settings from environment variables."""
    return create_app()


@wrap_entrypoint
def main() -> None:
    """Entry point for the Windows agent server.

    Reads all configuration from environment variables.  Refuses to start if
    ``REX_AGENT_TOKEN`` (or the var named by ``REX_AGENT_TOKEN_ENV``) is not set.
    """
    check_startup_env()
    if not logger.handlers:
        logging.basicConfig(
            level=os.getenv("REX_LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    token = _load_token()
    if not token:
        token_env = os.getenv("REX_AGENT_TOKEN_ENV") or _DEFAULT_TOKEN_ENV
        raise SystemExit(
            f"Error: Auth token not set.  "
            f"Set the {token_env!r} environment variable before starting the agent."
        )

    host = os.getenv("REX_AGENT_HOST", _DEFAULT_HOST)
    port = _parse_int_env("REX_AGENT_PORT", _DEFAULT_PORT)

    app = _build_default_app()

    # Log startup info — but never log the token itself.
    logger.info("Rex Windows Agent starting on %s:%d", host, port)
    logger.info("Allowlist: %s", ", ".join(sorted(_load_allowlist())) or "(none)")
    logger.info(
        "Rate limit: %d req/min per IP", _parse_int_env("REX_AGENT_RATE_LIMIT", _DEFAULT_RATE_LIMIT)
    )

    shutdown = get_shutdown_handler()
    shutdown.install()

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "AUTH_HEADER",
    "create_app",
    "main",
]
