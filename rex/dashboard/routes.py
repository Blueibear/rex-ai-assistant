"""Dashboard routes and API endpoints.

Provides a Flask Blueprint with routes for:
- Dashboard UI (HTML)
- Authentication (login/logout)
- Settings management
- Chat interface
- Scheduler/reminders management
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)

from rex.config_manager import (
    DEFAULT_CONFIG,
    load_config,
    save_config,
)
from rex.contracts import redact_sensitive_keys
from rex.dashboard.auth import (
    Session,
    get_session_manager,
    is_password_required,
    verify_password,
)
from rex.logging_utils import get_logger
from rex.scheduler import get_scheduler

logger = get_logger(__name__)

# Blueprint for dashboard routes
dashboard_bp = Blueprint(
    "dashboard",
    __name__,
    static_folder="static",
    template_folder="templates",
    url_prefix="",
)

# Track server start time for uptime calculation
_SERVER_START_TIME = time.time()

# In-memory chat history for v1 (simple implementation)
_CHAT_HISTORY: list[dict[str, Any]] = []
_CHAT_HISTORY_MAX = 100


# --- Helper Functions ---


def _get_dashboard_dir() -> Path:
    """Get the dashboard module directory."""
    return Path(__file__).parent


def _is_loopback_address(addr: str | None) -> bool:
    """Check if an address is a loopback address."""
    if not addr:
        return False
    if addr.startswith("::ffff:"):
        addr = addr.split("::ffff:", 1)[-1]
    return addr in {"127.0.0.1", "::1"}


def _get_session_from_request() -> Session | None:
    """Extract and validate session from request."""
    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return get_session_manager().validate_session(token)

    # Check X-Dashboard-Token header
    token = request.headers.get("X-Dashboard-Token")  # type: ignore[assignment]
    if token:
        return get_session_manager().validate_session(token)

    # Check cookie
    token = request.cookies.get("rex_dashboard_token")  # type: ignore[assignment]
    if token:
        return get_session_manager().validate_session(token)

    # Optional token query param for EventSource clients that cannot set headers.
    if request.path == "/api/notifications/stream":
        token = request.args.get("token")  # type: ignore[assignment]
        if token and _is_safe_query_token_request():
            return get_session_manager().validate_session(token)

    return None


def _is_safe_query_token_request() -> bool:
    """Limit query-token auth to secure contexts (HTTPS or localhost)."""
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    is_secure = request.is_secure or forwarded_proto.lower() == "https"
    if not is_secure and not _is_loopback_address(request.remote_addr):
        return False

    origin = request.headers.get("Origin")
    if not origin:
        return True
    return origin.rstrip("/") == request.host_url.rstrip("/")


def _allow_local_without_auth() -> bool:
    """Check if local access without auth is allowed."""
    return os.getenv("REX_DASHBOARD_ALLOW_LOCAL", "1") == "1"


def require_auth(f):
    """Decorator to require authentication for API endpoints."""

    @wraps(f)
    def decorated(*args, **kwargs):
        # Check if local access is allowed
        if _allow_local_without_auth() and _is_loopback_address(request.remote_addr):
            if not is_password_required():
                # No password configured and local access - allow
                request.dashboard_session = Session(  # type: ignore[attr-defined]
                    token="local",
                    created_at=datetime.now(),
                    expires_at=datetime.now(),
                    user_key=os.getenv("REX_ACTIVE_USER", "local"),
                    metadata={"local_bypass": True},
                )
                return f(*args, **kwargs)

        # Check for valid session
        session = _get_session_from_request()
        if session is None:
            return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401

        # Attach session to request context for use in handlers
        request.dashboard_session = session  # type: ignore[attr-defined]
        return f(*args, **kwargs)

    return decorated


# Additional sensitive keys specific to settings display
_SETTINGS_SENSITIVE_PATTERNS = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "key",
    "credential",
    "auth",
    "private",
}


def _redact_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values from settings for display.

    Uses the contracts redaction plus additional patterns.
    """
    # First pass: use contracts redaction
    redacted = redact_sensitive_keys(config)

    # Second pass: additional patterns for nested keys
    def deep_redact(obj: Any, key: str = "") -> Any:
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                key_lower = k.lower()
                if any(pattern in key_lower for pattern in _SETTINGS_SENSITIVE_PATTERNS):
                    result[k] = "[REDACTED]" if v else None
                else:
                    result[k] = deep_redact(v, k)
            return result
        if isinstance(obj, list):
            return [deep_redact(item, key) for item in obj]
        return obj

    return deep_redact(redacted)  # type: ignore[no-any-return]


# Settings that require restart when changed
_RESTART_REQUIRED_SETTINGS = {
    "models.llm_provider",
    "models.llm_model",
    "models.stt_model",
    "models.tts_provider",
    "audio.input_device_index",
    "audio.output_device_index",
    "wake_word.backend",
    "wake_word.wakeword",
    "api.rate_limit",
}


def _get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get value from nested dict using dot notation path."""
    keys = path.split(".")
    value: Any = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    """Set value in nested dict using dot notation path."""
    keys = path.split(".")
    current: Any = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


_MISSING = object()


def _has_nested(data: dict[str, Any], path: str) -> bool:
    """Check if a nested path exists in a dict."""
    keys = path.split(".")
    value: Any = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return False
    return True


def _is_valid_setting_value(expected: Any, value: Any) -> bool:
    """Validate value types against expected config."""
    if value is None:
        return True
    if expected is None:
        return True
    if isinstance(expected, bool):
        return isinstance(value, bool)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(expected, str):
        return isinstance(value, str)
    if isinstance(expected, list):
        return isinstance(value, list)
    if isinstance(expected, dict):
        return isinstance(value, dict)
    return False


# --- Dashboard UI Routes ---


@dashboard_bp.route("/dashboard")
def dashboard_ui():
    """Serve the main dashboard UI."""
    template_path = _get_dashboard_dir() / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "Dashboard template not found", 404


@dashboard_bp.route("/dashboard/notifications")
def notifications_ui():
    """Serve the notification inbox UI.

    Returns the same SPA as the main dashboard. The JavaScript layer
    detects the /dashboard/notifications path on load and activates
    the Notifications section.
    """
    template_path = _get_dashboard_dir() / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "Dashboard template not found", 404


@dashboard_bp.route("/dashboard/assets/<path:filename>")
def dashboard_assets(filename: str):
    """Serve static assets (CSS/JS)."""
    static_dir = _get_dashboard_dir() / "static"
    return send_from_directory(str(static_dir), filename)


# --- Status Endpoint ---


@dashboard_bp.route("/api/dashboard/status")
def dashboard_status():
    """Return dashboard status information.

    This endpoint is public to allow health checks.
    """
    try:
        config = load_config()
        version = config.get("version", "1.0.0")
    except Exception:
        version = "1.0.0"

    uptime_seconds = int(time.time() - _SERVER_START_TIME)

    return jsonify(
        {
            "version": version,
            "uptime_seconds": uptime_seconds,
            "auth_enabled": is_password_required(),
            "server_time": datetime.now().isoformat(),
            "status": "ok",
        }
    )


# --- Authentication Endpoints ---


@dashboard_bp.route("/api/dashboard/login", methods=["POST"])
def dashboard_login():
    """Authenticate and create a session.

    Request body: {"password": "..."}
    Response: {"token": "...", "expires_at": "..."}
    """
    data = request.get_json(silent=True) or {}
    password = data.get("password", "")

    # Check if password is required
    if not is_password_required():
        # No password configured - check if local access is allowed
        if _allow_local_without_auth() and _is_loopback_address(request.remote_addr):
            session = get_session_manager().create_session(
                user_key=os.getenv("REX_ACTIVE_USER", "local")
            )
            response = jsonify(
                {
                    "token": session.token,
                    "expires_at": session.expires_at.isoformat(),
                    "message": "Logged in (local access)",
                }
            )
            response.set_cookie(
                "rex_dashboard_token",
                session.token,
                httponly=True,
                samesite="Strict",
                max_age=int((session.expires_at - session.created_at).total_seconds()),
            )
            return response
        return jsonify({"error": "Password not configured and remote access denied"}), 403

    # Verify password
    if not verify_password(password):
        logger.warning("Failed dashboard login attempt from %s", request.remote_addr)
        return jsonify({"error": "Invalid password"}), 401

    # Create session
    session = get_session_manager().create_session(
        user_key=os.getenv("REX_ACTIVE_USER", "dashboard")
    )
    logger.info("Dashboard login successful from %s", request.remote_addr)

    response = jsonify(
        {
            "token": session.token,
            "expires_at": session.expires_at.isoformat(),
        }
    )

    response.set_cookie(
        "rex_dashboard_token",
        session.token,
        httponly=True,
        samesite="Strict",
        max_age=int((session.expires_at - session.created_at).total_seconds()),
    )

    return response


@dashboard_bp.route("/api/dashboard/logout", methods=["POST"])
def dashboard_logout():
    """Invalidate the current session."""
    session = _get_session_from_request()
    if session:
        get_session_manager().invalidate_session(session.token)
        logger.info("Dashboard logout from %s", request.remote_addr)

    response = jsonify({"message": "Logged out"})
    response.delete_cookie("rex_dashboard_token")
    return response


# --- Settings Endpoints ---


@dashboard_bp.route("/api/settings", methods=["GET"])
@require_auth
def get_settings():
    """Get effective configuration (with sensitive values redacted)."""
    try:
        config = load_config()
        redacted = _redact_settings(config)

        settings_meta: dict[str, Any] = {}
        for path in _RESTART_REQUIRED_SETTINGS:
            settings_meta[path] = {"restart_required": True}

        return jsonify(
            {
                "settings": redacted,
                "defaults": DEFAULT_CONFIG,
                "metadata": settings_meta,
            }
        )
    except Exception as e:
        logger.error("Failed to load settings: %s", e)
        return jsonify({"error": f"Failed to load settings: {e}"}), 500


@dashboard_bp.route("/api/settings", methods=["PATCH"])
@require_auth
def update_settings():
    """Update configuration keys.

    Request body: {"key.path": value, ...}
    """
    data = request.get_json(silent=True) or {}

    if not data:
        return jsonify({"error": "No updates provided"}), 400

    try:
        config = load_config()
        updated_keys: list[str] = []
        restart_required = False
        invalid_updates: dict[str, str] = {}

        for key_path, value in data.items():
            if not key_path or ".." in key_path:
                invalid_updates[str(key_path)] = "Invalid key path"
                continue

            if not _has_nested(DEFAULT_CONFIG, key_path):
                invalid_updates[key_path] = "Unknown setting key"
                continue

            expected_value = _get_nested(DEFAULT_CONFIG, key_path, _MISSING)
            if expected_value is _MISSING:
                invalid_updates[key_path] = "Unknown setting key"
                continue

            if not _is_valid_setting_value(expected_value, value):
                invalid_updates[key_path] = (
                    f"Invalid value type (expected {type(expected_value).__name__})"
                )
                continue

            key_lower = key_path.lower()
            if any(pattern in key_lower for pattern in _SETTINGS_SENSITIVE_PATTERNS):
                logger.warning("Sensitive setting updated via API: %s", key_path)

            if key_path in _RESTART_REQUIRED_SETTINGS:
                restart_required = True

            _set_nested(config, key_path, value)
            updated_keys.append(key_path)

        if invalid_updates:
            return (
                jsonify(
                    {
                        "error": "Invalid settings update",
                        "invalid": invalid_updates,
                    }
                ),
                400,
            )

        if updated_keys:
            save_config(config)
            logger.info("Settings updated: %s", updated_keys)

        return jsonify(
            {
                "updated": updated_keys,
                "restart_required": restart_required,
            }
        )

    except Exception as e:
        logger.error("Failed to update settings: %s", e)
        return jsonify({"error": f"Failed to update settings: {e}"}), 500


# --- Chat Endpoints ---


def _get_llm():
    """Get or create the LLM client."""
    from rex.llm_client import LanguageModel

    return LanguageModel()


@dashboard_bp.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    """Send a chat message and get a reply.

    Request body: {"message": "..."}
    Response: {"reply": "...", "timestamp": "..."}
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        llm = _get_llm()

        messages: list[dict[str, str]] = []
        for entry in _CHAT_HISTORY[-10:]:
            messages.append({"role": "user", "content": entry.get("user_message", "")})
            if entry.get("assistant_reply"):
                messages.append({"role": "assistant", "content": entry["assistant_reply"]})

        messages.append({"role": "user", "content": message})

        start_time = time.time()
        reply = llm.generate(messages=messages)
        elapsed_ms = int((time.time() - start_time) * 1000)

        entry = {
            "user_message": message,
            "assistant_reply": reply,
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": elapsed_ms,
        }
        _CHAT_HISTORY.append(entry)

        while len(_CHAT_HISTORY) > _CHAT_HISTORY_MAX:
            _CHAT_HISTORY.pop(0)

        return jsonify(
            {
                "reply": reply,
                "timestamp": entry["timestamp"],
                "elapsed_ms": elapsed_ms,
            }
        )

    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return jsonify({"error": f"Failed to generate reply: {e}"}), 500


@dashboard_bp.route("/api/chat/history", methods=["GET"])
@require_auth
def chat_history():
    """Get chat history.

    Query params:
    - limit: Max entries to return (default: 50)
    - offset: Number of entries to skip (default: 0)
    """
    limit = min(int(request.args.get("limit", 50)), 100)
    offset = int(request.args.get("offset", 0))

    history = _CHAT_HISTORY[offset : offset + limit]

    return jsonify(
        {
            "history": history,
            "total": len(_CHAT_HISTORY),
            "limit": limit,
            "offset": offset,
        }
    )


# --- Scheduler/Reminders Endpoints ---


@dashboard_bp.route("/api/scheduler/jobs", methods=["GET"])
@require_auth
def list_jobs():
    """List all scheduled jobs."""
    try:
        scheduler = get_scheduler()
        jobs = scheduler.list_jobs()

        jobs_data: list[dict[str, Any]] = []
        for job in jobs:
            jobs_data.append(
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "schedule": job.schedule,
                    "enabled": job.enabled,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                    "last_run_at": job.last_run_at.isoformat() if job.last_run_at else None,
                    "run_count": job.run_count,
                    "max_runs": job.max_runs,
                    "callback_name": job.callback_name,
                    "workflow_id": job.workflow_id,
                    "metadata": job.metadata,
                }
            )

        return jsonify(
            {
                "jobs": jobs_data,
                "total": len(jobs_data),
                "metrics": scheduler.get_metrics(),
            }
        )

    except Exception as e:
        logger.error("Failed to list jobs: %s", e)
        return jsonify({"error": f"Failed to list jobs: {e}"}), 500


@dashboard_bp.route("/api/scheduler/jobs", methods=["POST"])
@require_auth
def create_job():
    """Create a new scheduled job."""
    data = request.get_json(silent=True) or {}

    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Job name is required"}), 400

    schedule = data.get("schedule", "interval:3600")
    enabled = data.get("enabled", True)
    callback_name = data.get("callback_name")
    workflow_id = data.get("workflow_id")
    metadata = data.get("metadata", {})

    if not schedule.startswith(("interval:", "at:")):
        return (
            jsonify({"error": "Invalid schedule format. Use 'interval:SECONDS' or 'at:HH:MM'"}),
            400,
        )

    try:
        scheduler = get_scheduler()
        job = scheduler.add_job(
            name=name,
            schedule=schedule,
            enabled=enabled,
            callback_name=callback_name,
            workflow_id=workflow_id,
            metadata=metadata,
        )

        logger.info("Created job: %s (%s)", job.name, job.job_id)

        return (
            jsonify(
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "schedule": job.schedule,
                    "enabled": job.enabled,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                }
            ),
            201,
        )

    except Exception as e:
        logger.error("Failed to create job: %s", e)
        return jsonify({"error": f"Failed to create job: {e}"}), 500


@dashboard_bp.route("/api/scheduler/jobs/<job_id>", methods=["GET"])
@require_auth
def get_job(job_id: str):
    """Get a specific job by ID."""
    try:
        scheduler = get_scheduler()
        job = scheduler.get_job(job_id)

        if job is None:
            return jsonify({"error": "Job not found"}), 404

        return jsonify(
            {
                "job_id": job.job_id,
                "name": job.name,
                "schedule": job.schedule,
                "enabled": job.enabled,
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "last_run_at": job.last_run_at.isoformat() if job.last_run_at else None,
                "run_count": job.run_count,
                "max_runs": job.max_runs,
                "callback_name": job.callback_name,
                "workflow_id": job.workflow_id,
                "metadata": job.metadata,
            }
        )

    except Exception as e:
        logger.error("Failed to get job: %s", e)
        return jsonify({"error": f"Failed to get job: {e}"}), 500


@dashboard_bp.route("/api/scheduler/jobs/<job_id>/run", methods=["POST"])
@require_auth
def run_job(job_id: str):
    """Run a job immediately (manual trigger)."""
    try:
        scheduler = get_scheduler()
        job = scheduler.get_job(job_id)

        if job is None:
            return jsonify({"error": "Job not found"}), 404

        success = scheduler.run_job(job_id, manual=True)

        return jsonify(
            {
                "job_id": job_id,
                "success": success,
                "message": "Job executed" if success else "Job execution failed",
            }
        )

    except Exception as e:
        logger.error("Failed to run job: %s", e)
        return jsonify({"error": f"Failed to run job: {e}"}), 500


@dashboard_bp.route("/api/scheduler/jobs/<job_id>", methods=["PATCH"])
@require_auth
def update_job(job_id: str):
    """Update a job (enable/disable, change schedule, etc)."""
    data = request.get_json(silent=True) or {}

    if not data:
        return jsonify({"error": "No updates provided"}), 400

    try:
        scheduler = get_scheduler()
        job = scheduler.get_job(job_id)

        if job is None:
            return jsonify({"error": "Job not found"}), 404

        allowed_fields = {"enabled", "schedule", "name", "max_runs", "metadata"}
        updates = {k: v for k, v in data.items() if k in allowed_fields}

        if not updates:
            return jsonify({"error": "No valid updates provided"}), 400

        updated_job = scheduler.update_job(job_id, **updates)

        if updated_job is None:
            return jsonify({"error": "Failed to update job"}), 500

        logger.info("Updated job: %s", job_id)

        return jsonify(
            {
                "job_id": updated_job.job_id,
                "name": updated_job.name,
                "schedule": updated_job.schedule,
                "enabled": updated_job.enabled,
                "next_run": updated_job.next_run.isoformat() if updated_job.next_run else None,
            }
        )

    except Exception as e:
        logger.error("Failed to update job: %s", e)
        return jsonify({"error": f"Failed to update job: {e}"}), 500


@dashboard_bp.route("/api/scheduler/jobs/<job_id>", methods=["DELETE"])
@require_auth
def delete_job(job_id: str):
    """Delete a scheduled job."""
    try:
        scheduler = get_scheduler()
        existed = scheduler.remove_job(job_id)

        if not existed:
            return jsonify({"error": "Job not found"}), 404

        logger.info("Deleted job: %s", job_id)

        return jsonify(
            {
                "job_id": job_id,
                "deleted": True,
            }
        )

    except Exception as e:
        logger.error("Failed to delete job: %s", e)
        return jsonify({"error": f"Failed to delete job: {e}"}), 500


# --- Notification Endpoints ---


@dashboard_bp.route("/api/notifications", methods=["GET"])
@require_auth
def list_notifications():
    """List recent dashboard notifications."""
    try:
        from rex.dashboard_store import get_dashboard_store

        store = get_dashboard_store()

        limit = min(int(request.args.get("limit", 50)), 200)
        unread_only = request.args.get("unread", "").lower() == "true"
        priority = request.args.get("priority")
        session = request.dashboard_session  # type: ignore[attr-defined]
        user_id = request.args.get("user_id") or session.user_key
        if user_id != session.user_key:
            return jsonify({"error": "Forbidden user scope"}), 403

        notifications = store.query_recent(
            limit=limit,
            user_id=user_id,
            unread_only=unread_only,
            priority=priority,
        )
        unread_count = store.count_unread(user_id=user_id)

        return jsonify(
            {
                "notifications": [n.to_dict() for n in notifications],
                "total": len(notifications),
                "unread_count": unread_count,
            }
        )

    except Exception as e:
        logger.error("Failed to list notifications: %s", e)
        return jsonify({"error": f"Failed to list notifications: {e}"}), 500


@dashboard_bp.route("/api/notifications/<notification_id>/read", methods=["POST"])
@require_auth
def mark_notification_read(notification_id: str):
    """Mark a notification as read."""
    try:
        from rex.dashboard_store import get_dashboard_store

        store = get_dashboard_store()
        session = request.dashboard_session  # type: ignore[attr-defined]
        found = store.mark_as_read(notification_id, user_id=session.user_key)

        if not found:
            return jsonify({"error": "Notification not found or already read"}), 404

        return jsonify({"id": notification_id, "read": True})

    except Exception as e:
        logger.error("Failed to mark notification read: %s", e)
        return jsonify({"error": f"Failed to mark notification read: {e}"}), 500


@dashboard_bp.route("/api/notifications/read-all", methods=["POST"])
@require_auth
def mark_all_notifications_read():
    """Mark all notifications as read."""
    try:
        from rex.dashboard_store import get_dashboard_store

        store = get_dashboard_store()
        session = request.dashboard_session  # type: ignore[attr-defined]
        user_id = request.args.get("user_id") or session.user_key
        if user_id != session.user_key:
            return jsonify({"error": "Forbidden user scope"}), 403
        count = store.mark_all_read(user_id=user_id)

        return jsonify({"marked_read": count})

    except Exception as e:
        logger.error("Failed to mark all notifications read: %s", e)
        return jsonify({"error": f"Failed to mark all notifications read: {e}"}), 500


@dashboard_bp.route("/api/notifications/stream", methods=["GET"])
@require_auth
def stream_notifications():
    """Stream notification events via Server-Sent Events (SSE)."""
    from rex.dashboard.sse import get_broadcaster
    from rex.dashboard_store import get_dashboard_store

    session = request.dashboard_session  # type: ignore[attr-defined]
    store = get_dashboard_store()
    broadcaster = get_broadcaster()

    max_events = 100
    timeout = 15.0
    if current_app.testing:
        max_events = max(1, min(int(request.args.get("max_events", 100)), 1000))
        timeout = max(0.01, min(float(request.args.get("timeout", 15.0)), 60.0))

    subscriber = broadcaster.subscribe(max_events=max_events)

    def generate():
        try:
            init_payload = {"unread_count": store.count_unread(user_id=session.user_key)}
            yield f"event: init\ndata: {json.dumps(init_payload)}\n\n"

            for chunk in broadcaster.stream(subscriber, timeout=timeout):
                if "data:" not in chunk:
                    yield chunk
                    continue

                line = next((ln for ln in chunk.splitlines() if ln.startswith("data: ")), None)
                if line is None:
                    continue

                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_user_id = payload.get("user_id")
                if event_user_id is None:
                    continue
                if event_user_id != session.user_key:
                    continue

                yield chunk
        finally:
            unsubscribe = getattr(broadcaster, "unsubscribe", None)
            if callable(unsubscribe):
                try:
                    unsubscribe(subscriber)
                except Exception:
                    pass

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


# --- Voice Endpoints ---


def _transcribe_audio_file(audio_path: str) -> str:
    """Transcribe an audio file using Whisper.

    Separated as a standalone function so tests can patch it.
    """
    try:
        import whisper as _whisper
    except ImportError as exc:
        raise ImportError("openai-whisper is required for voice transcription") from exc

    model = _whisper.load_model("base")
    result = model.transcribe(audio_path, language="en", fp16=False)
    return str(result.get("text", "")).strip()


@dashboard_bp.route("/api/voice", methods=["POST"])
@require_auth
def voice_chat():
    """Accept audio from the browser microphone, transcribe, and return an LLM reply.

    Accepts multipart/form-data with an 'audio' field containing the recorded blob.
    Returns {"transcript": "...", "reply": "...", "timestamp": "..."}.
    """
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "No audio file provided"}), 400

    # Pick a temp file extension from the content-type so ffmpeg/whisper can load it.
    content_type = audio_file.content_type or ""
    if "webm" in content_type:
        suffix = ".webm"
    elif "ogg" in content_type:
        suffix = ".ogg"
    elif "mp4" in content_type or "m4a" in content_type:
        suffix = ".mp4"
    else:
        suffix = ".wav"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp)

        # Transcribe
        try:
            transcript = _transcribe_audio_file(tmp_path)
        except ImportError:
            return jsonify({"error": "Speech-to-text not available"}), 503
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return jsonify({"error": f"Transcription failed: {exc}"}), 500

        if not transcript or not transcript.strip():
            return jsonify({"error": "No speech detected"}), 422

        # Generate LLM reply
        try:
            llm = _get_llm()
            messages: list[dict[str, str]] = [{"role": "user", "content": transcript}]
            reply = llm.generate(messages=messages)
        except Exception as exc:
            logger.error("LLM error during voice chat: %s", exc)
            return jsonify({"error": f"Failed to generate reply: {exc}"}), 500

        # Persist to shared chat history
        entry: dict[str, Any] = {
            "user_message": transcript,
            "assistant_reply": reply,
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": 0,
            "source": "voice",
        }
        _CHAT_HISTORY.append(entry)
        while len(_CHAT_HISTORY) > _CHAT_HISTORY_MAX:
            _CHAT_HISTORY.pop(0)

        return jsonify(
            {
                "transcript": transcript,
                "reply": reply,
                "timestamp": entry["timestamp"],
            }
        )

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


__all__ = ["dashboard_bp"]
