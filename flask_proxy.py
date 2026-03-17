import importlib
import importlib.util
import json
import os
from types import SimpleNamespace

from flask import Flask, abort, jsonify, request
from flask import g as flask_g
from flask_cors import CORS

import utils.env_loader  # noqa: F401  # Auto-loads .env on import
from memory_utils import load_memory_profile, load_users_map, resolve_user_key
from rex.health import check_config, create_health_blueprint
from rex.http_errors import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    SERVICE_UNAVAILABLE,
    error_response,
    install_error_envelope_handler,
)
from rex.rate_limiter import install_rate_limiter
from rex.request_logging import install_request_logging
from rex.startup import log_service_ready, run_startup_sequence

# Import dashboard blueprint
try:
    from rex.dashboard import dashboard_bp

    _DASHBOARD_AVAILABLE = True
except ImportError:
    dashboard_bp = None
    _DASHBOARD_AVAILABLE = False

# Import contract version for /contracts endpoint
try:
    from rex.contracts import CONTRACT_VERSION
    from rex.contracts.core import ALL_MODELS

    _CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACT_VERSION = None
    ALL_MODELS = []
    _CONTRACTS_AVAILABLE = False

_TESTING_MODE = os.getenv("REX_TESTING", "").lower() in {"1", "true", "yes"}
if _TESTING_MODE:
    g = SimpleNamespace()
else:
    g = flask_g

# --- Startup sequence — config → database → migration → service init ---
if not _TESTING_MODE:
    run_startup_sequence()

# --- Flask Setup ---
app = Flask(__name__)
install_request_logging(app)
install_error_envelope_handler(app)
install_rate_limiter(app)
app.register_blueprint(create_health_blueprint(checks=[check_config]))

# CORS Configuration: Restrict origins based on environment
# Default to localhost for development; override via REX_ALLOWED_ORIGINS env var
ALLOWED_ORIGINS = os.getenv(
    "REX_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,http://127.0.0.1:3000,http://127.0.0.1:5000",
)
_CORS_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]

CORS(
    app,
    resources={
        r"/*": {
            "origins": _CORS_ORIGINS,
            "methods": ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "X-Rex-Proxy-Token",
                "X-Dashboard-Token",
                "Cf-Access-Authenticated-User-Email",
            ],
            "supports_credentials": True,
            "max_age": 600,
        }
    },
)

# Register dashboard blueprint if available
if _DASHBOARD_AVAILABLE and dashboard_bp is not None:
    app.register_blueprint(dashboard_bp)

# Register inbound SMS webhook blueprint (config-driven; disabled by default)
try:
    from rex.messaging_backends.webhook_wiring import register_inbound_sms_webhook

    _INBOUND_SMS_REGISTERED = register_inbound_sms_webhook(app)
except Exception as _inbound_exc:
    app.logger.debug("Inbound SMS webhook wiring skipped: %s", _inbound_exc)
    _INBOUND_SMS_REGISTERED = False

# Wire retention cleanup scheduler jobs (config-driven; safe to skip on failure)
try:
    from rex.retention import wire_retention_cleanup

    _RETENTION_CLEANUP_WIRED = wire_retention_cleanup()
except Exception as _retention_exc:
    app.logger.debug("Retention cleanup wiring skipped: %s", _retention_exc)
    _RETENTION_CLEANUP_WIRED = False

# --- Optional Plugin: Web Search ---
search_web = None
try:
    spec = importlib.util.find_spec("plugins.web_search")
    if spec:
        mod = importlib.import_module("plugins.web_search")
        search_web = getattr(mod, "search_web", None)
except Exception as e:
    app.logger.warning("Failed to load search plugin: %s", e)

# --- Constants ---
USERS_MAP = load_users_map()
PROXY_TOKEN = os.getenv("REX_PROXY_TOKEN")
ALLOW_LOCAL = os.getenv("REX_PROXY_ALLOW_LOCAL") == "1"

if not _TESTING_MODE:
    log_service_ready()


# --- Helpers ---
def _summarize_memory(profile: dict) -> dict:
    summary = {}
    if not isinstance(profile, dict):
        return summary

    if isinstance(name := profile.get("name"), str):
        summary["name"] = name

    if isinstance(role := profile.get("role"), str):
        summary["role"] = role

    prefs = profile.get("preferences")
    if isinstance(prefs, dict):
        summary["preferences"] = {
            "tone": prefs.get("tone"),
            "topics": [t for t in prefs.get("topics", []) if isinstance(t, str)],
        }

    return summary


def _extract_shared_secret() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None
    return request.headers.get("X-Rex-Proxy-Token")


def _is_loopback_address(addr: str | None) -> bool:
    if not addr:
        return False
    if addr.startswith("::ffff:"):
        addr = addr.split("::ffff:", 1)[-1]
    return addr in {"127.0.0.1", "::1"}


# --- Public Endpoints (no auth required) ---
# Note: Dashboard routes have their own auth mechanism
_PUBLIC_PATHS = frozenset({"/contracts"})
_DASHBOARD_PREFIXES = (
    "/dashboard",
    "/api/dashboard",
    "/api/settings",
    "/api/chat",
    "/api/scheduler",
    "/api/notifications",
    "/api/voice",
)
# Webhook routes have their own authentication (e.g. Twilio signature)
_WEBHOOK_PREFIXES = ("/webhooks/",)


# --- Request Hooks ---
@app.before_request
def load_user_memory():
    # Skip auth for public endpoints
    if request.path in _PUBLIC_PATHS:
        return

    # Skip main auth for dashboard routes (they have their own auth)
    if any(request.path.startswith(prefix) for prefix in _DASHBOARD_PREFIXES):
        return

    # Skip main auth for webhook routes (they use their own auth, e.g. Twilio signature)
    if any(request.path.startswith(prefix) for prefix in _WEBHOOK_PREFIXES):
        return

    if _TESTING_MODE:
        g.__dict__.clear()

    email = request.headers.get("Cf-Access-Authenticated-User-Email")
    token = _extract_shared_secret()
    remote_addr = request.remote_addr

    resolved_user_key = None

    if email:
        resolved_user_key = resolve_user_key(email, USERS_MAP)
        if not resolved_user_key:
            abort(403, "Access denied for this user.")
    elif token:
        if not PROXY_TOKEN or token != PROXY_TOKEN:
            abort(403, "Invalid or missing proxy token.")
        resolved_user_key = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP)
        if not resolved_user_key:
            abort(403, "REX_ACTIVE_USER is not configured or invalid.")
    elif ALLOW_LOCAL and _is_loopback_address(remote_addr):
        resolved_user_key = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP)
        if not resolved_user_key:
            abort(403, "Local access not permitted: REX_ACTIVE_USER missing.")
    else:
        abort(403, "No authenticated identity provided.")

    g.user_key = resolved_user_key
    g.user_folder = os.path.join("Memory", g.user_key)

    try:
        g.memory = load_memory_profile(g.user_key)
    except FileNotFoundError:
        abort(500, f"Memory file not found for user '{g.user_key}'.")
    except json.JSONDecodeError:
        abort(500, f"Memory file for user '{g.user_key}' is invalid JSON.")


# --- Routes ---
@app.route("/")
def index():
    return "🧠 Rex is online. Ask away."


@app.route("/whoami")
def whoami():
    return jsonify(
        {
            "user": g.user_key,
            "profile": _summarize_memory(g.memory),
        }
    )


@app.route("/search")
def search():
    if not search_web:
        return error_response(SERVICE_UNAVAILABLE, "Web search plugin is not installed.", 503)

    query = request.args.get("q")
    if not query:
        return error_response(BAD_REQUEST, "Missing query", 400)

    try:
        result = search_web(query)
    except Exception as e:
        app.logger.exception("Web search failed")
        return error_response(INTERNAL_ERROR, f"Search provider error: {e}", 502)

    return jsonify({"query": query, "result": result})


@app.route("/contracts")
def contracts():
    """Return contract schema metadata for discoverability."""
    if not _CONTRACTS_AVAILABLE:
        return error_response(SERVICE_UNAVAILABLE, "Contracts module not available", 503)

    model_names = [model.__name__ for model in ALL_MODELS]
    return jsonify(
        {
            "contract_version": CONTRACT_VERSION,
            "schema_docs_path": "docs/contracts/",
            "models": model_names,
        }
    )


# --- Entry Point ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
