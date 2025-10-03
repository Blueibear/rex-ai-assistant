import importlib
import importlib.util
import json
import os

from flask import Flask, request, abort, jsonify, g
from flask_cors import CORS

from memory_utils import load_memory_profile, load_users_map, resolve_user_key

app = Flask(__name__)
CORS(app)

# Optional: register search plugin if available
search_web = None
try:
    spec = importlib.util.find_spec("plugins.web_search")
    if spec:
        mod = importlib.import_module("plugins.web_search")
        search_web = getattr(mod, "search_web", None)
except Exception as e:
    app.logger.warning("Failed to load search plugin: %s", e)

USERS_MAP = load_users_map()
PROXY_TOKEN = os.getenv("REX_PROXY_TOKEN")
ALLOW_LOCAL = os.getenv("REX_PROXY_ALLOW_LOCAL") == "1"
user_key: str | None = None


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
            "topics": [t for t in prefs.get("topics", []) if isinstance(t, str)]
        }

    return summary


def _extract_shared_secret() -> str | None:
    """Return the shared-secret token provided with the request, if any."""
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


@app.before_request
def load_user_memory():
    email = request.headers.get("Cf-Access-Authenticated-User-Email")
    token = _extract_shared_secret()
    remote_addr = request.remote_addr

    global user_key
    resolved_user_key: str | None = None

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

    user_key = resolved_user_key
    memory_path = os.path.join("Memory", user_key)
    try:
        memory = load_memory_profile(user_key)
    except FileNotFoundError:
        abort(500, f"Memory file not found for user '{user_key}'.")
    except json.JSONDecodeError:
        abort(500, f"Memory file for user '{user_key}' is invalid JSON.")

    g.user_key = user_key
    g.memory = memory
    g.user_folder = memory_path


@app.route("/")
def index():
    return "🧠 Rex is online. Ask away."


@app.route("/whoami")
def whoami():
    """Return a redacted summary of the active user profile."""
    return jsonify({
        "user": g.user_key,
        "profile": _summarize_memory(g.memory),
    })


@app.route("/search")
def search():
    if not search_web:
        return jsonify({"error": "Search plugin is not available"}), 503

    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        result = search_web(query)
    except Exception as e:
        app.logger.warning("Search failed: %s", e)
        result = "Search failed due to internal error."

    return jsonify({
        "query": query,
        "result": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

