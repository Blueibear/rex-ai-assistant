import os
import json
from flask import Flask, request, abort, jsonify

from memory_utils import load_memory_profile, load_users_map, resolve_user_key
from plugins.web_search import search_web

app = Flask(__name__)

USERS_MAP = load_users_map()
PROXY_TOKEN = os.getenv("REX_PROXY_TOKEN")
ALLOW_LOCAL = os.getenv("REX_PROXY_ALLOW_LOCAL") == "1"


def _extract_shared_secret() -> str | None:
    """Return the shared-secret token provided with the request, if any."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    header_token = request.headers.get("X-Rex-Proxy-Token")
    if header_token:
        return header_token.strip()

    return None


def _is_loopback_address(address: str | None) -> bool:
    """Return ``True`` when the request originates from the local machine."""
    if not address:
        return False

    if address.startswith("::ffff:"):
        address = address.split("::ffff:", 1)[1]

    return address in {"127.0.0.1", "::1"}


@app.before_request
def load_user_memory():
    global user_key, memory, user_folder

    email = request.headers.get("Cf-Access-Authenticated-User-Email")
    shared_secret = _extract_shared_secret()
    remote_addr = request.remote_addr

    if email:
        user_key_candidate = resolve_user_key(email, USERS_MAP)
        if not user_key_candidate:
            abort(403, description="Access denied for this user.")
    elif shared_secret:
        if not PROXY_TOKEN:
            abort(403, description="Shared secret authentication is not configured.")
        if shared_secret != PROXY_TOKEN:
            abort(403, description="Invalid proxy token.")

        configured_user = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP)
        if not configured_user:
            abort(403, description="REX_ACTIVE_USER is not configured or invalid.")
        user_key_candidate = configured_user
    elif ALLOW_LOCAL and _is_loopback_address(remote_addr):
        configured_user = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP)
        if not configured_user:
            abort(403, description="Local access requested but REX_ACTIVE_USER is not configured or invalid.")
        user_key_candidate = configured_user
    else:
        abort(403, description="No authenticated identity provided.")

    user_key = user_key_candidate
    user_folder = os.path.join("Memory", user_key)

    try:
        memory = load_memory_profile(user_key)
    except FileNotFoundError:
        abort(500, description=f"Memory file not found for user '{user_key}'.")
    except json.JSONDecodeError:
        abort(500, description=f"Memory file for user '{user_key}' is invalid JSON.")


# ✅ Root route: status check
@app.route("/")
def index():
    return "🧠 Rex is online. Ask away."


# ✅ Whoami route: returns active memory profile
@app.route("/whoami")
def whoami():
    return jsonify({
        "user": user_key,
        "memory": memory
    })


# ✅ Search route: performs live web search
@app.route("/search")
def search():
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    result = search_web(query)
    return jsonify({
        "query": query,
        "result": result
    })


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
