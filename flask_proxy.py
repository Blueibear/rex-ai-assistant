"""Authenticated Flask proxy for Rex memory and search."""

from __future__ import annotations

import json
import os
from contextlib import suppress

from flask import Flask, abort, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from rex import settings
from rex.logging_utils import configure_logger
from rex.memory import load_memory_profile, load_users_map, resolve_user_key
from rex.plugins.base import PluginContext
from rex.plugins.loader import load_plugins

LOGGER = configure_logger(__name__)

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[settings.flask_rate_limit])

USERS_MAP = load_users_map()
PROXY_TOKEN = os.getenv("REX_PROXY_TOKEN")
ALLOW_LOCAL = os.getenv("REX_PROXY_ALLOW_LOCAL") == "1"
PLUGINS = load_plugins()


def _summarize_memory(profile: dict) -> dict:
    summary: dict = {}
    if not isinstance(profile, dict):
        return summary
    for key in ("name", "role"):
        value = profile.get(key)
        if isinstance(value, str):
            summary[key] = value
    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        pref_summary: dict = {}
        tone = preferences.get("tone")
        if isinstance(tone, str):
            pref_summary["tone"] = tone
        topics = preferences.get("topics")
        if isinstance(topics, list):
            pref_summary["topics"] = [topic for topic in topics if isinstance(topic, str)]
        if pref_summary:
            summary["preferences"] = pref_summary
    return summary


def _extract_shared_secret() -> str | None:
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
    if not address:
        return False
    if address.startswith("::ffff:" ):
        address = address.split("::ffff:", 1)[1]
    return address in {"127.0.0.1", "::1"}


@app.before_request
def load_user_memory():
    global user_key, memory

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
        configured_user = resolve_user_key(settings.user_id, USERS_MAP)
        if not configured_user:
            abort(403, description="REX_ACTIVE_USER is not configured or invalid.")
        user_key_candidate = configured_user
    elif ALLOW_LOCAL and _is_loopback_address(remote_addr):
        configured_user = resolve_user_key(settings.user_id, USERS_MAP)
        if not configured_user:
            abort(403, description="Local access requested but user profile is invalid.")
        user_key_candidate = configured_user
    else:
        abort(403, description="No authenticated identity provided.")

    user_key = user_key_candidate
    try:
        memory = load_memory_profile(user_key)
    except FileNotFoundError:
        abort(500, description=f"Memory file not found for user '{user_key}'.")
    except json.JSONDecodeError:
        abort(500, description=f"Memory file for user '{user_key}' is invalid JSON.")


@app.route("/")
@limiter.limit(lambda: settings.flask_rate_limit)
def index():
    return "🧠 Rex is online. Ask away."


@app.route("/whoami")
@limiter.limit(lambda: settings.flask_rate_limit)
def whoami():
    return jsonify({"user": user_key, "profile": _summarize_memory(memory)})


@app.route("/search")
@limiter.limit("3 per minute")
def search():
    query = request.args.get("q")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing 'q' parameter"}), 400
    query = query.strip()
    if len(query) > 200:
        return jsonify({"error": "Query too long"}), 400

    plugin = PLUGINS.get("web_search")
    if not plugin:
        return jsonify({"error": "Web search plugin not available"}), 503

    context = PluginContext(user_id=user_key, text=query)
    with suppress(Exception):
        result = plugin.process(context)
        if result:
            return jsonify({"query": query, "result": result})
    return jsonify({"query": query, "result": "No result found"})


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
