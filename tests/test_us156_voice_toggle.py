"""Tests for US-156: Voice mode toggle with status indicator.

Acceptance criteria:
- Voice panel has a prominent button labeled "Start Listening" / "Stop Listening"
- button toggles voice mode on and off via the Rex backend
- current voice state (Idle, Listening, Processing, Speaking) displayed as text label
- state updates in real time without requiring a page refresh
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

REPO_ROOT = Path(__file__).parent.parent
HTML_PATH = REPO_ROOT / "rex" / "dashboard" / "templates" / "index.html"
JS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "js" / "dashboard.js"
CSS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "css" / "dashboard.css"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _non_local_env():
    return {"REMOTE_ADDR": "10.0.0.1"}


@pytest.fixture()
def app():
    from rex.dashboard import dashboard_bp

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-156"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-156"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ---------------------------------------------------------------------------
# HTML: Voice panel structure
# ---------------------------------------------------------------------------


class TestVoicePanelHTML:
    def test_voice_section_exists(self):
        html = _html()
        assert 'id="voice-section"' in html, "voice-section must exist"

    def test_voice_mode_button_exists(self):
        html = _html()
        assert 'id="voice-mode-btn"' in html, "voice-mode-btn button must exist"

    def test_voice_mode_button_start_label(self):
        html = _html()
        assert "Start Listening" in html, "button must have Start Listening label"

    def test_voice_state_label_exists(self):
        html = _html()
        assert 'id="voice-state-label"' in html, "voice-state-label element must exist"

    def test_voice_state_label_initial_value(self):
        html = _html()
        # The label should have an initial "Idle" value
        idx = html.index('id="voice-state-label"')
        snippet = html[idx: idx + 100]
        assert "Idle" in snippet, "voice-state-label initial value must be Idle"


# ---------------------------------------------------------------------------
# Backend: GET /api/voice/mode
# ---------------------------------------------------------------------------


class TestGetVoiceMode:
    def test_get_voice_mode_route_registered(self, app):
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/voice/mode" in rules

    def test_get_voice_mode_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "secret156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.get(
            "/api/voice/mode",
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (401, 403)

    def test_get_voice_mode_returns_state(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        # Reset state before reading
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = False
        r._VOICE_STATE["state"] = "Idle"

        resp = client.get(
            "/api/voice/mode",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "active" in data
        assert "state" in data

    def test_get_voice_mode_initial_idle(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = False
        r._VOICE_STATE["state"] = "Idle"

        resp = client.get(
            "/api/voice/mode",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        data = resp.get_json()
        assert data["state"] == "Idle"
        assert data["active"] is False


# ---------------------------------------------------------------------------
# Backend: POST /api/voice/mode
# ---------------------------------------------------------------------------


class TestSetVoiceMode:
    def _post(self, client, auth_token, monkeypatch, active: bool):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        return client.post(
            "/api/voice/mode",
            data=json.dumps({"active": active}),
            content_type="application/json",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )

    def test_set_active_true_returns_listening(self, client, auth_token, monkeypatch):
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = False
        r._VOICE_STATE["state"] = "Idle"

        resp = self._post(client, auth_token, monkeypatch, active=True)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is True
        assert data["state"] == "Listening"

    def test_set_active_false_returns_idle(self, client, auth_token, monkeypatch):
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = True
        r._VOICE_STATE["state"] = "Listening"

        resp = self._post(client, auth_token, monkeypatch, active=False)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is False
        assert data["state"] == "Idle"

    def test_set_missing_active_field_returns_400(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.post(
            "/api/voice/mode",
            data=json.dumps({}),
            content_type="application/json",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        assert resp.status_code == 400

    def test_toggle_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "secret156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.post(
            "/api/voice/mode",
            data=json.dumps({"active": True}),
            content_type="application/json",
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Backend: GET /api/voice/state/stream (SSE)
# ---------------------------------------------------------------------------


class TestVoiceStateStream:
    def test_stream_route_registered(self, app):
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/voice/state/stream" in rules

    def test_stream_returns_event_stream(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = False
        r._VOICE_STATE["state"] = "Idle"

        resp = client.get(
            "/api/voice/state/stream?timeout=0.05",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        assert "text/event-stream" in resp.content_type

    def test_stream_emits_initial_state_event(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = False
        r._VOICE_STATE["state"] = "Idle"

        resp = client.get(
            "/api/voice/state/stream?timeout=0.05",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        text = resp.data.decode("utf-8")
        assert "event: state" in text

    def test_stream_initial_event_has_state_field(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-156")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        from rex.dashboard import routes as r
        r._VOICE_STATE["active"] = False
        r._VOICE_STATE["state"] = "Idle"

        resp = client.get(
            "/api/voice/state/stream?timeout=0.05",
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        for line in resp.data.decode("utf-8").splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                assert "state" in payload
                assert "active" in payload
                break


# ---------------------------------------------------------------------------
# Frontend JS: voice toggle implementation
# ---------------------------------------------------------------------------


class TestFrontendVoiceJS:
    def test_toggle_voice_mode_function_exists(self):
        js = _js()
        assert "toggleVoiceMode" in js

    def test_voice_mode_btn_listener_registered(self):
        js = _js()
        assert "voice-mode-btn" in js and "toggleVoiceMode" in js

    def test_api_voice_mode_called(self):
        js = _js()
        assert "/api/voice/mode" in js

    def test_update_voice_mode_ui_function_exists(self):
        js = _js()
        assert "_updateVoiceModeUI" in js or "updateVoiceModeUI" in js.lower()

    def test_button_label_updated_on_active(self):
        js = _js()
        assert "Stop Listening" in js, "JS must set Stop Listening label when active"

    def test_state_label_updated(self):
        js = _js()
        assert "voice-state-label" in js, "JS must reference voice-state-label"

    def test_sse_stream_started_for_voice_section(self):
        js = _js()
        assert "startVoiceStateStream" in js

    def test_sse_stream_stopped_when_leaving_voice(self):
        js = _js()
        assert "stopVoiceStateStream" in js

    def test_voice_state_stream_endpoint(self):
        js = _js()
        assert "/api/voice/state/stream" in js

    def test_state_event_parsed(self):
        js = _js()
        assert "'state'" in js or '"state"' in js

    def test_active_flag_read_from_state_data(self):
        js = _js()
        fn_start = js.index("_updateVoiceModeUI")
        fn_body = js[fn_start: fn_start + 500]
        assert "active" in fn_body


# ---------------------------------------------------------------------------
# CSS: voice panel styles
# ---------------------------------------------------------------------------


class TestVoicePanelCSS:
    def test_voice_mode_btn_class_defined(self):
        css = _css()
        assert ".voice-mode-btn" in css

    def test_voice_state_label_class_defined(self):
        css = _css()
        assert ".voice-state-label" in css

    def test_voice_mode_panel_class_defined(self):
        css = _css()
        assert ".voice-mode-panel" in css


# ---------------------------------------------------------------------------
# Typecheck
# ---------------------------------------------------------------------------


class TestTypecheck:
    def test_mypy_routes(self):
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "rex/dashboard/routes.py", "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        errors = [
            ln for ln in result.stdout.splitlines()
            if "error:" in ln and "routes.py" in ln
        ]
        assert errors == [], "mypy errors in routes.py:\n" + "\n".join(errors)
