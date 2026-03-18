"""Tests for US-173: Voice pipeline state announcements via GUI status."""

from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
TEMPLATE_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "templates"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
JS_FILE = STATIC_DIR / "js" / "dashboard.js"
HTML_FILE = TEMPLATE_DIR / "index.html"


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _js() -> str:
    return JS_FILE.read_text(encoding="utf-8")


def _html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


def _non_local_env():
    return {"REMOTE_ADDR": "10.0.0.1"}


@pytest.fixture()
def app():
    from flask import Flask

    from rex.dashboard import dashboard_bp

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-173"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-173")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-173"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ── HTML structure ─────────────────────────────────────────────────────────────


class TestHTMLVoiceStateUI:
    def test_voice_state_label_exists(self):
        assert 'id="voice-state-label"' in _html()

    def test_voice_waveform_exists(self):
        assert 'id="voice-waveform"' in _html()

    def test_voice_section_exists(self):
        assert 'id="voice-section"' in _html()


# ── CSS state styling ─────────────────────────────────────────────────────────


class TestCSSVoiceStateLabels:
    def test_voice_state_label_base_class(self):
        assert ".voice-state-label" in _css()

    def test_voice_state_listening_class(self):
        assert ".voice-state-listening" in _css()

    def test_voice_state_thinking_class(self):
        assert ".voice-state-thinking" in _css()

    def test_voice_state_speaking_class(self):
        assert ".voice-state-speaking" in _css()

    def test_thinking_state_has_italic(self):
        css = _css()
        idx = css.index(".voice-state-thinking {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "italic" in block

    def test_speaking_state_has_distinct_color(self):
        css = _css()
        idx = css.index(".voice-state-speaking {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "color" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSVoiceStateUI:
    def _get_update_fn_body(self) -> str:
        js = _js()
        idx = js.index("function _updateVoiceModeUI")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    return js[brace_start : i + 1]
            i += 1
        return js[brace_start:]

    def test_update_fn_adds_thinking_class(self):
        body = self._get_update_fn_body()
        assert "voice-state-thinking" in body

    def test_update_fn_adds_speaking_class(self):
        body = self._get_update_fn_body()
        assert "voice-state-speaking" in body

    def test_update_fn_adds_listening_class(self):
        body = self._get_update_fn_body()
        assert "voice-state-listening" in body

    def test_update_fn_removes_old_state_classes(self):
        body = self._get_update_fn_body()
        assert "classList.remove" in body

    def test_update_fn_shows_waveform_for_speaking(self):
        body = self._get_update_fn_body()
        assert "Speaking" in body

    def test_sse_stream_fn_exists(self):
        assert "startVoiceStateStream" in _js()

    def test_sse_stream_uses_event_source(self):
        js = _js()
        idx = js.index("function startVoiceStateStream")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    body = js[brace_start : i + 1]
                    break
            i += 1
        assert "EventSource" in body

    def test_sse_stream_listens_for_state_event(self):
        js = _js()
        idx = js.index("function startVoiceStateStream")
        body = js[idx : idx + 400]
        assert "'state'" in body or '"state"' in body


# ── Backend endpoint ───────────────────────────────────────────────────────────


class TestVoiceStateEndpoint:
    def _patch_state(self, client, token, state):
        return client.patch(
            "/api/voice/state",
            json={"state": state},
            headers={"X-Auth-Token": token},
            environ_base=_non_local_env(),
        )

    def test_patch_voice_state_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.patch(
            "/api/voice/state",
            json={"state": "Thinking"},
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (401, 403)

    def test_patch_voice_state_thinking(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._patch_state(client, auth_token, "Thinking")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"] == "Thinking"
        assert data["active"] is True

    def test_patch_voice_state_speaking(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._patch_state(client, auth_token, "Speaking")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"] == "Speaking"
        assert data["active"] is True

    def test_patch_voice_state_idle(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._patch_state(client, auth_token, "Idle")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"] == "Idle"
        assert data["active"] is False

    def test_patch_voice_state_invalid_returns_400(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._patch_state(client, auth_token, "Confused")
        assert resp.status_code == 400

    def test_patch_voice_state_missing_field_returns_400(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.patch(
            "/api/voice/state",
            json={},
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        assert resp.status_code == 400

    def test_valid_states_include_all_pipeline_stages(self):
        from rex.dashboard.routes import _VALID_VOICE_STATES

        assert "Idle" in _VALID_VOICE_STATES
        assert "Listening" in _VALID_VOICE_STATES
        assert "Thinking" in _VALID_VOICE_STATES
        assert "Speaking" in _VALID_VOICE_STATES
