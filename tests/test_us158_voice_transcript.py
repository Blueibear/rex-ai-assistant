"""Tests for US-158: Voice transcript display in the Voice panel."""

from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
TEMPLATE_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "templates"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
JS_FILE = STATIC_DIR / "js" / "dashboard.js"
HTML_FILE = TEMPLATE_DIR / "index.html"


def _html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _js() -> str:
    return JS_FILE.read_text(encoding="utf-8")


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _non_local_env():
    return {"REMOTE_ADDR": "10.0.0.1"}


@pytest.fixture()
def app():
    from flask import Flask

    from rex.dashboard import dashboard_bp

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-158"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-158")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-158"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ── HTML structure ────────────────────────────────────────────────────────────


class TestHTMLTranscript:
    def test_voice_input_transcript_element_exists(self):
        assert 'id="voice-input-transcript"' in _html()

    def test_voice_response_transcript_element_exists(self):
        assert 'id="voice-response-transcript"' in _html()

    def test_transcript_panel_exists(self):
        assert "voice-transcript-panel" in _html()

    def test_you_said_label(self):
        assert "You said:" in _html()

    def test_rex_replied_label(self):
        assert "Rex replied:" in _html()

    def test_transcript_elements_inside_voice_panel(self):
        html = _html()
        panel_start = html.index('class="voice-mode-panel"')
        assert html.index('id="voice-input-transcript"') > panel_start
        assert html.index('id="voice-response-transcript"') > panel_start


# ── CSS styling ───────────────────────────────────────────────────────────────


class TestCSSTranscript:
    def test_voice_transcript_panel_class_defined(self):
        assert ".voice-transcript-panel" in _css()

    def test_voice_transcript_label_class_defined(self):
        assert ".voice-transcript-label" in _css()

    def test_voice_transcript_text_class_defined(self):
        assert ".voice-transcript-text" in _css()

    def test_transcript_text_has_color(self):
        css = _css()
        idx = css.index(".voice-transcript-text {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "color" in block


# ── Backend endpoint ──────────────────────────────────────────────────────────


class TestTranscriptEndpoint:
    def _get(self, client, token, path):
        return client.get(path, headers={"X-Auth-Token": token}, environ_base=_non_local_env())

    def _post(self, client, token, path, body):
        return client.post(
            path, json=body, headers={"X-Auth-Token": token}, environ_base=_non_local_env()
        )

    def test_get_transcript_returns_200(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-158")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/voice/transcript")
        assert resp.status_code == 200

    def test_get_transcript_returns_json(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-158")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/voice/transcript")
        data = resp.get_json()
        assert data is not None
        assert "input" in data
        assert "response" in data

    def test_transcript_initially_empty(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-158")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        import rex.dashboard.routes as routes

        monkeypatch.setattr(routes, "_VOICE_TRANSCRIPT", {"input": "", "response": ""})
        resp = self._get(client, auth_token, "/api/voice/transcript")
        data = resp.get_json()
        assert data["input"] == ""
        assert data["response"] == ""

    def test_start_listening_clears_transcript(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-158")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        import rex.dashboard.routes as routes

        monkeypatch.setattr(routes, "_VOICE_TRANSCRIPT", {"input": "hello", "response": "hi"})
        self._post(client, auth_token, "/api/voice/mode", {"active": True})
        resp = self._get(client, auth_token, "/api/voice/transcript")
        data = resp.get_json()
        assert data["input"] == ""
        assert data["response"] == ""

    def test_stop_listening_does_not_clear_transcript(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-158")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        import rex.dashboard.routes as routes

        monkeypatch.setattr(routes, "_VOICE_TRANSCRIPT", {"input": "hello", "response": "world"})
        self._post(client, auth_token, "/api/voice/mode", {"active": False})
        resp = self._get(client, auth_token, "/api/voice/transcript")
        data = resp.get_json()
        assert data["input"] == "hello"
        assert data["response"] == "world"

    def test_transcript_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.get("/api/voice/transcript", environ_base=_non_local_env())
        assert resp.status_code in (401, 403)


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSTranscript:
    def _get_update_fn_body(self) -> str:
        js = _js()
        fn_idx = js.index("function _updateVoiceModeUI")
        brace_start = js.index("{", fn_idx)
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

    def test_clear_fn_exists(self):
        assert "_clearVoiceTranscript" in _js()

    def test_fetch_fn_exists(self):
        assert "_fetchVoiceTranscript" in _js()

    def test_fetch_fn_calls_transcript_endpoint(self):
        js = _js()
        idx = js.index("_fetchVoiceTranscript")
        fn_start = js.index("{", idx)
        depth = 0
        i = fn_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    fn_body = js[fn_start : i + 1]
                    break
            i += 1
        assert "/api/voice/transcript" in fn_body

    def test_update_fn_clears_on_listening(self):
        body = self._get_update_fn_body()
        assert "_clearVoiceTranscript" in body
        assert "Listening" in body

    def test_update_fn_fetches_on_idle(self):
        body = self._get_update_fn_body()
        assert "_fetchVoiceTranscript" in body
        assert "Idle" in body

    def test_clear_fn_targets_input_element(self):
        js = _js()
        idx = js.index("function _clearVoiceTranscript")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    fn_body = js[brace_start : i + 1]
                    break
            i += 1
        assert "voice-input-transcript" in fn_body
        assert "voice-response-transcript" in fn_body

    def test_fetch_fn_sets_input_text(self):
        js = _js()
        idx = js.index("async function _fetchVoiceTranscript")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    fn_body = js[brace_start : i + 1]
                    break
            i += 1
        assert "voice-input-transcript" in fn_body
        assert "voice-response-transcript" in fn_body
        assert "data.input" in fn_body
        assert "data.response" in fn_body
