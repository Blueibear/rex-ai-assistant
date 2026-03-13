"""Tests for US-162: Overview panel with Rex status summary."""

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
    from rex.dashboard import dashboard_bp
    from flask import Flask

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-162"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-162")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-162"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ── HTML structure ─────────────────────────────────────────────────────────────


class TestHTMLOverview:
    def test_overview_section_exists(self):
        assert 'id="overview-section"' in _html()

    def test_overview_status_grid_exists(self):
        assert 'id="overview-status-grid"' in _html()

    def test_overview_grid_has_class(self):
        html = _html()
        idx = html.index('id="overview-status-grid"')
        tag_end = html.index(">", idx)
        tag = html[idx:tag_end]
        assert "overview-status-grid" in tag

    def test_overview_has_heading(self):
        html = _html()
        section_start = html.index('id="overview-section"')
        section_end = html.index("</section>", section_start)
        section = html[section_start:section_end]
        assert "<h1>Overview</h1>" in section

    def test_placeholder_removed(self):
        assert "System overview coming soon" not in _html()


# ── CSS styling ────────────────────────────────────────────────────────────────


class TestCSSOverview:
    def test_overview_status_grid_class_defined(self):
        assert ".overview-status-grid" in _css()

    def test_overview_status_card_class_defined(self):
        assert ".overview-status-card" in _css()

    def test_overview_indicator_class_defined(self):
        assert ".overview-indicator" in _css()

    def test_overview_indicator_ok_class_defined(self):
        assert ".overview-indicator.ok" in _css()

    def test_overview_indicator_error_class_defined(self):
        assert ".overview-indicator.error" in _css()

    def test_overview_status_label_class_defined(self):
        assert ".overview-status-label" in _css()

    def test_overview_status_value_class_defined(self):
        assert ".overview-status-value" in _css()

    def test_overview_grid_uses_grid_layout(self):
        css = _css()
        idx = css.index(".overview-status-grid")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "grid" in block

    def test_indicator_is_circular(self):
        css = _css()
        idx = css.index(".overview-indicator {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "border-radius" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSOverview:
    def _get_load_fn_body(self) -> str:
        js = _js()
        idx = js.index("async function loadOverview")
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

    def _get_render_fn_body(self) -> str:
        js = _js()
        idx = js.index("function renderOverview")
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

    def test_load_overview_fn_exists(self):
        assert "loadOverview" in _js()

    def test_render_overview_fn_exists(self):
        assert "renderOverview" in _js()

    def test_default_section_is_overview(self):
        js = _js()
        idx = js.index("currentSection:")
        line_end = js.index("\n", idx)
        line = js[idx:line_end]
        assert "overview" in line

    def test_switch_case_includes_overview(self):
        assert "case 'overview'" in _js()

    def test_switch_case_calls_load_overview(self):
        js = _js()
        idx = js.index("case 'overview'")
        break_idx = js.index("break", idx)
        block = js[idx:break_idx]
        assert "loadOverview" in block

    def test_load_fn_fetches_status(self):
        body = self._get_load_fn_body()
        assert "/api/dashboard/status" in body

    def test_load_fn_fetches_voice_mode(self):
        body = self._get_load_fn_body()
        assert "/api/voice/mode" in body

    def test_load_fn_fetches_scheduler_jobs(self):
        body = self._get_load_fn_body()
        assert "/api/scheduler/jobs" in body

    def test_load_fn_fetches_notifications(self):
        body = self._get_load_fn_body()
        assert "/api/notifications" in body

    def test_render_fn_shows_rex_status(self):
        body = self._get_render_fn_body()
        assert "Rex Status" in body or "Online" in body

    def test_render_fn_shows_voice_mode(self):
        body = self._get_render_fn_body()
        assert "Voice Mode" in body or "Active" in body

    def test_render_fn_shows_lm_studio(self):
        body = self._get_render_fn_body()
        assert "LM Studio" in body

    def test_render_fn_shows_scheduled_count(self):
        body = self._get_render_fn_body()
        assert "Scheduled" in body or "scheduledCount" in body

    def test_render_fn_shows_notifications(self):
        body = self._get_render_fn_body()
        assert "Notification" in body or "unreadCount" in body

    def test_render_fn_uses_indicator_class(self):
        # overview-indicator class may be in the _overviewIndicator helper
        assert "overview-indicator" in _js()

    def test_render_fn_uses_status_card_class(self):
        body = self._get_render_fn_body()
        assert "overview-status-card" in body

    def test_render_fn_uses_status_label_class(self):
        body = self._get_render_fn_body()
        assert "overview-status-label" in body


# ── Backend endpoints ──────────────────────────────────────────────────────────


class TestOverviewEndpoints:
    def _get(self, client, token, path):
        return client.get(
            path,
            headers={"X-Auth-Token": token},
            environ_base=_non_local_env(),
        )

    def test_dashboard_status_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.get("/api/dashboard/status", environ_base=_non_local_env())
        assert resp.status_code == 200

    def test_dashboard_status_returns_status_ok(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.get("/api/dashboard/status", environ_base=_non_local_env())
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_voice_mode_returns_200(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-162")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/voice/mode")
        assert resp.status_code == 200

    def test_voice_mode_has_active_field(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-162")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/voice/mode")
        data = resp.get_json()
        assert "active" in data
