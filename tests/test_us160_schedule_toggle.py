"""Tests for US-160: Enable/disable scheduled items from the GUI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    flask_app.config["SECRET_KEY"] = "test-secret-160"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-160")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-160"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ── HTML structure ─────────────────────────────────────────────────────────────


class TestHTMLToggle:
    def test_schedule_section_still_exists(self):
        assert 'id="schedule-section"' in _html()

    def test_schedule_list_still_exists(self):
        assert 'id="schedule-list"' in _html()


# ── CSS styling ────────────────────────────────────────────────────────────────


class TestCSSToggle:
    def test_schedule_item_actions_class_defined(self):
        assert ".schedule-item-actions" in _css()

    def test_schedule_toggle_error_class_defined(self):
        assert ".schedule-toggle-error" in _css()

    def test_actions_uses_flex(self):
        css = _css()
        idx = css.index(".schedule-item-actions")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSToggle:
    def _get_toggle_fn_body(self) -> str:
        js = _js()
        idx = js.index("async function toggleScheduleJob")
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
        idx = js.index("function renderScheduleJobs")
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

    def test_toggle_fn_exists(self):
        assert "toggleScheduleJob" in _js()

    def test_render_fn_includes_toggle_button(self):
        assert "schedule-toggle-btn" in self._get_render_fn_body()

    def test_render_fn_toggle_calls_handler(self):
        assert "dashboardHandlers.toggleScheduleJob" in self._get_render_fn_body()

    def test_toggle_fn_calls_patch_endpoint(self):
        body = self._get_toggle_fn_body()
        assert "PATCH" in body
        assert "/api/scheduler/jobs/" in body

    def test_toggle_fn_sends_enabled_field(self):
        body = self._get_toggle_fn_body()
        assert "enabled" in body

    def test_toggle_fn_updates_status_element(self):
        body = self._get_toggle_fn_body()
        assert "schedule-item-status" in body

    def test_toggle_fn_reverts_on_error(self):
        body = self._get_toggle_fn_body()
        assert "catch" in body
        assert "currentEnabled" in body

    def test_toggle_fn_shows_error_message(self):
        body = self._get_toggle_fn_body()
        assert "schedule-toggle-error" in body or "error-message" in body

    def test_toggle_fn_registered_in_handlers(self):
        js = _js()
        idx = js.index("window.dashboardHandlers = {")
        brace_start = js.index("{", idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    handlers_block = js[brace_start : i + 1]
                    break
            i += 1
        assert "toggleScheduleJob" in handlers_block

    def test_toggle_btn_renders_with_job_id(self):
        body = self._get_render_fn_body()
        assert "data-job-id" in body

    def test_toggle_btn_renders_enabled_label(self):
        body = self._get_render_fn_body()
        assert "Disable" in body
        assert "Enable" in body


# ── Backend endpoint ───────────────────────────────────────────────────────────


class TestToggleEndpoint:
    def _patch(self, client, token, job_id, body):
        return client.patch(
            f"/api/scheduler/jobs/{job_id}",
            json=body,
            headers={"X-Auth-Token": token},
            environ_base=_non_local_env(),
        )

    def test_patch_job_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.patch(
            "/api/scheduler/jobs/test-job-id",
            json={"enabled": False},
            environ_base=_non_local_env(),
        )
        assert resp.status_code in (401, 403)

    def test_patch_job_not_found_returns_404(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-160")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        from rex.scheduler import get_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        with patch("rex.dashboard.routes.get_scheduler", return_value=mock_scheduler):
            resp = self._patch(client, auth_token, "nonexistent-job", {"enabled": False})
        assert resp.status_code == 404

    def test_patch_job_no_body_returns_400(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-160")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.patch(
            "/api/scheduler/jobs/some-job",
            json={},
            headers={"X-Auth-Token": auth_token},
            environ_base=_non_local_env(),
        )
        assert resp.status_code == 400

    def test_patch_job_updates_enabled(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-160")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        from datetime import datetime, timezone

        mock_job = MagicMock()
        mock_job.job_id = "job-abc"
        mock_job.name = "Test Job"
        mock_job.schedule = "interval:60"
        mock_job.enabled = False
        mock_job.next_run = None

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = mock_job
        mock_scheduler.update_job.return_value = mock_job

        with patch("rex.dashboard.routes.get_scheduler", return_value=mock_scheduler):
            resp = self._patch(client, auth_token, "job-abc", {"enabled": False})

        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data
        assert "enabled" in data
