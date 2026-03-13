"""Tests for US-159: Scheduled items list view in the Schedule panel."""

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
    flask_app.config["SECRET_KEY"] = "test-secret-159"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_token(client, monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-159")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pass-159"},
        environ_base=_non_local_env(),
    )
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ── HTML structure ─────────────────────────────────────────────────────────────


class TestHTMLScheduleList:
    def test_schedule_section_exists(self):
        assert 'id="schedule-section"' in _html()

    def test_schedule_list_container_exists(self):
        assert 'id="schedule-list"' in _html()

    def test_schedule_list_has_schedule_list_class(self):
        html = _html()
        idx = html.index('id="schedule-list"')
        tag_end = html.index(">", idx)
        tag = html[idx:tag_end]
        assert "schedule-list" in tag

    def test_schedule_empty_state_exists(self):
        assert "schedule-empty" in _html()

    def test_empty_state_has_message(self):
        html = _html()
        idx = html.index("schedule-empty")
        # find the text content of the element
        tag_end = html.index(">", idx)
        close_tag = html.index("</div>", tag_end)
        content = html[tag_end + 1 : close_tag]
        assert len(content.strip()) > 0

    def test_schedule_list_inside_schedule_section(self):
        html = _html()
        section_start = html.index('id="schedule-section"')
        assert html.index('id="schedule-list"') > section_start

    def test_schedule_section_has_heading(self):
        html = _html()
        section_start = html.index('id="schedule-section"')
        section_end = html.index("</section>", section_start)
        section = html[section_start:section_end]
        assert "<h1>Schedule</h1>" in section

    def test_placeholder_content_removed(self):
        assert "Schedule management coming soon" not in _html()


# ── CSS styling ────────────────────────────────────────────────────────────────


class TestCSSScheduleList:
    def test_schedule_list_class_defined(self):
        assert ".schedule-list" in _css()

    def test_schedule_empty_class_defined(self):
        assert ".schedule-empty" in _css()

    def test_schedule_item_class_defined(self):
        assert ".schedule-item {" in _css() or ".schedule-item{" in _css()

    def test_schedule_item_header_class_defined(self):
        assert ".schedule-item-header" in _css()

    def test_schedule_item_name_class_defined(self):
        assert ".schedule-item-name" in _css()

    def test_schedule_item_status_class_defined(self):
        assert ".schedule-item-status" in _css()

    def test_schedule_item_status_enabled_class_defined(self):
        assert ".schedule-item-status.enabled" in _css()

    def test_schedule_item_status_disabled_class_defined(self):
        assert ".schedule-item-status.disabled" in _css()

    def test_schedule_item_details_class_defined(self):
        assert ".schedule-item-details" in _css()

    def test_schedule_list_is_flex_column(self):
        css = _css()
        idx = css.index(".schedule-list")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSScheduleList:
    def _get_load_fn_body(self) -> str:
        js = _js()
        idx = js.index("async function loadScheduleJobs")
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

    def test_load_fn_exists(self):
        assert "loadScheduleJobs" in _js()

    def test_render_fn_exists(self):
        assert "renderScheduleJobs" in _js()

    def test_load_fn_calls_scheduler_jobs_endpoint(self):
        assert "/api/scheduler/jobs" in self._get_load_fn_body()

    def test_load_fn_targets_schedule_list_element(self):
        assert "schedule-list" in self._get_load_fn_body()

    def test_render_fn_targets_schedule_list_element(self):
        assert "schedule-list" in self._get_render_fn_body()

    def test_render_fn_renders_schedule_empty_on_empty(self):
        assert "schedule-empty" in self._get_render_fn_body()

    def test_render_fn_renders_schedule_item(self):
        assert "schedule-item" in self._get_render_fn_body()

    def test_render_fn_shows_item_name(self):
        assert "schedule-item-name" in self._get_render_fn_body()

    def test_render_fn_shows_item_status(self):
        assert "schedule-item-status" in self._get_render_fn_body()

    def test_render_fn_shows_schedule_field(self):
        body = self._get_render_fn_body()
        assert "formatSchedule" in body

    def test_render_fn_shows_next_run(self):
        body = self._get_render_fn_body()
        assert "next_run" in body

    def test_switch_case_includes_schedule(self):
        js = _js()
        assert "case 'schedule'" in js

    def test_switch_case_calls_load_fn(self):
        js = _js()
        idx = js.index("case 'schedule'")
        # Find the break that closes this case
        break_idx = js.index("break", idx)
        block = js[idx:break_idx]
        assert "loadScheduleJobs" in block


# ── Backend endpoint ───────────────────────────────────────────────────────────


class TestScheduleEndpoint:
    def _get(self, client, token, path):
        return client.get(
            path,
            headers={"X-Auth-Token": token},
            environ_base=_non_local_env(),
        )

    def test_get_jobs_returns_200(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-159")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/scheduler/jobs")
        assert resp.status_code == 200

    def test_get_jobs_returns_json_with_jobs_key(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-159")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/scheduler/jobs")
        data = resp.get_json()
        assert data is not None
        assert "jobs" in data

    def test_get_jobs_returns_total_key(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-159")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/scheduler/jobs")
        data = resp.get_json()
        assert "total" in data

    def test_get_jobs_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = client.get("/api/scheduler/jobs", environ_base=_non_local_env())
        assert resp.status_code in (401, 403)

    def test_jobs_is_a_list(self, client, auth_token, monkeypatch):
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pass-159")
        monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
        resp = self._get(client, auth_token, "/api/scheduler/jobs")
        data = resp.get_json()
        assert isinstance(data["jobs"], list)
