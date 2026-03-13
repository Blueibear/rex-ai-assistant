"""Tests for US-165: Loading and error states for all data-fetching panels."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
JS_FILE = STATIC_DIR / "js" / "dashboard.js"


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _js() -> str:
    return JS_FILE.read_text(encoding="utf-8")


def _fn_body(js: str, fn_signature: str) -> str:
    """Extract body of a named function by brace-depth tracking."""
    idx = js.index(fn_signature)
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


# ── CSS ────────────────────────────────────────────────────────────────────────


class TestCSSLoadingError:
    def test_panel_error_class_defined(self):
        assert ".panel-error" in _css()

    def test_retry_btn_class_defined(self):
        assert ".retry-btn" in _css()

    def test_panel_error_uses_flex(self):
        css = _css()
        idx = css.index(".panel-error")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block


# ── JS helper ─────────────────────────────────────────────────────────────────


class TestJSRetryHelper:
    def test_retry_error_html_helper_exists(self):
        assert "_retryErrorHtml" in _js()

    def test_retry_error_html_returns_retry_btn(self):
        js = _js()
        body = _fn_body(js, "function _retryErrorHtml")
        assert "retry-btn" in body

    def test_retry_error_html_uses_handler(self):
        js = _js()
        body = _fn_body(js, "function _retryErrorHtml")
        assert "dashboardHandlers" in body

    def test_retry_error_html_has_panel_error_class(self):
        js = _js()
        body = _fn_body(js, "function _retryErrorHtml")
        assert "panel-error" in body


# ── Loading indicators ────────────────────────────────────────────────────────


class TestLoadingIndicators:
    def test_load_overview_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadOverview")
        assert "Loading" in body or "loading" in body

    def test_load_chat_history_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadChatHistory")
        assert "Loading" in body or "loading" in body

    def test_load_settings_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadSettings")
        assert "Loading" in body or "loading" in body

    def test_load_reminders_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadReminders")
        assert "Loading" in body or "loading" in body

    def test_load_schedule_jobs_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadScheduleJobs")
        assert "Loading" in body or "loading" in body

    def test_load_notifications_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadNotifications")
        assert "Loading" in body or "loading" in body

    def test_load_status_has_loading(self):
        js = _js()
        body = _fn_body(js, "async function loadStatus")
        assert "Loading" in body or "loading" in body


# ── Specific error messages ───────────────────────────────────────────────────


class TestSpecificErrorMessages:
    def test_load_overview_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadOverview")
        assert "Failed to load overview" in body

    def test_load_chat_history_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadChatHistory")
        assert "Failed to load chat history" in body

    def test_load_settings_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadSettings")
        assert "Failed to load settings" in body

    def test_load_reminders_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadReminders")
        assert "Failed to load reminders" in body

    def test_load_schedule_jobs_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadScheduleJobs")
        assert "Failed to load schedule" in body

    def test_load_notifications_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadNotifications")
        assert "Failed to load notifications" in body

    def test_load_status_has_specific_error(self):
        js = _js()
        body = _fn_body(js, "async function loadStatus")
        assert "Failed to load status" in body


# ── Retry buttons ─────────────────────────────────────────────────────────────


class TestRetryButtons:
    def _has_retry(self, fn_sig: str) -> bool:
        js = _js()
        body = _fn_body(js, fn_sig)
        return "_retryErrorHtml" in body

    def test_load_overview_has_retry(self):
        assert self._has_retry("async function loadOverview")

    def test_load_chat_history_has_retry(self):
        assert self._has_retry("async function loadChatHistory")

    def test_load_settings_has_retry(self):
        assert self._has_retry("async function loadSettings")

    def test_load_reminders_has_retry(self):
        assert self._has_retry("async function loadReminders")

    def test_load_schedule_jobs_has_retry(self):
        assert self._has_retry("async function loadScheduleJobs")

    def test_load_notifications_has_retry(self):
        assert self._has_retry("async function loadNotifications")

    def test_load_status_has_retry(self):
        assert self._has_retry("async function loadStatus")


# ── Handlers exposed ──────────────────────────────────────────────────────────


class TestHandlersExposed:
    def _get_handlers_block(self) -> str:
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
                    return js[brace_start : i + 1]
            i += 1
        return js[brace_start:]

    def test_load_overview_in_handlers(self):
        assert "loadOverview" in self._get_handlers_block()

    def test_load_chat_history_in_handlers(self):
        assert "loadChatHistory" in self._get_handlers_block()

    def test_load_settings_in_handlers(self):
        assert "loadSettings" in self._get_handlers_block()

    def test_load_reminders_in_handlers(self):
        assert "loadReminders" in self._get_handlers_block()

    def test_load_schedule_jobs_in_handlers(self):
        assert "loadScheduleJobs" in self._get_handlers_block()

    def test_load_notifications_in_handlers(self):
        assert "loadNotifications" in self._get_handlers_block()

    def test_load_status_in_handlers(self):
        assert "loadStatus" in self._get_handlers_block()
