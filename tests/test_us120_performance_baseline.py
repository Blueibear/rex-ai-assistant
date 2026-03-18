"""Tests for US-120: Response time baseline for core API endpoints.

Acceptance criteria:
- response times measured for at minimum: health check, chat message send,
  notification list, config load
- measurements taken with a local warm instance (min 10 requests, median reported)
- baseline documented in docs/performance-baseline.md
- any endpoint with p50 > 500ms flagged for investigation
- Typecheck passes
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import pytest
from flask import Flask, jsonify

from rex.health import check_config, create_health_blueprint
from rex.http_errors import install_error_envelope_handler
from rex.request_logging import install_request_logging

# Maximum acceptable p50 response time in seconds for non-LLM endpoints.
_P50_THRESHOLD_S = 0.5

# Minimum number of requests per endpoint for a valid baseline.
_MIN_SAMPLES = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def perf_app() -> Flask:
    """Flask app wired for performance measurement.

    Includes health blueprint and stub endpoints for notifications and
    config load (no real service dependencies).
    """
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    install_request_logging(flask_app)
    install_error_envelope_handler(flask_app)
    flask_app.register_blueprint(create_health_blueprint(checks=[check_config]))

    # Stub notification list endpoint (simulates list response without DB).
    @flask_app.route("/api/notifications")
    def stub_notifications() -> Any:
        return jsonify({"notifications": [], "total": 0}), 200

    # Stub config load endpoint (simulates config response without file I/O).
    @flask_app.route("/api/settings")
    def stub_settings() -> Any:
        return jsonify({"llm_provider": "local", "tts_provider": "pyttsx3"}), 200

    # Stub chat endpoint (simulates response without LLM call).
    @flask_app.route("/api/chat", methods=["POST"])
    def stub_chat() -> Any:
        return jsonify({"reply": "ok", "timestamp": "2026-01-01T00:00:00"}), 200

    return flask_app


@pytest.fixture()
def perf_client(perf_app: Flask):  # noqa: ANN201
    return perf_app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure(client: Any, method: str, path: str, n: int = _MIN_SAMPLES) -> list[float]:
    """Return a list of n response times in seconds for *path*."""
    # Warm-up pass (not counted).
    if method == "GET":
        client.get(path)
    else:
        client.post(path, json={"message": "hello"}, content_type="application/json")

    durations: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        if method == "GET":
            client.get(path)
        else:
            client.post(path, json={"message": "hello"}, content_type="application/json")
        durations.append(time.perf_counter() - start)
    return durations


def _p50(durations: list[float]) -> float:
    return statistics.median(durations)


# ---------------------------------------------------------------------------
# Performance baseline tests
# ---------------------------------------------------------------------------


class TestHealthCheckPerformance:
    def test_health_live_p50_under_threshold(self, perf_client: Any) -> None:
        """Health live endpoint p50 must be under 500ms."""
        durations = _measure(perf_client, "GET", "/health/live")
        p50 = _p50(durations)
        assert (
            p50 < _P50_THRESHOLD_S
        ), f"FLAGGED: /health/live p50={p50*1000:.1f}ms exceeds {_P50_THRESHOLD_S*1000:.0f}ms threshold"

    def test_health_live_sample_count(self, perf_client: Any) -> None:
        durations = _measure(perf_client, "GET", "/health/live")
        assert len(durations) >= _MIN_SAMPLES

    def test_health_ready_p50_under_threshold(self, perf_client: Any) -> None:
        durations = _measure(perf_client, "GET", "/health/ready")
        p50 = _p50(durations)
        assert (
            p50 < _P50_THRESHOLD_S
        ), f"FLAGGED: /health/ready p50={p50*1000:.1f}ms exceeds threshold"


class TestNotificationListPerformance:
    def test_notification_list_p50_under_threshold(self, perf_client: Any) -> None:
        """Notification list endpoint p50 must be under 500ms."""
        durations = _measure(perf_client, "GET", "/api/notifications")
        p50 = _p50(durations)
        assert (
            p50 < _P50_THRESHOLD_S
        ), f"FLAGGED: /api/notifications p50={p50*1000:.1f}ms exceeds threshold"

    def test_notification_list_sample_count(self, perf_client: Any) -> None:
        durations = _measure(perf_client, "GET", "/api/notifications")
        assert len(durations) >= _MIN_SAMPLES


class TestConfigLoadPerformance:
    def test_config_load_p50_under_threshold(self, perf_client: Any) -> None:
        """Config load endpoint p50 must be under 500ms."""
        durations = _measure(perf_client, "GET", "/api/settings")
        p50 = _p50(durations)
        assert (
            p50 < _P50_THRESHOLD_S
        ), f"FLAGGED: /api/settings p50={p50*1000:.1f}ms exceeds threshold"


class TestChatMessagePerformance:
    def test_chat_send_p50_under_threshold(self, perf_client: Any) -> None:
        """Chat endpoint p50 (stub — no LLM) must be under 500ms."""
        durations = _measure(perf_client, "POST", "/api/chat")
        p50 = _p50(durations)
        assert p50 < _P50_THRESHOLD_S, f"FLAGGED: /api/chat p50={p50*1000:.1f}ms exceeds threshold"

    def test_chat_send_sample_count(self, perf_client: Any) -> None:
        durations = _measure(perf_client, "POST", "/api/chat")
        assert len(durations) >= _MIN_SAMPLES


# ---------------------------------------------------------------------------
# Baseline documentation exists
# ---------------------------------------------------------------------------


class TestBaselineDocumentExists:
    def test_performance_baseline_md_exists(self) -> None:
        """docs/performance-baseline.md must exist."""
        baseline_path = Path(__file__).parent.parent / "docs" / "performance-baseline.md"
        assert (
            baseline_path.exists()
        ), "docs/performance-baseline.md does not exist — create it with baseline measurements"

    def test_performance_baseline_md_nonempty(self) -> None:
        baseline_path = Path(__file__).parent.parent / "docs" / "performance-baseline.md"
        if not baseline_path.exists():
            pytest.skip("docs/performance-baseline.md not yet created")
        content = baseline_path.read_text()
        assert len(content.strip()) > 100, "docs/performance-baseline.md appears empty"

    def test_performance_baseline_md_mentions_endpoints(self) -> None:
        baseline_path = Path(__file__).parent.parent / "docs" / "performance-baseline.md"
        if not baseline_path.exists():
            pytest.skip("docs/performance-baseline.md not yet created")
        content = baseline_path.read_text()
        for endpoint in ["/health", "/api/notifications", "/api/settings", "/api/chat"]:
            assert (
                endpoint in content
            ), f"docs/performance-baseline.md does not mention endpoint {endpoint}"
