"""Tests for US-044: smart local/cloud routing (prefer local).

Covers:
- estimate_complexity for simple and complex queries
- route() under all three routing_mode values
- local-unavailable fallback to cloud
- AppConfig.llm_routing_mode field
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rex.model_router import ModelRouter

LOCAL = "llama3"
CLOUD = "gpt-4o"


@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter()


# ---------------------------------------------------------------------------
# estimate_complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    def test_short_no_tools_is_simple(self, router):
        assert router.estimate_complexity("What time is it?") == "simple"

    def test_requires_tools_is_complex(self, router):
        assert router.estimate_complexity("What time is it?", requires_tools=True) == "complex"

    def test_long_message_is_complex(self, router):
        # ~160 words ≈ 208 estimated tokens → complex
        words = " ".join(["word"] * 160)
        assert router.estimate_complexity(words) == "complex"

    def test_short_message_is_simple(self, router):
        assert router.estimate_complexity("Hello") == "simple"


# ---------------------------------------------------------------------------
# route() — cloud_only mode
# ---------------------------------------------------------------------------


class TestCloudOnlyMode:
    def test_simple_query_uses_cloud(self, router):
        result = router.route(
            "Hello", local_model=LOCAL, cloud_model=CLOUD, routing_mode="cloud_only"
        )
        assert result == CLOUD

    def test_complex_query_uses_cloud(self, router):
        words = " ".join(["word"] * 160)
        result = router.route(
            words, local_model=LOCAL, cloud_model=CLOUD, routing_mode="cloud_only"
        )
        assert result == CLOUD


# ---------------------------------------------------------------------------
# route() — local_only mode
# ---------------------------------------------------------------------------


class TestLocalOnlyMode:
    def test_local_available_returns_local(self, router):
        with patch.object(router, "_is_available", return_value=True):
            result = router.route(
                "Hello", local_model=LOCAL, cloud_model=CLOUD, routing_mode="local_only"
            )
        assert result == LOCAL

    def test_local_unavailable_stays_local_by_policy(self, router):
        with patch.object(router, "_is_available", return_value=False):
            result = router.route(
                "Hello", local_model=LOCAL, cloud_model=CLOUD, routing_mode="local_only"
            )
        assert result == LOCAL


# ---------------------------------------------------------------------------
# route() — local_preferred mode (default)
# ---------------------------------------------------------------------------


class TestLocalPreferredMode:
    def test_simple_query_uses_local_when_available(self, router):
        with patch.object(router, "_is_available", return_value=True):
            result = router.route("Hello", local_model=LOCAL, cloud_model=CLOUD)
        assert result == LOCAL

    def test_simple_query_falls_back_to_cloud_when_local_unavailable(self, router):
        with patch.object(router, "_is_available", return_value=False):
            result = router.route("Hello", local_model=LOCAL, cloud_model=CLOUD)
        assert result == CLOUD

    def test_complex_query_uses_cloud(self, router):
        with patch.object(router, "_is_available", return_value=True):
            words = " ".join(["word"] * 160)
            result = router.route(words, local_model=LOCAL, cloud_model=CLOUD)
        assert result == CLOUD

    def test_tool_query_uses_cloud(self, router):
        with patch.object(router, "_is_available", return_value=True):
            result = router.route(
                "What time is it?",
                local_model=LOCAL,
                cloud_model=CLOUD,
                requires_tools=True,
            )
        assert result == CLOUD

    def test_complex_no_cloud_configured_falls_back_to_local(self, router):
        with patch.object(router, "_is_available", return_value=True):
            words = " ".join(["word"] * 160)
            result = router.route(words, local_model=LOCAL, cloud_model="")
        assert result == LOCAL


# ---------------------------------------------------------------------------
# Invalid routing_mode falls back to local_preferred
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_invalid_mode_treated_as_local_preferred(self, router):
        with patch.object(router, "_is_available", return_value=True):
            result = router.route(
                "Hello", local_model=LOCAL, cloud_model=CLOUD, routing_mode="bogus"
            )
        assert result == LOCAL


# ---------------------------------------------------------------------------
# AppConfig.llm_routing_mode field
# ---------------------------------------------------------------------------


class TestAppConfigRoutingMode:
    def test_default_is_local_preferred(self):
        from rex.config import AppConfig

        cfg = AppConfig()
        assert cfg.llm_routing_mode == "local_preferred"

    def test_accepts_cloud_only(self):
        from rex.config import AppConfig

        cfg = AppConfig(llm_routing_mode="cloud_only")
        assert cfg.llm_routing_mode == "cloud_only"

    def test_accepts_local_only(self):
        from rex.config import AppConfig

        cfg = AppConfig(llm_routing_mode="local_only")
        assert cfg.llm_routing_mode == "local_only"
