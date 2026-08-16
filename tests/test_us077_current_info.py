from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rex.intent.router import IntentRouter

NEWS_PROMPT = "what is in the news today"


def _settings(providers: str) -> SimpleNamespace:
    return SimpleNamespace(search_providers=providers, brave_api_key=None)


def test_current_news_marks_configured_request_for_pre_llm_web_search() -> None:
    router = IntentRouter()
    with patch("plugins.web_search.configured_search_providers", return_value=["duckduckgo"]):
        result = router.route(NEWS_PROMPT, settings=_settings("duckduckgo"))

    assert result.handled is False
    assert result.intent_type == "current_info"
    assert result.response is None


def test_current_news_unconfigured_explains_real_setup_path() -> None:
    router = IntentRouter()
    with patch("plugins.web_search.configured_search_providers", return_value=[]):
        result = router.route(NEWS_PROMPT, settings=_settings(""))

    assert result.handled is True
    assert result.intent_type == "current_info_unavailable"
    response = result.response or ""
    assert "Web Search" in response
    assert "search.providers" in response
    assert "docs/configuration.md" in response
    assert "DuckDuckGo" in response
    assert "live news" not in response.lower()


def test_historical_news_question_is_not_forced_into_live_search() -> None:
    result = IntentRouter().route("What was in the news on July 20, 1969?", settings=_settings(""))

    assert result.handled is False


def test_current_news_setup_guidance_points_to_real_config_and_docs() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    schema = (root / "config" / "rex_config.schema.json").read_text(encoding="utf-8")
    docs = (root / "docs" / "configuration.md").read_text(encoding="utf-8")

    assert '"search"' in schema and '"providers"' in schema
    assert "### Web Search" in docs
    assert "search.providers" in docs


@pytest.mark.parametrize(
    "prompt",
    [
        "latest news about the 2026 election",
        "What are the latest updates on Ukraine?",
        "What happened today?",
        "What's the latest on the storm?",
    ],
)
def test_explicit_current_cues_route_to_verified_current_info_even_with_years(prompt: str) -> None:
    with patch("plugins.web_search.configured_search_providers", return_value=["duckduckgo"]):
        result = IntentRouter().route(prompt, settings=_settings("duckduckgo"))

    assert result.handled is False
    assert result.intent_type == "current_info"


def test_year_without_current_cue_is_not_forced_into_live_search() -> None:
    result = IntentRouter().route("Tell me about the 2026 election", settings=_settings(""))

    assert result.handled is False
    assert result.intent_type is None


def test_current_info_intent_bypasses_response_cache() -> None:
    from rex.assistant import Assistant
    from rex.intent.router import IntentResult

    assert (
        Assistant._should_check_response_cache(
            IntentResult(handled=False, response=None, intent_type="current_info")
        )
        is False
    )
    assert (
        Assistant._should_check_response_cache(
            IntentResult(handled=False, response=None, intent_type="general")
        )
        is True
    )
