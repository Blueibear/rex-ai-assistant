from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from plugins.web_search import configured_search_providers


def test_duckduckgo_is_usable_without_api_key_when_enabled() -> None:
    config = SimpleNamespace(search_providers="duckduckgo", brave_api_key=None)

    assert configured_search_providers(config, environ={}) == ["duckduckgo"]


def test_disabled_provider_list_is_unconfigured() -> None:
    config = SimpleNamespace(search_providers="", brave_api_key=None)

    assert configured_search_providers(config, environ={}) == []


def test_brave_runtime_config_key_counts_as_configured() -> None:
    config = SimpleNamespace(search_providers="brave", brave_api_key="vault-brave-key")

    assert configured_search_providers(config, environ={}) == ["brave"]


def test_serpapi_accepts_documented_key_name() -> None:
    config = SimpleNamespace(search_providers="serpapi", brave_api_key=None)

    assert configured_search_providers(config, environ={"SERPAPI_KEY": "serp-key"}) == ["serpapi"]


def test_serpapi_accepts_legacy_api_key_alias() -> None:
    config = SimpleNamespace(search_providers="serpapi", brave_api_key=None)

    assert configured_search_providers(config, environ={"SERPAPI_API_KEY": "serp-key"}) == [
        "serpapi"
    ]


def test_unconfigured_brave_does_not_count_vault_lookup_failure() -> None:
    config = SimpleNamespace(search_providers="brave", brave_api_key=None)
    with patch("rex.credentials.get_persisted_credential", return_value=None):
        assert configured_search_providers(config, environ={}) == []


def test_current_news_selects_web_search_in_canonical_dispatcher() -> None:
    from rex.tools.dispatcher import ToolDispatcher
    from rex.tools.registry import get_default_registry

    config = SimpleNamespace(search_providers="duckduckgo", tool_timeout_seconds=10)
    selected = ToolDispatcher(get_default_registry(), config=config).select_tools(
        "web search what is in the news today"
    )

    assert "web_search" in {tool.name for tool in selected}


def test_web_search_execution_uses_same_runtime_config_as_selection() -> None:
    from rex.tools.dispatcher import ToolDispatcher
    from rex.tools.registry import get_default_registry

    config = SimpleNamespace(search_providers="duckduckgo", tool_timeout_seconds=10)
    dispatcher = ToolDispatcher(get_default_registry(), config=config)
    with patch("plugins.web_search.search_web", return_value="verified") as search:
        result = dispatcher.dispatch("web_search", {"transcript": "latest news"})

    assert result.success is True
    assert result.output == "verified"
    search.assert_called_once_with("latest news", config=config)


def test_explicit_runtime_config_does_not_use_legacy_provider_env_override() -> None:
    config = SimpleNamespace(search_providers="", brave_api_key=None)

    assert configured_search_providers(config, environ={"REX_SEARCH_PROVIDERS": "duckduckgo"}) == []
