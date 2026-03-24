from __future__ import annotations

import pytest

from plugins.web_search import WebSearchPlugin


def test_web_search_plugin_fallback(monkeypatch):
    plugin = WebSearchPlugin()

    monkeypatch.setenv("REX_SEARCH_PROVIDERS", "serpapi,duckduckgo")
    monkeypatch.setenv("SERPAPI_KEY", "")

    def fake_duckduckgo(self, query):
        return "duck result"

    monkeypatch.setattr(WebSearchPlugin, "_search_duckduckgo", fake_duckduckgo, raising=False)

    result = plugin.process("hi")
    plugin.shutdown()
    assert result == "duck result"


def test_web_search_missing_requests_raises(monkeypatch):
    import plugins.web_search as web_search

    monkeypatch.setattr(web_search, "requests", None)
    monkeypatch.setattr(
        web_search, "_REQUESTS_IMPORT_ERROR", ImportError("requests missing"), raising=False
    )

    plugin = web_search.WebSearchPlugin()
    monkeypatch.setenv("SERPAPI_KEY", "token")

    with pytest.raises(RuntimeError, match="requests"):
        plugin._search_serpapi("hello")
