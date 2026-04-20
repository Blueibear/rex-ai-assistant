from __future__ import annotations

import pytest

from plugins.web_search import WebSearchPlugin


def test_web_search_plugin_fallback(monkeypatch):
    plugin = WebSearchPlugin()
    queries = []

    monkeypatch.setenv("REX_SEARCH_PROVIDERS", "serpapi,duckduckgo")
    monkeypatch.setenv("SERPAPI_KEY", "")

    def fake_duckduckgo(self, query):
        queries.append(query)
        return "duck result"

    monkeypatch.setattr(WebSearchPlugin, "_search_duckduckgo", fake_duckduckgo, raising=False)

    result = plugin.process("search hi")
    plugin.shutdown()
    assert result == "duck result"
    assert queries == ["hi"]


def test_web_search_plugin_ignores_non_search_transcript(monkeypatch):
    plugin = WebSearchPlugin()
    queries = []

    monkeypatch.setenv("REX_SEARCH_PROVIDERS", "duckduckgo")

    def fake_duckduckgo(self, query):
        queries.append(query)
        return "duck result"

    monkeypatch.setattr(WebSearchPlugin, "_search_duckduckgo", fake_duckduckgo, raising=False)

    result = plugin.process("thank you very much")
    plugin.shutdown()
    assert result is None
    assert queries == []


def test_search_web_uses_raw_query(monkeypatch):
    import plugins.web_search as web_search

    calls = []

    class FakePlugin:
        def search(self, query):
            calls.append(query)
            return "raw result"

    monkeypatch.setattr(web_search, "_get_plugin", lambda: FakePlugin())

    assert web_search.search_web("python news") == "raw result"
    assert calls == ["python news"]


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
