from __future__ import annotations

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
