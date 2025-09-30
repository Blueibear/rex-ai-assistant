"""Web search helpers and plugin registration."""

from __future__ import annotations

import logging
import os
from typing import Optional

from urllib.parse import quote_plus

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - minimal shim
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - graceful degradation
    BeautifulSoup = None  # type: ignore[assignment]

from rex.config import settings
from rex.plugins import Plugin

logger = logging.getLogger(__name__)
SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class _DummySession:
    def get(self, *_args, **_kwargs):  # pragma: no cover - defensive
        raise RuntimeError("requests is not installed")

    def close(self) -> None:  # pragma: no cover - defensive
        pass


class WebSearchPlugin:
    name = "web_search"

    def __init__(self) -> None:
        self._session = requests.Session() if requests is not None else _DummySession()

    def initialize(self) -> None:  # pragma: no cover - no-op
        pass

    def shutdown(self) -> None:
        self._session.close()

    def process(self, query: str) -> Optional[str]:
        for provider in self._provider_order():
            handler = getattr(self, f"_search_{provider}", None)
            if handler is None:
                logger.warning("Unknown search provider '%s' – skipping", provider)
                continue
            try:
                result = handler(query)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Search provider '%s' failed: %s", provider, exc)
                continue
            if result:
                return result
        return None

    def _provider_order(self) -> list[str]:
        providers = settings.search_providers
        return [provider.strip() for provider in providers.split(",") if provider.strip()]

    def _search_serpapi(self, query: str) -> Optional[str]:
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return None
        params = {
            "q": query,
            "api_key": api_key,
            "engine": os.getenv("SERPAPI_ENGINE", "google") or "google",
            "num": "3",
        }
        headers = {"X-Serpapi-Privacy": "true"}
        try:
            response = self._session.get(SERPAPI_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("SerpAPI search failed: %s", exc)
            return None
        results = data.get("organic_results", [])
        if not results:
            return None
        top = results[0]
        return f"{top.get('title')} - {top.get('link')}\n{top.get('snippet', '')}"

    def _search_brave(self, query: str) -> Optional[str]:
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            return None
        headers = {"X-Subscription-Token": api_key}
        params = {"q": query, "count": 3}
        try:
            response = self._session.get(BRAVE_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover
            logger.warning("Brave search failed: %s", exc)
            return None
        web = data.get("web", {})
        results = web.get("results", [])
        if not results:
            return None
        top = results[0]
        return f"{top.get('title')} - {top.get('url')}\n{top.get('description', '')}"

    def _search_duckduckgo(self, query: str) -> Optional[str]:
        if requests is not None:
            encoded = requests.utils.quote(query)
        else:
            encoded = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup is required for DuckDuckGo scraping")
            return None
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = self._session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:  # pragma: no cover
            logger.warning("DuckDuckGo search failed: %s", exc)
            return None
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            snippet_text = snippet.text if snippet else ""
            return f"{result.text} - {result['href']}\n{snippet_text}"
        return None


_PLUGIN_SINGLETON: WebSearchPlugin | None = None


def _get_plugin() -> WebSearchPlugin:
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = WebSearchPlugin()
        _PLUGIN_SINGLETON.initialize()
    return _PLUGIN_SINGLETON


def search_serpapi(query: str) -> Optional[str]:
    """Compatibility wrapper returning the SerpAPI result when possible."""

    return _get_plugin()._search_serpapi(query)


def search_duckduckgo(query: str) -> Optional[str]:
    """Perform a DuckDuckGo HTML scrape, mirroring the legacy helper."""

    return _get_plugin()._search_duckduckgo(query)


def search_web(query: str) -> Optional[str]:
    """Return the first available result from the configured providers."""

    return _get_plugin().process(query)


def register() -> Plugin:
    """Entry point used by :func:`rex.plugins.load_plugins`."""

    return WebSearchPlugin()
