"""Web search plugin with multiple API backends and graceful fallbacks."""

from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import quote_plus

from rex.config import settings
from rex.plugins import Plugin

try:  # pragma: no cover - optional dependency
    import requests
    from requests.adapters import HTTPAdapter, Retry
except ImportError:  # pragma: no cover - graceful degradation
    requests = None  # type: ignore[assignment]
    HTTPAdapter = Retry = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from duckduckgo_search import ddg
except ImportError:  # pragma: no cover - optional dependency
    ddg = None

try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - graceful degradation
    BeautifulSoup = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"
BROWSERLESS_URL = "https://chrome.browserless.io/content"


def _create_session() -> requests.Session:
    if requests is None:
        raise RuntimeError("requests must be installed for web search")

    session = requests.Session()
    if HTTPAdapter is not None and Retry is not None:
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


class WebSearchPlugin:
    name = "web_search"

    def __init__(self) -> None:
        self._session = _create_session()

    def initialize(self) -> None:  # pragma: no cover - no-op
        pass

    def shutdown(self) -> None:
        self._session.close()

    def process(self, query: str) -> Optional[str]:
        for provider in self._provider_order():
            handler = getattr(self, f"_search_{provider}", None)
            if handler is None:
                logger.warning("Unknown search provider '%s' - skipping", provider)
                continue
            try:
                result = handler(query)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Search provider '%s' failed: %s", provider, exc)
                continue
            if result:
                return result
        logger.warning("All search providers failed")
        return None

    def _provider_order(self) -> list[str]:
        providers = settings.search_providers
        return [provider.strip() for provider in providers.split(",") if provider.strip()]

    def _format_result(self, title: str, url: str, snippet: str) -> str:
        return f"{title} - {url}\n{snippet}"

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
        return self._format_result(top.get("title", ""), top.get("link", ""), top.get("snippet", ""))

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
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("Brave search failed: %s", exc)
            return None
        results = data.get("web", {}).get("results", [])
        if not results:
            return None
        top = results[0]
        return self._format_result(top.get("title", ""), top.get("url", ""), top.get("description", ""))

    def _search_duckduckgo(self, query: str) -> Optional[str]:
        if ddg is not None:
            try:
                results = ddg(query, max_results=3) or []
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("DuckDuckGo API failed: %s", exc)
            else:
                if results:
                    top = results[0]
                    return self._format_result(
                        top.get("title", "DuckDuckGo Result"),
                        top.get("href", ""),
                        top.get("body", ""),
                    )

        if BeautifulSoup is None:
            logger.warning("BeautifulSoup is required for DuckDuckGo scraping")
            return None

        encoded = requests.utils.quote(query) if requests is not None else quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = self._session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("DuckDuckGo search failed: %s", exc)
            return None

        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return self._format_result(result.text, result["href"], snippet.text if snippet else "")
        return None

    def _search_google(self, query: str) -> Optional[str]:
        api_key = os.getenv("GOOGLE_API_KEY")
        engine_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key or not engine_id:
            return None
        params = {"q": query, "key": api_key, "cx": engine_id, "num": 3}
        try:
            response = self._session.get(GOOGLE_URL, params=params, timeout=10)
            response.raise_for_status()
            items = response.json().get("items", [])
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("Google CSE search failed: %s", exc)
            return None
        if not items:
            return None
        top = items[0]
        return self._format_result(top.get("title", ""), top.get("link", ""), top.get("snippet", ""))

    def _search_browserless(self, query: str) -> Optional[str]:
        token = os.getenv("BROWSERLESS_API_KEY")
        if not token or BeautifulSoup is None:
            return None
        payload = {
            "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
            "gotoOptions": {"waitUntil": "networkidle2"},
        }
        headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
        try:
            response = self._session.post(
                BROWSERLESS_URL,
                headers=headers,
                params={"token": token},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("Browserless search failed: %s", exc)
            return None
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return self._format_result(result.text, result["href"], snippet.text if snippet else "")
        return None


_PLUGIN_SINGLETON: WebSearchPlugin | None = None


def _get_plugin() -> WebSearchPlugin:
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = WebSearchPlugin()
        _PLUGIN_SINGLETON.initialize()
    return _PLUGIN_SINGLETON


def search_serpapi(query: str) -> Optional[str]:
    """Force a SerpAPI search (if available)."""

    return _get_plugin()._search_serpapi(query)


def search_duckduckgo(query: str) -> Optional[str]:
    """Perform a DuckDuckGo HTML scrape."""

    return _get_plugin()._search_duckduckgo(query)


def search_web(query: str) -> Optional[str]:
    """Search the web using the configured fallback order."""

    return _get_plugin().process(query)


def register() -> Plugin:
    """Plugin entry point for dynamic loading."""

    return WebSearchPlugin()
