"""Web search plugin with multiple API backends and graceful fallbacks."""

from __future__ import annotations

import logging
import os
from urllib.parse import quote_plus

from rex.config import settings
from rex.plugins import Plugin

try:
    import requests
    from requests.adapters import HTTPAdapter, Retry
except ImportError as e:
    raise RuntimeError(
        "Install with: pip install requests  (or add to requirements.txt and reinstall)"
    ) from e

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

logger = logging.getLogger(__name__)

# API Endpoints
SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"
BROWSERLESS_URL = "https://chrome.browserless.io/content"


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class WebSearchPlugin:
    name = "web_search"

    def __init__(self) -> None:
        self._session = _create_session()

    def initialize(self) -> None:
        pass

    def shutdown(self) -> None:
        self._session.close()

    def process(self, query: str) -> str | None:
        for provider in self._provider_order():
            method = getattr(self, f"_search_{provider}", None)
            if method:
                try:
                    result = method(query)
                    if result:
                        return result
                except Exception as e:
                    logger.warning("Provider '%s' failed: %s", provider, e)
        logger.warning("All search providers failed")
        return None

    def _provider_order(self) -> list[str]:
        return [p.strip() for p in settings.search_providers.split(",") if p.strip()]

    def _format_result(self, title: str, url: str, snippet: str) -> str:
        return f"{title} - {url}\n{snippet}"

    def _search_serpapi(self, query: str) -> str | None:
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return None
        params = {
            "q": query,
            "api_key": api_key,
            "num": "3",
            "engine": os.getenv("SERPAPI_ENGINE", "google"),
        }
        headers = {"X-Serpapi-Privacy": "true"}
        try:
            resp = self._session.get(SERPAPI_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("organic_results", [])
            if not results:
                return None
            top = results[0]
            return self._format_result(top["title"], top["link"], top.get("snippet", ""))
        except Exception as e:
            logger.warning("SerpAPI search failed: %s", e)
            return None

    def _search_brave(self, query: str) -> str | None:
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            return None
        headers = {"X-Subscription-Token": api_key}
        params = {"q": query, "count": 3}
        try:
            resp = self._session.get(BRAVE_URL, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("web", {}).get("results", [])
            if not results:
                return None
            top = results[0]
            return self._format_result(top["title"], top["url"], top.get("description", ""))
        except Exception as e:
            logger.warning("Brave search failed: %s", e)
            return None

    def _search_duckduckgo(self, query: str) -> str | None:
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup is required for DuckDuckGo scraping")
            return None
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = self._session.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            result = soup.find("a", class_="result__a")
            snippet = soup.find("a", class_="result__snippet")
            if result:
                return self._format_result(
                    result.text, result["href"], snippet.text if snippet else ""
                )
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return None

    def _search_google(self, query: str) -> str | None:
        api_key = os.getenv("GOOGLE_API_KEY")
        engine_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key or not engine_id:
            return None
        params = {"q": query, "key": api_key, "cx": engine_id, "num": 3}
        try:
            resp = self._session.get(GOOGLE_URL, params=params, timeout=10)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return None
            top = items[0]
            return self._format_result(top["title"], top["link"], top.get("snippet", ""))
        except Exception as e:
            logger.warning("Google CSE search failed: %s", e)
            return None

    def _search_browserless(self, query: str) -> str | None:
        token = os.getenv("BROWSERLESS_API_KEY")
        if not token or BeautifulSoup is None:
            return None
        payload = {
            "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
            "gotoOptions": {"waitUntil": "networkidle2"},
        }
        headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
        try:
            resp = self._session.post(
                BROWSERLESS_URL, headers=headers, params={"token": token}, json=payload, timeout=20
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            result = soup.find("a", class_="result__a")
            snippet = soup.find("a", class_="result__snippet")
            if result:
                return self._format_result(
                    result.text, result["href"], snippet.text if snippet else ""
                )
        except Exception as e:
            logger.warning("Browserless search failed: %s", e)
            return None


# Singleton for use across the app
_PLUGIN_SINGLETON: WebSearchPlugin | None = None


def _get_plugin() -> WebSearchPlugin:
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = WebSearchPlugin()
        _PLUGIN_SINGLETON.initialize()
    return _PLUGIN_SINGLETON


# --- Public API ---


def search_web(query: str) -> str | None:
    """Search the web using the configured fallback order."""
    return _get_plugin().process(query)


def search_serpapi(query: str) -> str | None:
    """Force a SerpAPI search (if available)."""
    return _get_plugin()._search_serpapi(query)


def search_duckduckgo(query: str) -> str | None:
    """Perform a DuckDuckGo HTML scrape."""
    return _get_plugin()._search_duckduckgo(query)


def register() -> Plugin:
    """Plugin entry point for dynamic loading."""
    return WebSearchPlugin()
