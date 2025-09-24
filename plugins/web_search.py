"""Web search plugin with multiple API backends and graceful fallbacks."""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional
from urllib.parse import quote_plus

try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter, Retry
except ImportError:
    requests = None
    BeautifulSoup = None

from config import load_config
from logging_utils import get_logger
from rex.plugins import Plugin
from rex.config import settings  # for search_providers list

logger = get_logger(__name__)
config = load_config()

# === API URLs ===
SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"
BROWSERLESS_URL = "https://chrome.browserless.io/content"


# === Retry-enabled session ===
def _init_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess if requests is not None else None


def format_result(title: str, url: str, snippet: str) -> str:
    return f"{title} - {url}\n{snippet}"


class WebSearchPlugin:
    name = "web_search"

    def __init__(self) -> None:
        self._session = _init_session()
        self._provider_map: dict[str, Callable[[str], Optional[str]]] = {
            "brave": self._search_brave,
            "google": self._search_google,
            "serpapi": self._search_serpapi,
            "browserless": self._search_browserless,
            "duckduckgo": self._search_duckduckgo,
        }

    def initialize(self) -> None:
        logger.info("WebSearchPlugin initialized")

    def shutdown(self) -> None:
        self._session.close() if self._session else None

    def process(self, query: str) -> Optional[str]:
        for provider in self._provider_order():
            handler = self._provider_map.get(provider)
            if not handler:
                continue
            result = handler(query)
            if result:
                logger.info("Search succeeded via %s", provider)
                return result
        logger.warning("All search backends failed. Using DuckDuckGo fallback.")
        return self._search_duckduckgo(query)

    def _provider_order(self) -> list[str]:
        return [p.strip() for p in settings.search_providers.split(",") if p.strip()]

    def _search_brave(self, query: str) -> Optional[str]:
        api_key = config.brave_api_key or os.getenv("BRAVE_API_KEY")
        if not api_key or not self._session:
            return None
        try:
            headers = {"X-Subscription-Token": api_key}
            params = {"q": query, "count": 3}
            r = self._session.get(BRAVE_URL, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            top = (data.get("web") or {}).get("results", [])
            if top:
                return format_result(top[0].get("title", "Brave result"), top[0].get("url", ""), top[0].get("description", ""))
        except Exception as exc:
            logger.warning("Brave search failed: %s", exc)
        return None

    def _search_google(self, query: str) -> Optional[str]:
        api_key = config.google_api_key or os.getenv("GOOGLE_API_KEY")
        cse_id = config.google_cse_id or os.getenv("GOOGLE_CSE_ID")
        if not api_key or not cse_id or not self._session:
            return None
        try:
            params = {"key": api_key, "cx": cse_id, "q": query, "num": 3}
            r = self._session.get(GOOGLE_URL, params=params, timeout=10)
            r.raise_for_status()
            items = r.json().get("items", [])
            if items:
                return format_result(items[0].get("title", "Google result"), items[0].get("link", ""), items[0].get("snippet", ""))
        except Exception as exc:
            logger.warning("Google search failed: %s", exc)
        return None

    def _search_serpapi(self, query: str) -> Optional[str]:
        api_key = config.serpapi_key or os.getenv("SERPAPI_KEY")
        if not api_key or not self._session:
            return None
        try:
            params = {"q": query, "api_key": api_key, "engine": "google", "num": "3"}
            headers = {"X-Serpapi-Privacy": "true"}
            r = self._session.get(SERPAPI_URL, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            results = r.json().get("organic_results", [])
            if results:
                return format_result(results[0].get("title", "SerpAPI result"), results[0].get("link", ""), results[0].get("snippet", ""))
        except Exception as exc:
            logger.warning("SerpAPI search failed: %s", exc)
        return None

    def _search_browserless(self, query: str) -> Optional[str]:
        token = config.browserless_key or os.getenv("BROWSERLESS_API_KEY")
        if not token or not self._session:
            return None
        try:
            payload = {
                "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
                "gotoOptions": {"waitUntil": "networkidle2"},
            }
            headers = {
                "Cache-Control": "no-cache",
                "Content-Type": "application/json"
            }
            r = self._session.post(BROWSERLESS_URL, json=payload, headers=headers, params={"token": token}, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser") if BeautifulSoup else None
            if not soup:
                return None
            result = soup.find("a", class_="result__a")
            snippet = soup.find("a", class_="result__snippet")
            if result:
                return format_result(result.text, result["href"], snippet.text if snippet else "")
        except Exception as exc:
            logger.warning("Browserless fallback failed: %s", exc)
        return None

    def _search_duckduckgo(self, query: str) -> Optional[str]:
        if not self._session or not BeautifulSoup:
            return None
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = self._session.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            result = soup.find("a", class_="result__a")
            snippet = soup.find("a", class_="result__snippet")
            if result:
                return format_result(result.text, result["href"], snippet.text if snippet else "")
        except Exception as exc:
            logger.warning("DuckDuckGo scraping failed: %s", exc)
        return None


# === Plugin entrypoints ===

_PLUGIN_SINGLETON: WebSearchPlugin | None = None


def _get_plugin() -> WebSearchPlugin:
    global _PLUGIN_SINGLETON
    if _PLUGIN_SINGLETON is None:
        _PLUGIN_SINGLETON = WebSearchPlugin()
        _PLUGIN_SINGLETON.initialize()
    return _PLUGIN_SINGLETON


def search_web(query: str) -> Optional[str]:
    return _get_plugin().process(query)


def register() -> Plugin:
    return WebSearchPlugin()

