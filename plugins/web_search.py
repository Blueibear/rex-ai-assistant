"""Search plugin implementing multiple provider fallbacks."""

from __future__ import annotations

import os
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

from rex import settings
from rex.logging_utils import configure_logger
from rex.plugins.base import Plugin, PluginContext

LOGGER = configure_logger(__name__)

SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"
BROWSERLESS_URL = "https://chrome.browserless.io/content"


def _search_brave(query: str) -> Optional[str]:
    api_key = settings.brave_api_key or os.getenv("BRAVE_API_KEY")
    if not api_key:
        return None
    headers = {"X-Subscription-Token": api_key}
    params = {"q": query, "count": 3}
    try:
        response = requests.get(BRAVE_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        first = (data.get("web", {}) or {}).get("results", [])
        if not first:
            return None
        top = first[0]
        title = top.get("title", "Brave result")
        url = top.get("url", "")
        snippet = top.get("description", "")
        return f"{title} - {url}\n{snippet}"
    except Exception as exc:  # pragma: no cover - external HTTP
        LOGGER.warning("Brave search failed: %s", exc)
        return None


def _search_google(query: str) -> Optional[str]:
    api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
    engine_id = settings.google_cse_id or os.getenv("GOOGLE_CSE_ID")
    if not api_key or not engine_id:
        return None
    params = {"key": api_key, "cx": engine_id, "q": query, "num": 3}
    try:
        response = requests.get(GOOGLE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return None
        top = items[0]
        return f"{top.get('title')} - {top.get('link')}\n{top.get('snippet', '')}"
    except Exception as exc:  # pragma: no cover - external HTTP
        LOGGER.warning("Google search failed: %s", exc)
        return None


def _search_serpapi(query: str) -> Optional[str]:
    api_key = settings.serpapi_key or os.getenv("SERPAPI_KEY")
    if not api_key:
        return None
    params = {"q": query, "api_key": api_key, "num": "3"}
    try:
        response = requests.get(SERPAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", [])
        if not results:
            return None
        top = results[0]
        return f"{top.get('title')} - {top.get('link')}\n{top.get('snippet', '')}"
    except Exception as exc:  # pragma: no cover - external HTTP
        LOGGER.warning("SerpAPI search failed: %s", exc)
        return None


def _search_browserless(query: str) -> Optional[str]:
    token = settings.browserless_api_key or os.getenv("BROWSERLESS_API_KEY")
    if not token:
        return None
    payload = {
        "url": f"https://duckduckgo.com/?q={requests.utils.quote(query)}",
        "gotoOptions": {"waitUntil": "networkidle2"},
    }
    headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
    params = {"token": token}
    try:
        response = requests.post(
            BROWSERLESS_URL, params=params, json=payload, headers=headers, timeout=20
        )
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return f"{result.text} - {result['href']}\n{snippet.text if snippet else ''}"
    except Exception as exc:  # pragma: no cover - external HTTP
        LOGGER.warning("Browserless fallback failed: %s", exc)
    return None


def _search_duckduckgo(query: str) -> str:
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return f"{result.text} - {result['href']}\n{snippet.text if snippet else ''}"
        return "No result found."
    except Exception as exc:  # pragma: no cover - external HTTP
        LOGGER.error("DuckDuckGo scraping failed: %s", exc)
        return "Search failed."


class WebSearchPlugin:
    name = "web_search"

    def __init__(self) -> None:
        self._backends: tuple[Callable[[str], Optional[str]], ...] = (
            _search_brave,
            _search_google,
            _search_serpapi,
            _search_browserless,
        )

    def initialise(self) -> None:
        LOGGER.info("WebSearch plugin initialised")

    def process(self, context: PluginContext) -> str | None:
        for backend in self._backends:
            result = backend(context.text)
            if result:
                return result
        return _search_duckduckgo(context.text)

    def shutdown(self) -> None:
        LOGGER.info("WebSearch plugin shutting down")


PLUGIN: Plugin = WebSearchPlugin()
