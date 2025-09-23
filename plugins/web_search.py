"""Web search plugin with provider fallbacks."""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests
from bs4 import BeautifulSoup

from rex.plugins import Plugin

logger = logging.getLogger(__name__)
SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchPlugin:
    name = "web_search"

    def __init__(self) -> None:
        self._session = requests.Session()

    def initialize(self) -> None:  # pragma: no cover - no-op
        pass

    def shutdown(self) -> None:
        self._session.close()

    def process(self, query: str) -> Optional[str]:
        for provider in self._provider_order():
            handler = getattr(self, f"_search_{provider}")
            result = handler(query)
            if result:
                return result
        return None

    def _provider_order(self) -> list[str]:
        providers = os.getenv("REX_SEARCH_PROVIDERS", "serpapi,brave,duckduckgo")
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
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
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


def register() -> Plugin:
    return WebSearchPlugin()
