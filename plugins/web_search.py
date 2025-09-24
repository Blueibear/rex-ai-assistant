"""Web search plugin with multiple API backends and graceful fallbacks."""

from __future__ import annotations

import os
from typing import Optional, Callable
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

from config import load_config
from logging_utils import get_logger

LOGGER = get_logger(__name__)
CONFIG = load_config()

# === Search API URLs ===
SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"
BROWSERLESS_URL = "https://chrome.browserless.io/content"

# === Session with retry strategy ===
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)


# === Result formatting ===
def format_result(title: str, url: str, snippet: str) -> str:
    return f"{title} - {url}\n{snippet}"


# === Backends ===
def search_brave(query: str) -> Optional[str]:
    api_key = CONFIG.brave_api_key or os.getenv("BRAVE_API_KEY")
    if not api_key:
        return None
    headers = {"X-Subscription-Token": api_key}
    params = {"q": query, "count": 3}
    try:
        response = session.get(BRAVE_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        top = (data.get("web") or {}).get("results", [])
        if not top:
            return None
        return format_result(
            top[0].get("title", "Brave result"),
            top[0].get("url", ""),
            top[0].get("description", "")
        )
    except Exception as exc:
        LOGGER.warning("Brave search failed: %s", exc)
        return None


def search_google(query: str) -> Optional[str]:
    api_key = CONFIG.google_api_key or os.getenv("GOOGLE_API_KEY")
    engine_id = CONFIG.google_cse_id or os.getenv("GOOGLE_CSE_ID")
    if not api_key or not engine_id:
        return None
    params = {"key": api_key, "cx": engine_id, "q": query, "num": 3}
    try:
        response = session.get(GOOGLE_URL, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            return None
        return format_result(
            items[0].get("title", "Google result"),
            items[0].get("link", ""),
            items[0].get("snippet", "")
        )
    except Exception as exc:
        LOGGER.warning("Google search failed: %s", exc)
        return None


def search_serpapi(query: str) -> Optional[str]:
    api_key = CONFIG.serpapi_key or os.getenv("SERPAPI_KEY")
    if not api_key:
        return None
    params = {"q": query, "api_key": api_key, "num": "3"}
    try:
        response = session.get(SERPAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        if not results:
            return None
        return format_result(
            results[0].get("title", "SerpAPI result"),
            results[0].get("link", ""),
            results[0].get("snippet", "")
        )
    except Exception as exc:
        LOGGER.warning("SerpAPI search failed: %s", exc)
        return None


def search_browserless(query: str) -> Optional[str]:
    token = CONFIG.browserless_key or os.getenv("BROWSERLESS_API_KEY")
    if not token:
        return None
    payload = {
        "url": f"https://duckduckgo.com/?q={requests.utils.quote(query)}",
        "gotoOptions": {"waitUntil": "networkidle2"},
    }
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json"
    }
    params = {"token": token}
    try:
        response = session.post(
            BROWSERLESS_URL,
            params=params,
            json=payload,
            headers=headers,
            timeout=20
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return format_result(result.text, result["href"], snippet.text if snippet else "")
    except Exception as exc:
        LOGGER.warning("Browserless fallback failed: %s", exc)
    return None


def search_duckduckgo(query: str) -> str:
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = session.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return format_result(result.text, result["href"], snippet.text if snippet else "")
        return "No result found."
    except Exception as exc:
        LOGGER.error("DuckDuckGo scraping failed: %s", exc)
        return "Search failed."


def search_web(query: str) -> str:
    backends: list[Callable[[str], Optional[str]]] = [
        search_brave,
        search_google,
        search_serpapi,
        search_browserless,
    ]
    for backend in backends:
        result = backend(query)
        if result:
            LOGGER.info("Search succeeded via %s", backend.__name__)
            return result
    LOGGER.warning("All structured search backends failed. Using DuckDuckGo fallback.")
    return search_duckduckgo(query)


def register() -> dict:
    return {"search": search_web}
