"""Web search plugin with multiple API backends and graceful fallbacks."""

from __future__ import annotations

import os
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import load_config
from logging_utils import get_logger

LOGGER = get_logger(__name__)
CONFIG = load_config()

SERPAPI_URL = "https://serpapi.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"
BROWSERLESS_URL = "https://chrome.browserless.io/content"


def search_brave(query: str) -> Optional[str]:
    api_key = CONFIG.brave_api_key or os.getenv("BRAVE_API_KEY")
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
    except Exception as exc:
        LOGGER.warning("Brave search failed: %s", exc)
        return None


def search_google(query: str) -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    engine_id = os.getenv("GOOGLE_CSE_ID")
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
    except Exception as exc:
        LOGGER.warning("Google search failed: %s", exc)
        return None


def search_serpapi(query: str) -> Optional[str]:
    api_key = os.getenv("SERPAPI_KEY")
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
    except Exception as exc:
        LOGGER.warning("SerpAPI search failed: %s", exc)
        return None


def search_browserless(query: str) -> Optional[str]:
    token = os.getenv("BROWSERLESS_API_KEY")
    if not token:
        return None
    payload = {
        "url": f"https://duckduckgo.com/?q={requests.utils.quote(query)}",
        "gotoOptions": {"waitUntil": "networkidle2"},
    }
    headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
    params = {"token": token}
    try:
        response = requests.post(BROWSERLESS_URL, params=params, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return f"{result.text} - {result['href']}\n{snippet.text if snippet else ''}"
    except Exception as exc:
        LOGGER.warning("Browserless fallback failed: %s", exc)
    return None


def search_duckduckgo(query: str) -> str:
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return f"{result.text} - {result['href']}\n{snippet.text if snippet else ''}"
        return "No result found."
    except Exception as exc:
        LOGGER.error("DuckDuckGo scraping failed: %s", exc)
        return "Search failed."


def search_web(query: str) -> str:
    for backend in (search_brave, search_google, search_serpapi, search_browserless):
        result = backend(query)
        if result:
            return result
    return search_duckduckgo(query)


def register() -> dict:
    return {"search": search_web}
