import os
import requests
from bs4 import BeautifulSoup

def search_serpapi(query):
    """Search using SerpAPI."""
    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    if not serpapi_key:
        print("[Search] SerpAPI key not found in environment variables.")
        return None

    print("[Search] Using SerpAPI...")
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": "3"
    }
    headers = {
        "X-Serpapi-Privacy": "true"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", [])
        if not results:
            return None

        top = results[0]
        return f"{top.get('title')} - {top.get('link')}\n{top.get('snippet')}"
    except Exception as e:
        print("[SerpAPI Error]:", str(e))
        return None

def search_duckduckgo(query):
    print("[Search] Falling back to DuckDuckGo...")
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("a", class_="result__a")
        snippet = soup.find("a", class_="result__snippet")
        if result:
            return f"{result.text} - {result['href']}\n{snippet.text if snippet else ''}"
        return "No result found."
    except Exception as e:
        print("[DuckDuckGo Error]:", str(e))
        return "Search failed."

def search_web(query):
    result = search_serpapi(query)
    if result:
        return result
    return search_duckduckgo(query)
