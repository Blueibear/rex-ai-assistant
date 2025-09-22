"""Manual demo for the web search plugin."""

from __future__ import annotations

from plugins.web_search import search_web


def main() -> None:
    query = input("Enter your search query: ")
    result = search_web(query)
    print("\n--- Result ---")
    print(result)


if __name__ == "__main__":
    main()
