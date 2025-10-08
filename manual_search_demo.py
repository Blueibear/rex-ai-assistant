"""Manual demo for the web search plugin."""

from __future__ import annotations
import argparse

from plugins.web_search import search_web


def run_search(query: str) -> None:
    if not query.strip():
        print("⚠️  Empty query. Please type something meaningful.")
        return

    try:
        result = search_web(query)
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return

    print("\n--- Result ---")
    if result:
        print(result)
    else:
        print("⚠️  No result found or search failed.")


def interactive_loop() -> None:
    print("🔎 Rex Web Search Plugin Demo (type 'exit' to quit)\n")
    while True:
        try:
            query = input("Enter your search query: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("👋 Exiting search demo.")
                break
            run_search(query)
        except KeyboardInterrupt:
            print("\n👋 Exiting search demo.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Rex Web Search CLI Tester")
    parser.add_argument("--query", "-q", type=str, help="Search query (non-interactive mode)")
    args = parser.parse_args()

    if args.query:
        run_search(args.query)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()

