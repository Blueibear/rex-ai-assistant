from plugins.web_search import search_web

query = input("Enter your search query: ")
result = search_web(query)
print("\n--- Result ---")
print(result)
