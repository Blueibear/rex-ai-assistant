import os
import json
from flask import Flask, request, abort, jsonify
from plugins.web_search import search_web

app = Flask(__name__)

# Load user map from users.json
with open("users.json", "r") as f:
    users = json.load(f)

@app.before_request
def load_user_memory():
    global user_key, memory, user_folder

    email = request.headers.get("Cf-Access-Authenticated-User-Email")
    if not email:
        abort(403, description="No authenticated email provided.")

    email = email.lower()
    user_key = users.get(email)

    if not user_key:
        abort(403, description="Access denied for this user.")

    user_folder = os.path.join("memory", user_key)
    try:
        with open(os.path.join(user_folder, "core.json"), "r") as f:
            memory = json.load(f)
    except FileNotFoundError:
        abort(500, description="Memory file not found for this user.")

# ✅ Root route: status check
@app.route("/")
def index():
    return "🧠 Rex is online. Ask away."

# ✅ Whoami route: returns active memory profile
@app.route("/whoami")
def whoami():
    return jsonify({
        "user": user_key,
        "memory": memory
    })

# ✅ Search route: performs live web search
@app.route("/search")
def search():
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    result = search_web(query)
    return jsonify({
        "query": query,
        "result": result
    })

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")


