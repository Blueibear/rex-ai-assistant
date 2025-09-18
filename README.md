# 🧠 Rex AI Assistant

**Rex** is a locally running, voice-activated AI assistant designed for privacy, power, and personality. Built from scratch with modular Python components, Rex supports wake-word detection, Whisper STT, Coqui XTTS speech synthesis, and full memory-backed interactions — all without relying on the cloud.

---

## 💡 Features

- ✅ Wake-word detection using [OpenWakeWord](https://github.com/dscripka/openWakeWord)
- 🎙️ STT (Speech-to-Text) via [Whisper](https://github.com/openai/whisper)
- 🗣️ TTS (Text-to-Speech) via [Coqui XTTS v2](https://github.com/coqui-ai/TTS)
- 🧠 Personalized memory for multiple users
- 🌐 Optional web search plugin
- 🧰 Modular architecture for easy extension
- 🔐 Runs locally — no Big Brother required

---

## 🚀 Getting Started

### Requirements

- Python 3.10+
- Git
- A decent microphone + speakers
- A brain (optional but encouraged)

### 1. Clone the repo

```bash
git clone https://github.com/Blueibear/rex-ai-assistant.git
cd rex-ai-assistant

---

## ⚙️ Configuration

Rex reads several environment variables at start-up so you can tune
authentication, search plugins, and the default memory profile. Restart
the assistant or proxy after changing any of the values.

### Web search

- `SERPAPI_KEY` — enables the SerpAPI integration in
  `plugins/web_search.py`. When unset, Rex falls back to DuckDuckGo
  scraping.
- `SERPAPI_ENGINE` *(optional)* — overrides the SerpAPI engine (defaults
  to `google`).

### Proxy authentication

- `REX_PROXY_TOKEN` — shared-secret used by `flask_proxy.py`. Provide it
  via an `Authorization: Bearer <token>` header or the
  `X-Rex-Proxy-Token` header.
- `REX_PROXY_ALLOW_LOCAL` — set to `1` while debugging to allow
  loopback traffic without Cloudflare Access. Leave unset in
  production.

Requests that come through Cloudflare Access must continue to include
the `Cf-Access-Authenticated-User-Email` header so the proxy can pick
the correct memory profile.

### Active memory profile

- `REX_ACTIVE_USER` — selects the default user profile. Accepts a memory
  folder name (for example `james`) or an email address listed in
  `users.json`. The CLI assistant and Flask services load this profile
  on start-up.
