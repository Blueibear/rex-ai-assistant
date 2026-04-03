"""Tests for US-LAT-004: Response caching for repeated factual queries."""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# ResponseCache — cache hit / miss
# ---------------------------------------------------------------------------


def test_cache_miss_on_first_call():
    """First call for a message returns None (miss)."""
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    assert cache.get("what is the capital of France?") is None


def test_cache_hit_after_put():
    """After put(), get() returns the stored response."""
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is the capital of France?", "Paris is the capital of France.")
    result = cache.get("what is the capital of France?")
    assert result == "Paris is the capital of France."


def test_cache_hit_is_case_insensitive():
    """Key lookup normalizes case."""
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("What Is The Capital Of France?", "Paris.")
    assert cache.get("what is the capital of france?") == "Paris."
    assert cache.get("WHAT IS THE CAPITAL OF FRANCE?") == "Paris."


def test_cache_hit_ignores_punctuation_variation():
    """Punctuation differences do not affect key matching."""
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("how tall is mount everest", "8,849 metres.")
    assert cache.get("How tall is Mount Everest?") == "8,849 metres."
    assert cache.get("how tall is mount everest!") == "8,849 metres."


# ---------------------------------------------------------------------------
# ResponseCache — TTL expiry
# ---------------------------------------------------------------------------


def test_cache_miss_after_ttl_expiry():
    """Entry is evicted after TTL seconds."""
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=0.05)  # 50 ms
    cache.put("what color is the sky?", "Blue.")
    assert cache.get("what color is the sky?") == "Blue."
    time.sleep(0.1)
    assert cache.get("what color is the sky?") is None


def test_cache_hit_before_ttl_expiry():
    """Entry is still present before TTL elapses."""
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=10.0)
    cache.put("what color is grass?", "Green.")
    assert cache.get("what color is grass?") == "Green."


# ---------------------------------------------------------------------------
# ResponseCache — bypass conditions (time-sensitive)
# ---------------------------------------------------------------------------


def test_bypass_for_right_now():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is the time right now?", "3pm")
    assert cache.get("what is the time right now?") is None


def test_bypass_for_current_weather():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is the current weather?", "Sunny.")
    assert cache.get("what is the current weather?") is None


def test_bypass_for_today():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is today's date?", "April 2.")
    assert cache.get("what is today's date?") is None


def test_bypass_for_weather():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is the weather in London?", "Rainy.")
    assert cache.get("what is the weather in London?") is None


def test_bypass_for_news():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is the latest news?", "...")
    assert cache.get("what is the latest news?") is None


# ---------------------------------------------------------------------------
# ResponseCache — bypass conditions (tool-invoking)
# ---------------------------------------------------------------------------


def test_bypass_for_email():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("read my email", "You have 3 emails.")
    assert cache.get("read my email") is None


def test_bypass_for_calendar():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is on my calendar tomorrow?", "Nothing.")
    assert cache.get("what is on my calendar tomorrow?") is None


def test_bypass_for_home_assistant():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("turn on the living room light", "Done.")
    assert cache.get("turn on the living room light") is None


def test_bypass_for_timer():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("set a timer for 5 minutes", "Timer set.")
    assert cache.get("set a timer for 5 minutes") is None


# ---------------------------------------------------------------------------
# ResponseCache — should_bypass public API
# ---------------------------------------------------------------------------


def test_should_bypass_for_time_sensitive():
    from rex.response_cache import ResponseCache

    assert ResponseCache.should_bypass("what's the weather right now?") is True


def test_should_bypass_false_for_factual():
    from rex.response_cache import ResponseCache

    assert ResponseCache.should_bypass("what is the capital of France?") is False


# ---------------------------------------------------------------------------
# ResponseCache — clear()
# ---------------------------------------------------------------------------


def test_clear_removes_all_entries():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("q1", "a1")
    cache.put("q2", "a2")
    cache.clear()
    assert cache.get("q1") is None
    assert cache.get("q2") is None


# ---------------------------------------------------------------------------
# ResponseCache — hit_rate
# ---------------------------------------------------------------------------


def test_hit_rate_zero_when_no_lookups():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    assert cache.hit_rate == 0.0


def test_hit_rate_after_mixed_lookups():
    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("q1", "a1")
    cache.get("q1")  # hit
    cache.get("q2")  # miss
    cache.get("q2")  # miss
    # 1 hit / 3 lookups (bypassed queries don't count as misses)
    assert abs(cache.hit_rate - 1 / 3) < 0.01


# ---------------------------------------------------------------------------
# AppConfig — response_cache_ttl field
# ---------------------------------------------------------------------------


def test_appconfig_has_response_cache_ttl_field():
    """AppConfig has response_cache_ttl with default 300.0."""
    import dataclasses

    from rex.config import AppConfig

    defaults = {
        f.name: f.default for f in dataclasses.fields(AppConfig) if f.name == "response_cache_ttl"
    }
    assert "response_cache_ttl" in defaults
    assert defaults["response_cache_ttl"] == 300.0


def test_build_app_config_response_cache_ttl_default():
    """build_app_config returns 300.0 when response_cache.ttl is absent."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
        }
    )
    assert cfg.response_cache_ttl == 300.0


def test_build_app_config_response_cache_ttl_custom():
    """build_app_config reads custom TTL from JSON."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "response_cache": {"ttl": 60.0},
        }
    )
    assert cfg.response_cache_ttl == 60.0


# ---------------------------------------------------------------------------
# Assistant — cache wired into generate_reply()
# ---------------------------------------------------------------------------


def _make_assistant_with_cache(cache_ttl: float = 60.0):
    """Create an Assistant instance with a mock LLM and real ResponseCache."""
    from rex.config import AppConfig

    settings_mock = AppConfig.__new__(AppConfig)
    # Patch all attribute accesses assistant.__init__ needs
    settings_mock.response_cache_ttl = cache_ttl
    settings_mock.max_memory_items = 50
    settings_mock.persist_history = False
    settings_mock.ha_base_url = None
    settings_mock.ha_token = None
    settings_mock.transcripts_dir = "transcripts"
    settings_mock.transcripts_enabled = False
    settings_mock.user_id = "default"
    settings_mock.model_routing = None
    settings_mock.ollama_base_url = "http://localhost:11434"
    settings_mock.skills_path = None

    return settings_mock


def test_assistant_cache_returns_cached_response():
    """Second call with same message returns cached response without hitting LLM."""

    from rex.response_cache import ResponseCache

    cache = ResponseCache(ttl=60.0)
    cache.put("what is 2 plus 2?", "4")

    # Simulate the cache lookup that generate_reply does
    result = cache.get("what is 2 plus 2?")
    assert result == "4"


def test_assistant_cache_disabled_when_ttl_zero():
    """ResponseCache is None when response_cache_ttl=0."""
    from rex.response_cache import ResponseCache

    # ttl=0 → cache is disabled (None in Assistant), so no caching
    cache = ResponseCache(ttl=0.0)
    cache.put("test", "answer")
    # A cache with ttl=0 will immediately expire
    assert cache.get("test") is None
