"""Configuration management for Rex using :mod:`pydantic` settings."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

try:  # pragma: no cover - exercised indirectly via import
    from pydantic import BaseSettings, Field, validator
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency unavailable
    from ._pydantic_stub import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application wide configuration.

    ``BaseSettings`` reads values from environment variables and optionally a
    ``.env`` file, giving us a central, validated configuration surface.  The
    defaults keep local development friction free while allowing production
    deployments to override what they need.
    """

    whisper_model: str = Field("base", description="Whisper model to load")
    temperature: float = Field(0.8, ge=0.0, le=1.0)
    user_id: str = Field("default", description="Default user profile key")
    llm_backend: str = Field("transformers", description="LLM backend strategy")
    llm_model: str = Field("distilgpt2", description="Model identifier for LLM backend")
    openai_api_key: str | None = Field(None, description="API key for OpenAI backend")
    wakeword: str = Field("rex", description="Wake word to activate the assistant")
    wakeword_threshold: float = Field(0.5, ge=0.0, le=1.0)
    command_duration: float = Field(4.0, gt=0.0)
    wake_sound_path: str | None = Field(None, description="Optional wake confirmation sound")
    enable_search_plugin: bool = Field(True, description="Enable the web search plugin")
    memory_max_turns: int = Field(50, gt=0, description="Number of conversation turns to retain")
    flask_rate_limit: str = Field("5 per minute", description="Rate limit for Flask endpoints")
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    log_file: Optional[str] = Field(None, description="Optional log file path")
    brave_api_key: str | None = None
    google_api_key: str | None = None
    google_cse_id: str | None = None
    serpapi_key: str | None = None
    browserless_api_key: str | None = None
    speak_api_key: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("llm_backend")
    def _normalise_backend(cls, value: str) -> str:
        return value.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()


# A module level singleton that mirrors the behaviour shown in the
# instructions.  ``settings`` is imported by most modules so memoisation keeps
# the configuration load cheap.
settings: Settings = get_settings()

__all__ = ["Settings", "settings", "get_settings"]
