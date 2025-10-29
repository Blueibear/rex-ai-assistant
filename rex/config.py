"""Central configuration management for the Rex assistant."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Environment variables that must be provided for a functional deployment.
REQUIRED_ENV_KEYS: Sequence[str] = (
    "OPENAI_API_KEY",
)

# Optional dependencies
try:
    from dotenv import load_dotenv, set_key
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return False
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return True

    def set_key(dotenv_path: str, key: str, value: str) -> None:
        env_file = Path(dotenv_path)
        lines: list[str] = []
        if env_file.exists():
            lines = env_file.read_text(encoding="utf-8").splitlines()
        prefix = f"{key}="
        for index, line in enumerate(lines):
            if line.startswith(prefix):
                lines[index] = f"{prefix}{value}"
                break
        else:
            lines.append(f"{prefix}{value}")
        env_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

try:  # pragma: no cover - pydantic may be unavailable in slim environments
    from pydantic import BaseSettings, Field
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - fallback to dataclass implementation
    BaseSettings = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _HAS_PYDANTIC = False


_FIELD_DEFAULTS: Dict[str, Any] = {
    "whisper_model": "medium",
    "whisper_device": "cuda",
    "temperature": 0.8,
    "user_id": "default",
    "llm_model": "distilgpt2",
    "llm_backend": "transformers",
    "max_memory_items": 50,
    "wakeword_keyword": "hey_jarvis",
    "wakeword_threshold": 0.5,
    "sample_rate": 16000,
    "detection_frame_seconds": 0.5,
    "capture_seconds": 5.0,
    "wakeword_poll_interval": 0.05,
    "log_path": "logs/rex.log",
    "error_log_path": "logs/error.log",
    "transcripts_dir": "transcripts",
    "search_providers": "serpapi,brave,duckduckgo",
    "speak_language": "en",
    "input_device": None,
    "output_device": None,
}

_ENV_ALIASES: Dict[str, Sequence[str]] = {
    "whisper_model": ("WHISPER_MODEL", "REX_WHISPER_MODEL"),
    "whisper_device": ("WHISPER_DEVICE", "REX_WHISPER_DEVICE"),
    "temperature": ("REX_LLM_TEMPERATURE",),
    "user_id": ("REX_ACTIVE_USER",),
    "llm_model": ("REX_LLM_MODEL",),
    "llm_backend": ("REX_LLM_BACKEND",),
    "max_memory_items": ("REX_MEMORY_MAX_ITEMS",),
    "wakeword_keyword": ("REX_WAKEWORD_KEYWORD",),
    "wakeword_threshold": ("REX_WAKEWORD_THRESHOLD",),
    "sample_rate": ("REX_SAMPLE_RATE",),
    "detection_frame_seconds": ("REX_DETECTION_FRAME_SECONDS",),
    "capture_seconds": ("REX_CAPTURE_SECONDS",),
    "wakeword_poll_interval": ("REX_WAKEWORD_POLL_INTERVAL",),
    "log_path": ("REX_LOG_PATH",),
    "error_log_path": ("REX_ERROR_LOG_PATH",),
    "transcripts_dir": ("REX_TRANSCRIPTS_DIR",),
    "search_providers": ("REX_SEARCH_PROVIDERS",),
    "speak_language": ("REX_SPEAK_LANGUAGE",),
    "input_device": ("REX_INPUT_DEVICE",),
    "output_device": ("REX_OUTPUT_DEVICE",),
}


def _cast_value(key: str, raw: str) -> Any:
    if key in {"temperature", "wakeword_threshold", "detection_frame_seconds", "capture_seconds", "wakeword_poll_interval"}:
        return float(raw)
    if key in {"max_memory_items", "sample_rate"}:
        return int(raw)
    if key in {"input_device", "output_device"}:
        return int(raw) if raw not in {"", "none", "None"} else None
    return raw


if _HAS_PYDANTIC:

    class Settings(BaseSettings):
        """Typed configuration loaded from environment variables."""

        whisper_model: str = Field(_FIELD_DEFAULTS["whisper_model"], env=["WHISPER_MODEL", "REX_WHISPER_MODEL"])
        whisper_device: str = Field(_FIELD_DEFAULTS["whisper_device"], env=["WHISPER_DEVICE", "REX_WHISPER_DEVICE"])
        temperature: float = Field(_FIELD_DEFAULTS["temperature"], env="REX_LLM_TEMPERATURE")
        user_id: str = Field(_FIELD_DEFAULTS["user_id"], env="REX_ACTIVE_USER")
        llm_model: str = Field(_FIELD_DEFAULTS["llm_model"], env="REX_LLM_MODEL")
        llm_backend: str = Field(_FIELD_DEFAULTS["llm_backend"], env="REX_LLM_BACKEND")
        max_memory_items: int = Field(_FIELD_DEFAULTS["max_memory_items"], env="REX_MEMORY_MAX_ITEMS")
        wakeword_keyword: str = Field(_FIELD_DEFAULTS["wakeword_keyword"], env="REX_WAKEWORD_KEYWORD")
        wakeword_threshold: float = Field(_FIELD_DEFAULTS["wakeword_threshold"], env="REX_WAKEWORD_THRESHOLD")
        sample_rate: int = Field(_FIELD_DEFAULTS["sample_rate"], env="REX_SAMPLE_RATE")
        detection_frame_seconds: float = Field(_FIELD_DEFAULTS["detection_frame_seconds"], env="REX_DETECTION_FRAME_SECONDS")
        capture_seconds: float = Field(_FIELD_DEFAULTS["capture_seconds"], env="REX_CAPTURE_SECONDS")
        wakeword_poll_interval: float = Field(_FIELD_DEFAULTS["wakeword_poll_interval"], env="REX_WAKEWORD_POLL_INTERVAL")
        log_path: str = Field(_FIELD_DEFAULTS["log_path"], env="REX_LOG_PATH")
        error_log_path: str = Field(_FIELD_DEFAULTS["error_log_path"], env="REX_ERROR_LOG_PATH")
        transcripts_dir: str = Field(_FIELD_DEFAULTS["transcripts_dir"], env="REX_TRANSCRIPTS_DIR")
        search_providers: str = Field(_FIELD_DEFAULTS["search_providers"], env="REX_SEARCH_PROVIDERS")
        speak_language: str = Field(_FIELD_DEFAULTS["speak_language"], env="REX_SPEAK_LANGUAGE")
        input_device: Optional[int] = Field(default=None, env="REX_INPUT_DEVICE")
        output_device: Optional[int] = Field(default=None, env="REX_OUTPUT_DEVICE")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

else:

    @dataclasses.dataclass
    class Settings:
        whisper_model: str = _FIELD_DEFAULTS["whisper_model"]
        whisper_device: str = _FIELD_DEFAULTS["whisper_device"]
        temperature: float = _FIELD_DEFAULTS["temperature"]
        user_id: str = _FIELD_DEFAULTS["user_id"]
        llm_model: str = _FIELD_DEFAULTS["llm_model"]
        llm_backend: str = _FIELD_DEFAULTS["llm_backend"]
        max_memory_items: int = _FIELD_DEFAULTS["max_memory_items"]
        wakeword_keyword: str = _FIELD_DEFAULTS["wakeword_keyword"]
        wakeword_threshold: float = _FIELD_DEFAULTS["wakeword_threshold"]
        sample_rate: int = _FIELD_DEFAULTS["sample_rate"]
        detection_frame_seconds: float = _FIELD_DEFAULTS["detection_frame_seconds"]
        capture_seconds: float = _FIELD_DEFAULTS["capture_seconds"]
        wakeword_poll_interval: float = _FIELD_DEFAULTS["wakeword_poll_interval"]
        log_path: str = _FIELD_DEFAULTS["log_path"]
        error_log_path: str = _FIELD_DEFAULTS["error_log_path"]
        transcripts_dir: str = _FIELD_DEFAULTS["transcripts_dir"]
        search_providers: str = _FIELD_DEFAULTS["search_providers"]
        speak_language: str = _FIELD_DEFAULTS["speak_language"]
        input_device: Optional[int] = _FIELD_DEFAULTS["input_device"]
        output_device: Optional[int] = _FIELD_DEFAULTS["output_device"]

        def dict(self) -> Dict[str, Any]:  # pragma: no cover - simple accessor
            return dataclasses.asdict(self)


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    load_dotenv()
    if _HAS_PYDANTIC:
        return Settings()  # type: ignore[call-arg]

    values: Dict[str, Any] = {}
    for field, aliases in _ENV_ALIASES.items():
        for env_var in aliases:
            raw = os.getenv(env_var)
            if raw is None or raw == "":
                continue
            try:
                values[field] = _cast_value(field, raw)
            except (TypeError, ValueError):
                logger.warning("Invalid value for %s: %s", env_var, raw)
                continue
            else:
                break
    return Settings(**values)


settings = _load_settings()


def reload_settings() -> Settings:
    """Force settings to be reloaded from the environment."""
    _load_settings.cache_clear()
    new_settings = _load_settings()
    globals()["settings"] = new_settings
    return new_settings


def load_config(*, env_path: Optional[Path] = None, reload: bool = False) -> Settings:
    """Load configuration (backward compatibility wrapper)."""
    if reload:
        return reload_settings()
    return settings


def show_config() -> None:
    """Print the current configuration values for debugging."""
    for key, value in sorted(settings.dict().items()):
        print(f"{key}: {value}")


def validate_config(config: Settings) -> None:
    """Validate configuration (calls __post_init__ internally)."""
    config.__post_init__()


def _env_path() -> Path:
    return Path.cwd() / ".env"


def update_env_value(key: str, value: str) -> None:
    """Persist a configuration value into the ``.env`` file."""
    env_path = _env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(env_path), key, value)
    reload_settings()


def _format_settings() -> Dict[str, Any]:
    current = settings.dict()
    current_sorted: Dict[str, Any] = dict(sorted(current.items()))
    return current_sorted


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Rex configuration values.")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Persist a key/value pair to .env")
    parser.add_argument("--get", metavar="KEY", help="Print a single configuration value")
    parser.add_argument("--show", action="store_true", help="Print all resolved configuration values")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.set:
        key, value = args.set
        update_env_value(key, value)
        print(f"Updated {key} -> {value}")
        return 0

    if args.get:
        key = args.get.lower()
        value = settings.dict().get(key)
        if value is None:
            print("<unset>")
        else:
            print(value)
        return 0

    if args.show:
        for key, value in _format_settings().items():
            print(f"{key}: {value}")
        return 0

    parser.print_help()
    return 0


def cli(argv: Iterable[str] | None = None) -> int:
    """Public CLI entrypoint wrapper."""
    return _cli(argv)


# Backward compatibility aliases
AppConfig = Settings
ENV_PATH = Path.cwd() / ".env"
ENV_MAPPING = _ENV_ALIASES


__all__ = [
    "Settings",
    "AppConfig",
    "cli",
    "settings",
    "load_config",
    "reload_settings",
    "validate_config",
    "update_env_value",
    "show_config",
    "REQUIRED_ENV_KEYS",
]


if __name__ == "__main__":
    raise SystemExit(_cli())