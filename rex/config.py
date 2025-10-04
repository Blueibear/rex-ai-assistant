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

# Optional dotenv support
try:
    from dotenv import load_dotenv, set_key
except ImportError:
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

# Optional pydantic support
try:
    from pydantic import BaseSettings, Field
    _HAS_PYDANTIC = True
except ImportError:
    BaseSettings = object  # type: ignore
    Field = None  # type: ignore
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
        return int(raw) if raw.lower() not in {"", "none"} else None
    return raw


if _HAS_PYDANTIC:
    class Settings(BaseSettings):
        whisper_model: str = Field("medium", env=["WHISPER_MODEL", "REX_WHISPER_MODEL"])
        whisper_device: str = Field("cuda", env=["WHISPER_DEVICE", "REX_WHISPER_DEVICE"])
        temperature: float = Field(0.8, env="REX_LLM_TEMPERATURE")
        user_id: str = Field("default", env="REX_ACTIVE_USER")
        llm_model: str = Field("distilgpt2", env="REX_LLM_MODEL")
        llm_backend: str = Field("transformers", env="REX_LLM_BACKEND")
        max_memory_items: int = Field(50, env="REX_MEMORY_MAX_ITEMS")
        wakeword_keyword: str = Field("hey_jarvis", env="REX_WAKEWORD_KEYWORD")
        wakeword_threshold: float = Field(0.5, env="REX_WAKEWORD_THRESHOLD")
        sample_rate: int = Field(16000, env="REX_SAMPLE_RATE")
        detection_frame_seconds: float = Field(0.5, env="REX_DETECTION_FRAME_SECONDS")
        capture_seconds: float = Field(5.0, env="REX_CAPTURE_SECONDS")
        wakeword_poll_interval: float = Field(0.05, env="REX_WAKEWORD_POLL_INTERVAL")
        log_path: str = Field("logs/rex.log", env="REX_LOG_PATH")
        error_log_path: str = Field("logs/error.log", env="REX_ERROR_LOG_PATH")
        transcripts_dir: str = Field("transcripts", env="REX_TRANSCRIPTS_DIR")
        search_providers: str = Field("serpapi,brave,duckduckgo", env="REX_SEARCH_PROVIDERS")
        speak_language: str = Field("en", env="REX_SPEAK_LANGUAGE")
        input_device: Optional[int] = Field(default=None, env="REX_INPUT_DEVICE")
        output_device: Optional[int] = Field(default=None, env="REX_OUTPUT_DEVICE")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

else:
    @dataclasses.dataclass
    class Settings:
        whisper_model: str = "medium"
        whisper_device: str = "cuda"
        temperature: float = 0.8
        user_id: str = "default"
        llm_model: str = "distilgpt2"
        llm_backend: str = "transformers"
        max_memory_items: int = 50
        wakeword_keyword: str = "hey_jarvis"
        wakeword_threshold: float = 0.5
        sample_rate: int = 16000
        detection_frame_seconds: float = 0.5
        capture_seconds: float = 5.0
        wakeword_poll_interval: float = 0.05
        log_path: str = "logs/rex.log"
        error_log_path: str = "logs/error.log"
        transcripts_dir: str = "transcripts"
        search_providers: str = "serpapi,brave,duckduckgo"
        speak_language: str = "en"
        input_device: Optional[int] = None
        output_device: Optional[int] = None

        def dict(self) -> Dict[str, Any]:
            return dataclasses.asdict(self)


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    load_dotenv()
    if _HAS_PYDANTIC:
        return Settings()
    values: Dict[str, Any] = {}
    for key, aliases in _ENV_ALIASES.items():
        for var in aliases:
            raw = os.getenv(var)
            if raw:
                try:
                    values[key] = _cast_value(key, raw)
                except Exception:
                    logger.warning("Invalid value for %s = %s", var, raw)
                break
    return Settings(**values)


settings = _load_settings()


def reload_settings() -> Settings:
    _load_settings.cache_clear()
    new_settings = _load_settings()
    globals()["settings"] = new_settings
    return new_settings


def update_env_value(key: str, value: str) -> None:
    path = Path.cwd() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(path), key, value)
    reload_settings()


def _format_settings() -> Dict[str, Any]:
    return dict(sorted(settings.dict().items()))


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
        print(value if value is not None else "<unset>")
        return 0

    if args.show:
        for key, value in _format_settings().items():
            print(f"{key}: {value}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

