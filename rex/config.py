"""Central configuration loader and CLI utilities for the Rex assistant."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv, set_key
except ImportError:
    def load_dotenv(*args, **kwargs): return False
    def set_key(env_path: str, key: str, value: str):
        path = Path(env_path)
        lines = [line for line in path.read_text().splitlines() if not line.startswith(f"{key}=")] if path.exists() else []
        lines.append(f"{key}={value}")
        path.write_text("\n".join(lines) + "\n")
        return key, value, True

from rex.assistant_errors import ConfigurationError
from rex.logging_utils import get_logger, set_global_level

LOGGER = get_logger(__name__)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
REQUIRED_ENV_KEYS = {"REX_WAKEWORD"}

@dataclass
class AppConfig:
    wakeword: str = "rex"
    wakeword_threshold: float = 0.5
    wakeword_window: float = 1.0
    wakeword_poll_interval: float = 0.01
    command_duration: float = 5.0

    sample_rate: int = 16000
    detection_frame_seconds: float = 1.0
    capture_seconds: float = 5.0

    whisper_model: str = "base"
    whisper_device: str = "cpu"
    llm_provider: str = "transformers"
    llm_model: str = "sshleifer/tiny-gpt2"
    llm_max_tokens: int = 120
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k: int = 50
    llm_seed: int = 42

    speak_api_key: Optional[str] = None
    rate_limit: str = "30/minute"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

    memory_max_turns: int = 50
    transcripts_enabled: bool = True
    transcripts_dir: Path = Path("transcripts")
    default_user: Optional[str] = None
    wake_sound_path: Optional[str] = None

    audio_input_device: Optional[int] = None
    audio_output_device: Optional[int] = None

    debug_logging: bool = False
    conversation_export: bool = True

    brave_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None

    ollama_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_use_cloud: bool = False

    search_providers: str = "serpapi,brave,duckduckgo,google"
    speak_language: str = "en"

    # Aliases
    llm_backend: Optional[str] = None
    temperature: Optional[float] = None
    max_memory_items: Optional[int] = None
    user_id: str = "default"
    wakeword_keyword: Optional[str] = None

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["transcripts_dir"] = str(self.transcripts_dir)
        return raw

    def __post_init__(self) -> None:
        model_path = Path(self.llm_model)
        if model_path.is_absolute() or ".." in model_path.parts:
            raise ValueError("llm_model must not contain path traversal components.")
        if self.llm_backend is None:
            self.llm_backend = self.llm_provider
        if self.temperature is None:
            self.temperature = self.llm_temperature
        if self.max_memory_items is None:
            self.max_memory_items = self.memory_max_turns
        if self.wakeword_keyword is None:
            self.wakeword_keyword = self.wakeword

_cached_config: Optional[AppConfig] = None

ENV_MAPPING: Dict[str, str] = {
    "wakeword": "REX_WAKEWORD",
    "wakeword_threshold": "REX_WAKEWORD_THRESHOLD",
    "wakeword_window": "REX_WAKEWORD_WINDOW",
    "wakeword_poll_interval": "REX_WAKEWORD_POLL_INTERVAL",
    "command_duration": "REX_COMMAND_DURATION",
    "sample_rate": "REX_SAMPLE_RATE",
    "detection_frame_seconds": "REX_DETECTION_FRAME_SECONDS",
    "capture_seconds": "REX_CAPTURE_SECONDS",
    "whisper_model": "REX_WHISPER_MODEL",
    "whisper_device": "REX_WHISPER_DEVICE",
    "speak_language": "REX_SPEAK_LANGUAGE",
    "llm_provider": "REX_LLM_PROVIDER",
    "llm_model": "REX_LLM_MODEL",
    "llm_max_tokens": "REX_LLM_MAX_TOKENS",
    "llm_temperature": "REX_LLM_TEMPERATURE",
    "llm_top_p": "REX_LLM_TOP_P",
    "llm_top_k": "REX_LLM_TOP_K",
    "llm_seed": "REX_LLM_SEED",
    "speak_api_key": "REX_SPEAK_API_KEY",
    "rate_limit": "REX_RATE_LIMIT",
    "allowed_origins": "REX_ALLOWED_ORIGINS",
    "memory_max_turns": "REX_MEMORY_MAX_TURNS",
    "transcripts_enabled": "REX_TRANSCRIPTS_ENABLED",
    "transcripts_dir": "REX_TRANSCRIPTS_DIR",
    "default_user": "REX_ACTIVE_USER",
    "wake_sound_path": "REX_WAKE_SOUND",
    "audio_input_device": "REX_INPUT_DEVICE",
    "audio_output_device": "REX_OUTPUT_DEVICE",
    "debug_logging": "REX_DEBUG_LOGGING",
    "conversation_export": "REX_CONVERSATION_EXPORT",
    "brave_api_key": "BRAVE_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model": "OPENAI_MODEL",
    "openai_base_url": "OPENAI_BASE_URL",
    "ollama_api_key": "OLLAMA_API_KEY",
    "ollama_base_url": "OLLAMA_HOST",
    "ollama_use_cloud": "OLLAMA_USE_CLOUD",
    "user_id": "REX_USER_ID",
    "search_providers": "REX_SEARCH_PROVIDERS",
}

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value}")

def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer: {value}") from exc

def _first_env_value(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return None

def _env_key_for(field_name: str) -> Optional[str]:
    if field_name in ENV_MAPPING:
        return ENV_MAPPING[field_name]
    elif field_name.startswith("REX_") or field_name in os.environ:
        return field_name
    return None

def _write_env(key: str, value: str, *, env_path: Path) -> None:
    set_key(str(env_path), key, value)

def load_config(*, env_path: Optional[Path] = None, reload: bool = False) -> AppConfig:
    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    load_dotenv(env_path or ENV_PATH, override=False)

    def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    allowed_origins = [
        origin.strip().rstrip("/")
        for origin in (getenv("REX_ALLOWED_ORIGINS", "*").split(","))
        if origin.strip()
    ] or ["*"]

    config = AppConfig(
        wakeword=getenv("REX_WAKEWORD", "rex"),
        wakeword_threshold=float(getenv("REX_WAKEWORD_THRESHOLD", "0.5")),
        wakeword_window=float(getenv("REX_WAKEWORD_WINDOW", "1.0")),
        wakeword_poll_interval=float(getenv("REX_WAKEWORD_POLL_INTERVAL", "0.01")),
        command_duration=float(getenv("REX_COMMAND_DURATION", "5.0")),
        sample_rate=int(getenv("REX_SAMPLE_RATE", "16000")),
        detection_frame_seconds=float(getenv("REX_DETECTION_FRAME_SECONDS", "1.0")),
        capture_seconds=float(getenv("REX_CAPTURE_SECONDS", "5.0")),
        whisper_model=getenv("REX_WHISPER_MODEL", "base"),
        whisper_device=getenv("REX_WHISPER_DEVICE", "cpu"),
        speak_language=getenv("REX_SPEAK_LANGUAGE", "en"),
        llm_provider=getenv("REX_LLM_PROVIDER", "transformers"),
        llm_model=getenv("REX_LLM_MODEL", "sshleifer/tiny-gpt2"),
        llm_max_tokens=int(getenv("REX_LLM_MAX_TOKENS", "120")),
        llm_temperature=float(getenv("REX_LLM_TEMPERATURE", "0.7")),
        llm_top_p=float(getenv("REX_LLM_TOP_P", "0.9")),
        llm_top_k=int(getenv("REX_LLM_TOP_K", "50")),
        llm_seed=int(getenv("REX_LLM_SEED", "42")),
        speak_api_key=getenv("REX_SPEAK_API_KEY"),
        rate_limit=getenv("REX_RATE_LIMIT", "30/minute"),
        allowed_origins=allowed_origins,
        memory_max_turns=int(getenv("REX_MEMORY_MAX_TURNS", "50")),
        transcripts_enabled=_parse_bool(getenv("REX_TRANSCRIPTS_ENABLED"), default=True),
        transcripts_dir=Path(getenv("REX_TRANSCRIPTS_DIR", "transcripts")),
        default_user=getenv("REX_ACTIVE_USER"),
        wake_sound_path=getenv("REX_WAKE_SOUND"),
        audio_input_device=_parse_optional_int(_first_env_value("REX_INPUT_DEVICE", "REX_AUDIO_INPUT_DEVICE")),
        audio_output_device=_parse_optional_int(_first_env_value("REX_OUTPUT_DEVICE", "REX_AUDIO_OUTPUT_DEVICE")),
        debug_logging=_parse_bool(getenv("REX_DEBUG_LOGGING")),
        conversation_export=_parse_bool(getenv("REX_CONVERSATION_EXPORT"), default=True),
        brave_api_key=getenv("BRAVE_API_KEY"),
        openai_api_key=getenv("OPENAI_API_KEY"),
        openai_model=getenv("OPENAI_MODEL"),
        openai_base_url=getenv("OPENAI_BASE_URL"),
        ollama_api_key=getenv("OLLAMA_API_KEY"),
        ollama_base_url=getenv("OLLAMA_HOST") or getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_use_cloud=_parse_bool(getenv("OLLAMA_USE_CLOUD")),
        user_id=getenv("REX_USER_ID", "default"),
        search_providers=getenv("REX_SEARCH_PROVIDERS", "serpapi,brave,duckduckgo,google"),
    )

    validate_config(config)
    _cached_config = config

    if config.debug_logging:
        set_global_level(10)

    return config

def validate_config(config: AppConfig) -> None:
    if not (0 < config.wakeword_threshold <= 1):
        raise ConfigurationError("wakeword_threshold must be between 0 and 1.")
    if config.command_duration <= 0:
        raise ConfigurationError("command_duration must be positive.")
    if config.wakeword_window <= 0:
        raise ConfigurationError("wakeword_window must be positive.")
    if config.llm_max_tokens <= 0:
        raise ConfigurationError("llm_max_tokens must be positive.")
    if not (0 <= config.llm_temperature <= 5.0):
        raise ConfigurationError("llm_temperature must be between 0 and 5.")
    if config.memory_max_turns <= 0:
        raise ConfigurationError("memory_max_turns must be positive.")

def reload_settings(*, env_path: Optional[Path] = None) -> AppConfig:
    return load_config(env_path=env_path, reload=True)

def show_config(config: Optional[AppConfig] = None) -> None:
    cfg = config or load_config()
    for key, value in cfg.to_dict().items():
        LOGGER.info("%s = %s", key, value)

def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Configure Rex Assistant")
    parser.add_argument("--show", action="store_true", help="Print current configuration")
    parser.add_argument("--set", nargs="*", help="Set config values (key=value)")
    parser.add_argument("--reload", action="store_true", help="Reload config after changes")
    args = parser.parse_args(argv)

    ENV_PATH.touch(exist_ok=True)

    if args.set:
        for pair in args.set:
            if "=" not in pair:
                raise ConfigurationError(f"Invalid --set argument (use key=value): {pair}")
            key, value = map(str.strip, pair.split("=", 1))
            env_key = _env_key_for(key)
            if not env_key:
                raise ConfigurationError(f"Unknown config key: {key}")
            _write_env(env_key, value, env_path=ENV_PATH)
            LOGGER.info("Updated %s → %s", env_key, value)

    if args.reload or args.set or args.show:
        load_config(env_path=ENV_PATH, reload=True)

    if args.show or not args.set:
        show_config(_cached_config)

    return 0

settings = load_config()
Settings = AppConfig

if __name__ == "__main__":
    try:
        raise SystemExit(cli())
    except ConfigurationError as exc:
        LOGGER.error("Config error: %s", exc)
        raise SystemExit(1) from exc


