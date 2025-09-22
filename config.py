"""Central configuration loader and CLI utilities for the Rex assistant."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv, set_key
except ImportError:  # pragma: no cover - fallback for minimal environments
    def load_dotenv(*args, **kwargs):
        return False

    def set_key(env_path: str, key: str, value: str):
        path = Path(env_path)
        if path.is_file():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(f"{key}=")]
        else:
            lines = []
        lines.append(f"{key}={value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return key, value, True

from assistant_errors import ConfigurationError
from logging_utils import get_logger, set_global_level

LOGGER = get_logger(__name__)

ENV_PATH = Path(__file__).resolve().parent / ".env"
REQUIRED_ENV_KEYS = {"REX_WAKEWORD"}


@dataclass
class AppConfig:
    """Dataclass capturing all runtime configuration for the assistant."""

    wakeword: str = "rex"
    wakeword_threshold: float = 0.5
    wakeword_window: float = 1.0
    command_duration: float = 5.0
    whisper_model: str = "base"
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

    def to_dict(self) -> Dict[str, object]:
        raw = asdict(self)
        raw["transcripts_dir"] = str(self.transcripts_dir)
        return raw


_cached_config: Optional[AppConfig] = None

ENV_MAPPING: Dict[str, str] = {
    "wakeword": "REX_WAKEWORD",
    "wakeword_threshold": "REX_WAKEWORD_THRESHOLD",
    "wakeword_window": "REX_WAKEWORD_WINDOW",
    "command_duration": "REX_COMMAND_DURATION",
    "whisper_model": "REX_WHISPER_MODEL",
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
    "audio_input_device": "REX_AUDIO_INPUT_DEVICE",
    "audio_output_device": "REX_AUDIO_OUTPUT_DEVICE",
    "debug_logging": "REX_DEBUG_LOGGING",
    "conversation_export": "REX_CONVERSATION_EXPORT",
    "brave_api_key": "BRAVE_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
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
        raise ConfigurationError(f"Invalid integer value: {value}") from exc


def load_config(*, env_path: Optional[Path] = None, reload: bool = False) -> AppConfig:
    """Load configuration values from ``.env`` and process environment variables."""

    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    env_file = env_path or ENV_PATH
    load_dotenv(env_file, override=False)

    for key in REQUIRED_ENV_KEYS:
        if not os.getenv(key):
            LOGGER.warning("Environment variable %s missing; using defaults", key)

    wakeword = os.getenv("REX_WAKEWORD", "rex")
    threshold = float(os.getenv("REX_WAKEWORD_THRESHOLD", "0.5"))
    window = float(os.getenv("REX_WAKEWORD_WINDOW", "1.0"))
    command_duration = float(os.getenv("REX_COMMAND_DURATION", "5.0"))
    whisper_model = os.getenv("REX_WHISPER_MODEL", "base")
    llm_provider = os.getenv("REX_LLM_PROVIDER", "transformers")
    llm_model = os.getenv("REX_LLM_MODEL", "sshleifer/tiny-gpt2")
    llm_max_tokens = int(os.getenv("REX_LLM_MAX_TOKENS", "120"))
    llm_temperature = float(os.getenv("REX_LLM_TEMPERATURE", "0.7"))
    llm_top_p = float(os.getenv("REX_LLM_TOP_P", "0.9"))
    llm_top_k = int(os.getenv("REX_LLM_TOP_K", "50"))
    llm_seed = int(os.getenv("REX_LLM_SEED", "42"))
    speak_api_key = os.getenv("REX_SPEAK_API_KEY")
    rate_limit = os.getenv("REX_RATE_LIMIT", "30/minute")
    allowed_origins_raw = os.getenv("REX_ALLOWED_ORIGINS", "*")
    allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]
    if not allowed_origins:
        allowed_origins = ["*"]
    memory_max_turns = int(os.getenv("REX_MEMORY_MAX_TURNS", "50"))
    transcripts_enabled = _parse_bool(os.getenv("REX_TRANSCRIPTS_ENABLED"), default=True)
    transcripts_dir = Path(os.getenv("REX_TRANSCRIPTS_DIR", "transcripts"))
    default_user = os.getenv("REX_ACTIVE_USER")
    wake_sound_path = os.getenv("REX_WAKE_SOUND")
    audio_input_device = _parse_optional_int(os.getenv("REX_AUDIO_INPUT_DEVICE"))
    audio_output_device = _parse_optional_int(os.getenv("REX_AUDIO_OUTPUT_DEVICE"))
    debug_logging = _parse_bool(os.getenv("REX_DEBUG_LOGGING"), default=False)
    conversation_export = _parse_bool(os.getenv("REX_CONVERSATION_EXPORT"), default=True)
    brave_api_key = os.getenv("BRAVE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    config = AppConfig(
        wakeword=wakeword,
        wakeword_threshold=threshold,
        wakeword_window=window,
        command_duration=command_duration,
        whisper_model=whisper_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_top_k=llm_top_k,
        llm_seed=llm_seed,
        speak_api_key=speak_api_key,
        rate_limit=rate_limit,
        allowed_origins=allowed_origins,
        memory_max_turns=memory_max_turns,
        transcripts_enabled=transcripts_enabled,
        transcripts_dir=transcripts_dir,
        default_user=default_user,
        wake_sound_path=wake_sound_path,
        audio_input_device=audio_input_device,
        audio_output_device=audio_output_device,
        debug_logging=debug_logging,
        conversation_export=conversation_export,
        brave_api_key=brave_api_key,
        openai_api_key=openai_api_key,
    )

    validate_config(config)
    _cached_config = config
    if config.debug_logging:
        set_global_level(10)
    return config


def validate_config(config: AppConfig) -> None:
    if not 0 < config.wakeword_threshold <= 1:
        raise ConfigurationError("Wakeword threshold must be in the range (0, 1].")
    if config.command_duration <= 0:
        raise ConfigurationError("Command duration must be positive.")
    if config.wakeword_window <= 0:
        raise ConfigurationError("Wakeword window must be positive.")
    if config.llm_max_tokens <= 0:
        raise ConfigurationError("LLM max tokens must be positive.")
    if config.llm_temperature < 0:
        raise ConfigurationError("LLM temperature must be non-negative.")
    if config.memory_max_turns <= 0:
        raise ConfigurationError("memory_max_turns must be positive.")


def _env_key_for(field_name: str) -> Optional[str]:
    return ENV_MAPPING.get(field_name)


def _write_env(key: str, value: str, *, env_path: Path) -> None:
    set_key(str(env_path), key, value)


def _iter_config_fields() -> Iterable[str]:
    return ENV_MAPPING.keys()


def show_config(config: Optional[AppConfig] = None) -> None:
    cfg = config or load_config()
    for key, value in cfg.to_dict().items():
        LOGGER.info("%s = %s", key, value)


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Configure the Rex assistant")
    parser.add_argument("--show", action="store_true", help="Display the current configuration")
    parser.add_argument("--set", nargs="*", default=[], help="Set configuration values (key=value)")
    parser.add_argument("--reload", action="store_true", help="Reload configuration after applying changes")
    args = parser.parse_args(argv)

    env_path = ENV_PATH
    env_path.touch(exist_ok=True)

    if args.set:
        for assignment in args.set:
            if "=" not in assignment:
                raise ConfigurationError(f"Invalid --set argument: {assignment}")
            field_name, value = assignment.split("=", 1)
            field_name = field_name.strip()
            value = value.strip()
            env_key = _env_key_for(field_name)
            if not env_key:
                raise ConfigurationError(f"Unknown configuration key: {field_name}")
            _write_env(env_key, value, env_path=env_path)
            LOGGER.info("Set %s (%s) to %s", field_name, env_key, value)

    if args.reload or args.set or args.show:
        load_config(env_path=env_path, reload=True)

    if args.show or not args.set:
        show_config(_cached_config)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        raise SystemExit(cli())
    except ConfigurationError as exc:
        LOGGER.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc
