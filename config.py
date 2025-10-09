"""Central configuration management with YAML plus environment overrides."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # Optional YAML support
    import yaml
except ImportError:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None  # type: ignore[assignment]

try:  # Optional helper for persisting .env overrides
    from dotenv import load_dotenv, set_key
except ImportError:  # pragma: no cover - provide minimal fallbacks

    def load_dotenv(path: str | Path, override: bool = False) -> bool:  # type: ignore[override]
        env_path = Path(path)
        if not env_path.exists():
            return False
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if override or key not in os.environ:
                os.environ[key.strip()] = value.strip()
        return True

    def set_key(env_path: str, key: str, value: str) -> tuple[str, str, bool]:  # type: ignore[override]
        path = Path(env_path)
        lines = []
        if path.exists():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        prefix = f"{key}="
        replaced = False
        for idx, line in enumerate(lines):
            if line.startswith(prefix):
                lines[idx] = f"{prefix}{value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{prefix}{value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return key, value, True

from assistant_errors import ConfigurationError
from logging_utils import get_logger, set_global_level

LOGGER = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass
class AppConfig:
    """Application-wide configuration with runtime helpers."""

    # Runtime & hardware
    gpu: bool = False
    cuda_device: int = 0

    # Wake word detection
    wakeword_backend: str = "picovoice"
    wakeword: str = "rex"
    wakeword_threshold: float = 0.5
    wakeword_sensitivity: float = 0.6
    wakeword_window: float = 1.0
    wakeword_poll_interval: float = 0.05

    # Speech-to-text
    whisper_backend: str = "faster-whisper"
    whisper_model: str = "medium"
    whisper_device: str | None = "auto"
    whisper_compute_type: str = "int8"
    sample_rate: int = 16000
    detection_frame_seconds: float = 0.5
    capture_seconds: float = 5.0

    # LLM
    llm_backend: str = "ollama"
    llm_model: str = "llama3"
    llm_url: str = "http://localhost:11434"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k: int = 50
    llm_seed: int = 42
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    user_id: str = "default"

    # Text-to-speech
    tts_backend: str = "xtts"
    speak_language: str = "en"
    voice_profile: str = "default"

    # Web & API
    rate_limit: str = "30/minute"
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    brave_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None

    # Memory & transcripts
    transcripts_enabled: bool = True
    transcripts_dir: Path = Path("transcripts")
    memory_backend: str = "tinydb"
    memory_path: Path = Path("Memory/memory.json")
    user_profiles_dir: Path = Path("Memory")
    max_memory_items: int = 50
    memory_max_turns: int = 50
    default_user: Optional[str] = None

    # Audio devices
    audio_input_device: str | int | None = "default"
    audio_output_device: str | int | None = "default"
    command_duration: float = 5.0

    # Logging & debugging
    debug_logging: bool = False
    conversation_export: bool = True

    # Search
    search_providers: str = "serpapi,duckduckgo"

    # Legacy compatibility hooks
    wake_sound_path: Optional[str] = None
    speak_api_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transcripts_dir"] = str(self.transcripts_dir)
        data["memory_path"] = str(self.memory_path)
        data["user_profiles_dir"] = str(self.user_profiles_dir)
        return data

    # Compatibility aliases used throughout the codebase
    @property
    def llm_provider(self) -> str:
        return self.llm_backend

    @property
    def temperature(self) -> float:
        return self.llm_temperature

    @property
    def wakeword_keyword(self) -> str:
        return self.wakeword


ENV_MAPPING: Dict[str, str] = {
    "gpu": "REX_GPU",
    "cuda_device": "REX_CUDA_DEVICE",
    "wakeword_backend": "REX_WAKEWORD_BACKEND",
    "wakeword": "REX_WAKEWORD",
    "wakeword_threshold": "REX_WAKEWORD_THRESHOLD",
    "wakeword_sensitivity": "REX_WAKEWORD_SENSITIVITY",
    "wakeword_window": "REX_WAKEWORD_WINDOW",
    "wakeword_poll_interval": "REX_WAKEWORD_POLL_INTERVAL",
    "command_duration": "REX_COMMAND_DURATION",
    "whisper_backend": "REX_WHISPER_BACKEND",
    "whisper_model": "REX_WHISPER_MODEL",
    "whisper_device": "REX_WHISPER_DEVICE",
    "whisper_compute_type": "REX_WHISPER_COMPUTE_TYPE",
    "sample_rate": "REX_SAMPLE_RATE",
    "detection_frame_seconds": "REX_DETECTION_FRAME_SECONDS",
    "capture_seconds": "REX_CAPTURE_SECONDS",
    "llm_backend": "REX_LLM_BACKEND",
    "llm_model": "REX_LLM_MODEL",
    "llm_url": "REX_LLM_URL",
    "llm_max_tokens": "REX_LLM_MAX_TOKENS",
    "llm_temperature": "REX_LLM_TEMPERATURE",
    "llm_top_p": "REX_LLM_TOP_P",
    "llm_top_k": "REX_LLM_TOP_K",
    "llm_seed": "REX_LLM_SEED",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model": "OPENAI_MODEL",
    "user_id": "REX_ACTIVE_USER",
    "tts_backend": "REX_TTS_BACKEND",
    "speak_language": "REX_SPEAK_LANGUAGE",
    "voice_profile": "REX_VOICE_PROFILE",
    "rate_limit": "REX_RATE_LIMIT",
    "allowed_origins": "REX_ALLOWED_ORIGINS",
    "brave_api_key": "BRAVE_API_KEY",
    "serpapi_key": "SERPAPI_KEY",
    "transcripts_enabled": "REX_TRANSCRIPTS_ENABLED",
    "transcripts_dir": "REX_TRANSCRIPTS_DIR",
    "memory_backend": "REX_MEMORY_BACKEND",
    "memory_path": "REX_MEMORY_PATH",
    "user_profiles_dir": "REX_USER_PROFILES_DIR",
    "max_memory_items": "REX_MEMORY_MAX_ITEMS",
    "memory_max_turns": "REX_MEMORY_MAX_TURNS",
    "default_user": "REX_DEFAULT_USER",
    "audio_input_device": "REX_INPUT_DEVICE",
    "audio_output_device": "REX_OUTPUT_DEVICE",
    "debug_logging": "REX_DEBUG_LOGGING",
    "conversation_export": "REX_CONVERSATION_EXPORT",
    "search_providers": "REX_SEARCH_PROVIDERS",
    "wake_sound_path": "REX_WAKE_SOUND",
    "speak_api_key": "REX_SPEAK_API_KEY",
}

_cached_config: Optional[AppConfig] = None


def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value}")


def _parse_int(value: Optional[str], *, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Invalid integer value: {value}") from exc


def _parse_float(value: Optional[str], *, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Invalid float value: {value}") from exc


def _normalize_device(value: Any) -> str | int | None:
    if value in {None, "", "none", "None"}:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() == "default":
            return "default"
        if cleaned.isdigit():
            return int(cleaned)
        return cleaned
    return value


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is None:
        LOGGER.warning("PyYAML not installed; skipping %s parsing and using defaults.", path)
        return {}
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ConfigurationError("config.yaml must contain a mapping of keys to values")
    return content


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return _parse_bool(value, default=default)
    return bool(value)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    data = config.to_dict()
    for attr, env_key in ENV_MAPPING.items():
        raw = os.getenv(env_key)
        if raw in (None, ""):
            continue
        if attr in {"gpu", "transcripts_enabled", "debug_logging", "conversation_export"}:
            data[attr] = _parse_bool(raw, default=getattr(config, attr))
        elif attr in {"wakeword_threshold", "llm_temperature", "llm_top_p", "wakeword_poll_interval",
                      "wakeword_window", "command_duration", "detection_frame_seconds", "capture_seconds"}:
            data[attr] = _parse_float(raw, default=float(getattr(config, attr)))
        elif attr in {"llm_top_k", "llm_max_tokens", "llm_seed", "memory_max_turns", "max_memory_items",
                      "cuda_device", "sample_rate"}:
            data[attr] = _parse_int(raw, default=int(getattr(config, attr)))
        elif attr in {"audio_input_device", "audio_output_device"}:
            data[attr] = raw if raw.lower() == "default" else _parse_int(raw, default=None)
        elif attr in {"transcripts_dir", "memory_path", "user_profiles_dir"}:
            data[attr] = str(Path(raw))
        elif attr == "allowed_origins":
            data[attr] = [segment.strip().rstrip("/") for segment in raw.split(",") if segment.strip()]
        else:
            data[attr] = raw
    return AppConfig(
        **{
            **data,
            "transcripts_dir": Path(data["transcripts_dir"]),
            "memory_path": Path(data["memory_path"]),
            "user_profiles_dir": Path(data["user_profiles_dir"]),
        }
    )


def _finalise_config(config: AppConfig) -> AppConfig:
    preferred_device = (config.whisper_device or "auto").strip().lower()
    if preferred_device == "auto":
        config.whisper_device = "cuda" if config.gpu else "cpu"
    elif preferred_device == "cuda" and not config.gpu:
        LOGGER.warning("whisper_device set to CUDA but GPU acceleration is disabled; falling back to CPU.")
        config.whisper_device = "cpu"
    elif preferred_device not in {"cuda", "cpu"}:
        config.whisper_device = preferred_device

    if str(config.whisper_device).lower() != "cuda":
        config.whisper_compute_type = "int8"
    elif config.whisper_compute_type.lower() == "int8":
        config.whisper_compute_type = "float16"
    return config


def load_config(*, config_path: Path = CONFIG_FILE, env_path: Path = ENV_PATH, reload: bool = False) -> AppConfig:
    """Load configuration from YAML, apply environment overrides, and cache the result."""

    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    load_dotenv(env_path, override=False)

    yaml_data = _load_yaml_config(config_path)
    defaults = AppConfig()
    merged: dict[str, Any] = defaults.to_dict()
    for key, value in yaml_data.items():
        if key in {"transcripts_dir", "memory_path", "user_profiles_dir"}:
            merged[key] = str(Path(value))
        else:
            merged[key] = value

    allowed_origins_value = merged.get("allowed_origins", defaults.allowed_origins)
    if isinstance(allowed_origins_value, str):
        allowed_origins = [origin.strip().rstrip("/") for origin in allowed_origins_value.split(",") if origin.strip()]
    elif isinstance(allowed_origins_value, Iterable):
        allowed_origins = [str(origin).strip().rstrip("/") for origin in allowed_origins_value if str(origin).strip()]
    else:
        allowed_origins = list(defaults.allowed_origins)
    if not allowed_origins:
        allowed_origins = ["*"]

    audio_input = _normalize_device(merged.get("audio_input_device", defaults.audio_input_device))
    audio_output = _normalize_device(merged.get("audio_output_device", defaults.audio_output_device))

    config = AppConfig(
        gpu=_coerce_bool(merged.get("gpu", defaults.gpu), defaults.gpu),
        cuda_device=int(merged.get("cuda_device", defaults.cuda_device)),
        wakeword_backend=str(merged.get("wakeword_backend", defaults.wakeword_backend)),
        wakeword=str(merged.get("wakeword", defaults.wakeword)),
        wakeword_threshold=float(merged.get("wakeword_threshold", defaults.wakeword_threshold)),
        wakeword_sensitivity=float(merged.get("wakeword_sensitivity", defaults.wakeword_sensitivity)),
        wakeword_window=float(merged.get("wakeword_window", defaults.wakeword_window)),
        wakeword_poll_interval=float(merged.get("wakeword_poll_interval", defaults.wakeword_poll_interval)),
        whisper_backend=str(merged.get("whisper_backend", defaults.whisper_backend)),
        whisper_model=str(merged.get("whisper_model", defaults.whisper_model)),
        whisper_device=str(merged.get("whisper_device", defaults.whisper_device)),
        whisper_compute_type=str(merged.get("whisper_compute_type", defaults.whisper_compute_type)),
        sample_rate=int(merged.get("sample_rate", defaults.sample_rate)),
        detection_frame_seconds=float(merged.get("detection_frame_seconds", defaults.detection_frame_seconds)),
        capture_seconds=float(merged.get("capture_seconds", defaults.capture_seconds)),
        llm_backend=str(merged.get("llm_backend", defaults.llm_backend)),
        llm_model=str(merged.get("llm_model", defaults.llm_model)),
        llm_url=str(merged.get("llm_url", defaults.llm_url)),
        llm_max_tokens=int(merged.get("llm_max_tokens", defaults.llm_max_tokens)),
        llm_temperature=float(merged.get("llm_temperature", defaults.llm_temperature)),
        llm_top_p=float(merged.get("llm_top_p", defaults.llm_top_p)),
        llm_top_k=int(merged.get("llm_top_k", defaults.llm_top_k)),
        llm_seed=int(merged.get("llm_seed", defaults.llm_seed)),
        openai_api_key=merged.get("openai_api_key"),
        openai_model=merged.get("openai_model"),
        user_id=str(merged.get("user_id", defaults.user_id)),
        tts_backend=str(merged.get("tts_backend", defaults.tts_backend)),
        speak_language=str(merged.get("speak_language", defaults.speak_language)),
        voice_profile=str(merged.get("voice_profile", defaults.voice_profile)),
        rate_limit=str(merged.get("rate_limit", defaults.rate_limit)),
        allowed_origins=allowed_origins,
        brave_api_key=merged.get("brave_api_key"),
        serpapi_key=merged.get("serpapi_key"),
        transcripts_enabled=_coerce_bool(
            merged.get("transcripts_enabled", defaults.transcripts_enabled), defaults.transcripts_enabled
        ),
        transcripts_dir=Path(merged.get("transcripts_dir", defaults.transcripts_dir)),
        memory_backend=str(merged.get("memory_backend", defaults.memory_backend)),
        memory_path=Path(merged.get("memory_path", defaults.memory_path)),
        user_profiles_dir=Path(merged.get("user_profiles_dir", defaults.user_profiles_dir)),
        max_memory_items=int(merged.get("max_memory_items", defaults.max_memory_items)),
        memory_max_turns=int(merged.get("memory_max_turns", defaults.memory_max_turns)),
        default_user=merged.get("default_user"),
        audio_input_device=audio_input,
        audio_output_device=audio_output,
        command_duration=float(merged.get("command_duration", defaults.command_duration)),
        debug_logging=_coerce_bool(merged.get("debug_logging", defaults.debug_logging), defaults.debug_logging),
        conversation_export=_coerce_bool(
            merged.get("conversation_export", defaults.conversation_export), defaults.conversation_export
        ),
        search_providers=str(merged.get("search_providers", defaults.search_providers)),
        wake_sound_path=merged.get("wake_sound_path"),
        speak_api_key=merged.get("speak_api_key"),
    )

    config = _apply_env_overrides(config)
    config = _finalise_config(config)
    validate_config(config)

    if config.debug_logging:
        set_global_level(10)

    _cached_config = config
    return config


def validate_config(config: AppConfig) -> None:
    if not (0 < config.wakeword_threshold <= 1):
        raise ConfigurationError("wakeword_threshold must be between 0 and 1.")
    if config.command_duration <= 0:
        raise ConfigurationError("command_duration must be positive.")
    if config.memory_max_turns <= 0:
        raise ConfigurationError("memory_max_turns must be positive.")
    if config.max_memory_items <= 0:
        raise ConfigurationError("max_memory_items must be positive.")


settings = load_config()


def reload_settings() -> AppConfig:
    global settings
    settings = load_config(reload=True)
    return settings


def update_env_value(key: str, value: str) -> None:
    """Persist a configuration override into the shared .env file."""

    set_key(str(ENV_PATH), key, value)
    load_config(reload=True)


def persist_override(key: str, value: str) -> None:
    update_env_value(key, value)


def configure_cli(argv: Optional[Iterable[str]] = None) -> AppConfig:
    """Simple CLI for inspecting and updating configuration values."""

    parser = argparse.ArgumentParser(description="Inspect Rex configuration")
    parser.add_argument("--show", action="store_true", help="Print the current configuration values")
    parser.add_argument("--reload", action="store_true", help="Force a configuration reload")
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        help="Persist one or more configuration overrides",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.set:
        for pair in args.set:
            if "=" not in pair:
                raise ConfigurationError(f"Invalid --set argument (use key=value): {pair}")
            field, value = pair.split("=", 1)
            field = field.strip()
            env_key = ENV_MAPPING.get(field, field)
            update_env_value(env_key, value.strip())

    cfg = load_config(reload=args.reload or bool(args.set))

    if args.show:
        for key, value in cfg.to_dict().items():
            LOGGER.info("%s=%s", key, value)

    return cfg


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        configure_cli()
    except ConfigurationError as exc:
        LOGGER.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc
