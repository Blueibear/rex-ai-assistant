"""Central configuration management for the Rex assistant.

Unified configuration that replaces both config.py and rex/config.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hmac
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rex.assistant_errors import ConfigurationError

logger = logging.getLogger(__name__)

# Optional dependencies
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

try:
    from pydantic import BaseSettings, Field
    _HAS_PYDANTIC = True
except ImportError:
    BaseSettings = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _HAS_PYDANTIC = False


@dataclasses.dataclass
class Settings:
    """Unified settings for Rex AI Assistant."""
    
    # Wake word configuration
    wakeword: str = "rex"
    wakeword_keyword: str = "hey_jarvis"
    wakeword_threshold: float = 0.5
    wakeword_window: float = 1.0
    wakeword_poll_interval: float = 0.05
    
    # Audio configuration
    sample_rate: int = 16000
    detection_frame_seconds: float = 0.5
    capture_seconds: float = 5.0
    command_duration: float = 5.0
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    
    # Speech recognition
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    
    # LLM configuration
    llm_provider: str = "transformers"
    llm_backend: str = "transformers"
    llm_model: str = "distilgpt2"
    llm_max_tokens: int = 120
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k: int = 50
    llm_seed: int = 42
    temperature: float = 0.8  # Alias for llm_temperature
    
    # API keys (sensitive)
    speak_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    brave_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    
    # TTS configuration
    tts_provider: str = "xtts"
    speak_language: str = "en"
    
    # Memory and storage
    max_memory_items: int = 50
    memory_max_turns: int = 50
    transcripts_enabled: bool = True
    transcripts_dir: str = "transcripts"
    conversation_export: bool = True
    
    # User configuration
    user_id: str = "james"
    default_user: Optional[str] = None
    
    # Rate limiting
    rate_limit: str = "30/minute"
    allowed_origins: List[str] = dataclasses.field(default_factory=lambda: ["*"])
    
    # Paths
    wake_sound_path: Optional[str] = None
    log_path: str = "logs/rex.log"
    error_log_path: str = "logs/error.log"
    
    # Search providers
    search_providers: str = "serpapi,brave,duckduckgo"
    
    # Debug
    debug_logging: bool = False

    # MQTT configuration
    mqtt_broker: str = "localhost"
    mqtt_port: int = 8883
    mqtt_tls: bool = True
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_client_id: str = "rex-core"
    mqtt_keepalive: int = 60
    mqtt_tls_ca: Optional[str] = None
    mqtt_tls_cert: Optional[str] = None
    mqtt_tls_key: Optional[str] = None
    mqtt_tls_insecure: bool = False
    mqtt_watchdog_interval: int = 30
    mqtt_watchdog_timeout: int = 90
    mqtt_node_id: str = "rex_core"
    ha_base_url: Optional[str] = None
    ha_token: Optional[str] = None
    ha_secret: Optional[str] = None
    ha_verify_ssl: bool = True
    ha_timeout: float = 5.0
    ha_entity_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Prevent path traversal
        if ".." in Path(self.llm_model).parts:
            raise ConfigurationError("llm_model must not contain path traversal")
        
        # Validate ranges
        if not (0 < self.wakeword_threshold <= 1):
            raise ConfigurationError("wakeword_threshold must be between 0 and 1")
        if self.command_duration <= 0:
            raise ConfigurationError("command_duration must be positive")
        if self.llm_max_tokens <= 0:
            raise ConfigurationError("llm_max_tokens must be positive")
        if not (0 <= self.llm_temperature <= 5.0):
            raise ConfigurationError("llm_temperature must be between 0 and 5")
        if self.mqtt_port <= 0:
            raise ConfigurationError("mqtt_port must be positive")
        if self.mqtt_keepalive <= 0:
            raise ConfigurationError("mqtt_keepalive must be positive")
        if self.mqtt_watchdog_interval <= 0:
            raise ConfigurationError("mqtt_watchdog_interval must be positive")
        if self.mqtt_watchdog_timeout <= 0:
            raise ConfigurationError("mqtt_watchdog_timeout must be positive")
        if self.ha_base_url:
            self.ha_base_url = self.ha_base_url.rstrip("/")
        if isinstance(self.ha_entity_map, str):
            try:
                self.ha_entity_map = json.loads(self.ha_entity_map)  # type: ignore[assignment]
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"Invalid HA entity map JSON: {exc}") from exc
        if not isinstance(self.ha_entity_map, dict):
            raise ConfigurationError("ha_entity_map must be a dict of aliases to entity IDs")
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return dataclasses.asdict(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Alias for dict() for backward compatibility."""
        result = self.dict()
        result["transcripts_dir"] = str(self.transcripts_dir)
        return result
    
    def validate_api_key(self, provided_key: Optional[str], expected_key: Optional[str]) -> bool:
        """Securely validate API key using constant-time comparison."""
        if not provided_key or not expected_key:
            return False
        try:
            return hmac.compare_digest(provided_key, expected_key)
        except TypeError:
            return False


_ENV_MAPPING: Dict[str, Sequence[str]] = {
    "wakeword": ("REX_WAKEWORD",),
    "wakeword_keyword": ("REX_WAKEWORD_KEYWORD",),
    "wakeword_threshold": ("REX_WAKEWORD_THRESHOLD",),
    "wakeword_window": ("REX_WAKEWORD_WINDOW",),
    "wakeword_poll_interval": ("REX_WAKEWORD_POLL_INTERVAL",),
    "sample_rate": ("REX_SAMPLE_RATE",),
    "detection_frame_seconds": ("REX_DETECTION_FRAME_SECONDS",),
    "capture_seconds": ("REX_CAPTURE_SECONDS",),
    "command_duration": ("REX_COMMAND_DURATION",),
    "input_device": ("REX_INPUT_DEVICE", "REX_AUDIO_INPUT_DEVICE"),
    "output_device": ("REX_OUTPUT_DEVICE", "REX_AUDIO_OUTPUT_DEVICE"),
    "whisper_model": ("REX_WHISPER_MODEL", "WHISPER_MODEL"),
    "whisper_device": ("REX_WHISPER_DEVICE", "WHISPER_DEVICE"),
    "llm_provider": ("REX_LLM_PROVIDER",),
    "llm_backend": ("REX_LLM_BACKEND",),
    "llm_model": ("REX_LLM_MODEL",),
    "llm_max_tokens": ("REX_LLM_MAX_TOKENS",),
    "llm_temperature": ("REX_LLM_TEMPERATURE",),
    "llm_top_p": ("REX_LLM_TOP_P",),
    "llm_top_k": ("REX_LLM_TOP_K",),
    "llm_seed": ("REX_LLM_SEED",),
    "temperature": ("REX_LLM_TEMPERATURE",),
    "speak_api_key": ("REX_SPEAK_API_KEY",),
    "openai_api_key": ("OPENAI_API_KEY",),
    "openai_model": ("OPENAI_MODEL",),
    "openai_base_url": ("OPENAI_BASE_URL",),
    "brave_api_key": ("BRAVE_API_KEY",),
    "serpapi_key": ("SERPAPI_KEY",),
    "tts_provider": ("REX_TTS_PROVIDER",),
    "speak_language": ("REX_SPEAK_LANGUAGE",),
    "max_memory_items": ("REX_MEMORY_MAX_ITEMS", "REX_MEMORY_MAX_TURNS"),
    "memory_max_turns": ("REX_MEMORY_MAX_TURNS",),
    "transcripts_enabled": ("REX_TRANSCRIPTS_ENABLED",),
    "transcripts_dir": ("REX_TRANSCRIPTS_DIR",),
    "conversation_export": ("REX_CONVERSATION_EXPORT",),
    "user_id": ("REX_ACTIVE_USER",),
    "default_user": ("REX_ACTIVE_USER",),
    "rate_limit": ("REX_RATE_LIMIT",),
    "wake_sound_path": ("REX_WAKE_SOUND",),
    "log_path": ("REX_LOG_PATH",),
    "error_log_path": ("REX_ERROR_LOG_PATH",),
    "search_providers": ("REX_SEARCH_PROVIDERS",),
    "debug_logging": ("REX_DEBUG_LOGGING",),
    "mqtt_broker": ("REX_MQTT_BROKER",),
    "mqtt_port": ("REX_MQTT_PORT",),
    "mqtt_tls": ("REX_MQTT_TLS",),
    "mqtt_username": ("REX_MQTT_USERNAME",),
    "mqtt_password": ("REX_MQTT_PASSWORD",),
    "mqtt_client_id": ("REX_MQTT_CLIENT_ID",),
    "mqtt_keepalive": ("REX_MQTT_KEEPALIVE",),
    "mqtt_tls_ca": ("REX_MQTT_TLS_CA", "REX_MQTT_CA"),
    "mqtt_tls_cert": ("REX_MQTT_TLS_CERT",),
    "mqtt_tls_key": ("REX_MQTT_TLS_KEY",),
    "mqtt_tls_insecure": ("REX_MQTT_TLS_INSECURE",),
    "mqtt_watchdog_interval": ("REX_MQTT_WATCHDOG_INTERVAL",),
    "mqtt_watchdog_timeout": ("REX_MQTT_WATCHDOG_TIMEOUT",),
    "mqtt_node_id": ("REX_MQTT_NODE_ID",),
    "ha_base_url": ("REX_HA_BASE_URL",),
    "ha_token": ("REX_HA_TOKEN",),
    "ha_secret": ("REX_HA_SECRET",),
    "ha_verify_ssl": ("REX_HA_VERIFY_SSL",),
    "ha_timeout": ("REX_HA_TIMEOUT",),
    "ha_entity_map": ("REX_HA_ENTITY_MAP",),
}


def _cast_value(key: str, raw: str) -> Any:
    """Cast environment variable string to appropriate type."""
    float_keys = {
        "temperature", "wakeword_threshold", "detection_frame_seconds",
        "capture_seconds", "wakeword_poll_interval", "wakeword_window",
        "command_duration", "llm_temperature", "llm_top_p", "ha_timeout"
    }
    int_keys = {
        "max_memory_items", "sample_rate", "memory_max_turns",
        "llm_max_tokens", "llm_top_k", "llm_seed",
        "mqtt_port", "mqtt_keepalive", "mqtt_watchdog_interval", "mqtt_watchdog_timeout"
    }
    bool_keys = {
        "transcripts_enabled", "conversation_export", "debug_logging",
        "mqtt_tls", "mqtt_tls_insecure", "ha_verify_ssl"
    }
    optional_int_keys = {"input_device", "output_device"}
    json_keys = {"ha_entity_map"}
    
    if key in float_keys:
        return float(raw)
    elif key in int_keys:
        return int(raw)
    elif key in bool_keys:
        return raw.lower() in {"1", "true", "yes", "on"}
    elif key in optional_int_keys:
        return int(raw) if raw not in {"", "none", "None"} else None
    elif key in json_keys:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON for %s: %s (%s)", key, raw, exc)
            return {}
    return raw


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    """Load settings from environment with caching."""
    load_dotenv()
    
    values: Dict[str, Any] = {}
    for field, aliases in _ENV_MAPPING.items():
        for env_var in aliases:
            raw = os.getenv(env_var)
            if raw is None or raw == "":
                continue
            try:
                values[field] = _cast_value(field, raw)
                break
            except (TypeError, ValueError) as exc:
                logger.warning("Invalid value for %s: %s (%s)", env_var, raw, exc)
    
    # Handle allowed_origins specially
    if "REX_ALLOWED_ORIGINS" in os.environ:
        values["allowed_origins"] = [
            origin.strip().rstrip("/")
            for origin in os.environ["REX_ALLOWED_ORIGINS"].split(",")
            if origin.strip()
        ] or ["*"]
    
    return Settings(**values)


# Module-level settings instance
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


def validate_config(config: Settings) -> None:
    """Validate configuration (calls __post_init__ internally)."""
    config.__post_init__()


def update_env_value(key: str, value: str) -> None:
    """Persist a configuration value into the .env file."""
    env_path = Path.cwd() / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(env_path), key, value)
    reload_settings()


def _cli(argv: Iterable[str] | None = None) -> int:
    """CLI for managing configuration."""
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
        for key, value in sorted(settings.dict().items()):
            print(f"{key}: {value}")
        return 0
    
    parser.print_help()
    return 0


# Backward compatibility aliases
AppConfig = Settings
ENV_PATH = Path.cwd() / ".env"
ENV_MAPPING = _ENV_MAPPING


__all__ = [
    "Settings",
    "AppConfig",
    "settings",
    "load_config",
    "reload_settings",
    "validate_config",
    "update_env_value",
    "ConfigurationError",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
