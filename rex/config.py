"""Central configuration loader and CLI utilities for the Rex assistant.

Now uses rex_config.json for non-secret settings and .env only for secrets.
"""

from __future__ import annotations

# ruff: noqa: I001, UP006, UP035, UP045

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
from rex.profile_manager import (
    DEFAULT_PROFILES_DIR,
    apply_profile,
    get_active_profile_name,
    load_profile,
)

LOGGER = get_logger(__name__)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _parse_int(name: str, value: Optional[str], *, default: int = 0) -> int:
    """Parse integer from string value.

    Args:
        name: Parameter name (for error messages, unused)
        value: String value to parse
        default: Default value if parsing fails

    Returns:
        Parsed integer or default
    """
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def resolve_wakeword_keyword(
    keyword: Optional[str],
    wakeword: Optional[str],
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    for candidate in (keyword, wakeword, default):
        if candidate is None:
            continue
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


@dataclass
class AppConfig:
    """Application configuration combining JSON config and environment secrets."""

    wakeword: str = "rex"
    wakeword_backend: str = "openwakeword"
    wakeword_threshold: float = 0.5
    wakeword_window: float = 1.0
    wakeword_poll_interval: float = 0.01
    wakeword_model_path: Optional[str] = None
    wakeword_embedding_path: Optional[str] = None
    wakeword_fallback_to_builtin: bool = True
    wakeword_fallback_keyword: str = "hey jarvis"
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
    tts_provider: str = "xtts"
    tts_voice: Optional[str] = None
    tts_speed: float = 1.08

    speak_api_key: Optional[str] = None
    rate_limit: str = "30/minute"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

    memory_max_turns: int = 50
    transcripts_enabled: bool = True
    transcripts_dir: Path = Path("transcripts")
    default_user: Optional[str] = None
    wake_sound_path: Optional[str] = None

    active_profile: str = "default"
    capabilities: List[str] = field(default_factory=list)

    audio_input_device: Optional[int] = None
    audio_output_device: Optional[int] = None

    debug_logging: bool = False
    file_logging_enabled: bool = False
    memory_max_bytes: int = 131072
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

    followups_enabled: bool = False
    followups_max_per_session: int = 2
    followups_lookback_hours: int = 72
    followups_expire_hours: int = 168

    # Home Assistant integration
    ha_base_url: Optional[str] = None
    ha_token: Optional[str] = None
    ha_secret: Optional[str] = None
    ha_verify_ssl: bool = True
    ha_timeout: float = 10.0
    ha_entity_map: Optional[Dict[str, str]] = None

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
        provider = self.llm_provider.lower()
        local_path_providers = {"transformers"}
        if provider in local_path_providers and isinstance(self.llm_model, str):
            model_path = Path(self.llm_model)
            if model_path.is_absolute() or ".." in model_path.parts:
                raise ValueError("llm_model must not contain path traversal components.")
        if provider == "openai" and not self.openai_model:
            raise ValueError("openai.model must be set when llm_provider is 'openai'.")
        if self.llm_backend is None:
            self.llm_backend = self.llm_provider
        if self.temperature is None:
            self.temperature = self.llm_temperature
        if self.max_memory_items is None:
            self.max_memory_items = self.memory_max_turns
        self.wakeword_keyword = resolve_wakeword_keyword(self.wakeword_keyword, self.wakeword)

_cached_config: Optional[AppConfig] = None

# Required environment variables (secrets only)
REQUIRED_ENV_KEYS: set = set()  # No required env vars - secrets are optional

# Backward compatibility: ENV_MAPPING removed - use rex_config.json for runtime settings
# For migration, see rex.config_manager.ENV_TO_CONFIG_MAPPING
ENV_MAPPING: Dict[str, str] = {}


def _get_nested(data: dict, path: str, default=None):
    """Get value from nested dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def _merge_profile_config(base_config: dict) -> dict:
    profile_name = get_active_profile_name(base_config)
    profiles_dir = base_config.get("profiles_dir", DEFAULT_PROFILES_DIR)
    profile = load_profile(profile_name, profiles_dir=profiles_dir)
    merged_config = apply_profile(base_config, profile)
    merged_config["active_profile"] = profile_name
    merged_config.setdefault("profiles_dir", profiles_dir)
    merged_config["capabilities"] = profile.get("capabilities", [])
    return merged_config


def build_app_config(json_config: dict) -> AppConfig:
    """Build an AppConfig from a merged JSON configuration."""
    # Parse allowed origins from JSON config
    allowed_origins_value = _get_nested(json_config, "api.allowed_origins", ["*"])
    if isinstance(allowed_origins_value, str):
        allowed_origins = [
            origin.strip().rstrip("/")
            for origin in allowed_origins_value.split(",")
            if origin.strip()
        ] or ["*"]
    elif isinstance(allowed_origins_value, list):
        allowed_origins = [str(o).strip().rstrip("/") for o in allowed_origins_value if o]
    else:
        allowed_origins = ["*"]

    capabilities_value = _get_nested(json_config, "capabilities", [])
    if isinstance(capabilities_value, list):
        capabilities = [str(item) for item in capabilities_value if item]
    else:
        capabilities = []

    # Build config from JSON config + env secrets
    config = AppConfig(
        # Wake word settings from JSON
        wakeword=_get_nested(json_config, "wake_word.wakeword", "rex"),
        wakeword_backend=_get_nested(json_config, "wake_word.backend", "openwakeword"),
        wakeword_keyword=_get_nested(json_config, "wake_word.keyword"),
        wakeword_threshold=float(_get_nested(json_config, "wake_word.threshold", 0.5)),
        wakeword_window=float(_get_nested(json_config, "wake_word.window", 1.0)),
        wakeword_poll_interval=float(_get_nested(json_config, "wake_word.poll_interval", 0.01)),
        wake_sound_path=_get_nested(json_config, "wake_word.wake_sound_path"),
        wakeword_model_path=_get_nested(json_config, "wake_word.model_path"),
        wakeword_embedding_path=_get_nested(json_config, "wake_word.embedding_path"),
        wakeword_fallback_to_builtin=bool(_get_nested(json_config, "wake_word.fallback_to_builtin", True)),
        wakeword_fallback_keyword=_get_nested(json_config, "wake_word.fallback_keyword", "hey jarvis"),

        # Runtime settings from JSON
        command_duration=float(_get_nested(json_config, "runtime.command_duration", 5.0)),
        detection_frame_seconds=float(_get_nested(json_config, "runtime.detection_frame_seconds", 1.0)),
        capture_seconds=float(_get_nested(json_config, "runtime.capture_seconds", 5.0)),
        memory_max_turns=int(_get_nested(json_config, "runtime.memory_max_turns", 50)),
        transcripts_enabled=bool(_get_nested(json_config, "runtime.transcripts_enabled", True)),
        transcripts_dir=Path(_get_nested(json_config, "runtime.transcripts_dir", "transcripts")),
        default_user=_get_nested(json_config, "runtime.active_user"),
        conversation_export=bool(_get_nested(json_config, "runtime.conversation_export", True)),
        speak_language=_get_nested(json_config, "runtime.speak_language", "en"),
        user_id=_get_nested(json_config, "runtime.user_id", "default"),

        # Audio settings from JSON
        sample_rate=int(_get_nested(json_config, "audio.sample_rate", 16000)),
        audio_input_device=_get_nested(json_config, "audio.input_device_index"),
        audio_output_device=_get_nested(json_config, "audio.output_device_index"),

        # Model settings from JSON
        whisper_model=_get_nested(json_config, "models.stt_model", "base"),
        whisper_device=_get_nested(json_config, "models.stt_device", "cpu"),
        llm_provider=_get_nested(json_config, "models.llm_provider", "transformers"),
        llm_model=_get_nested(json_config, "models.llm_model", "sshleifer/tiny-gpt2"),
        llm_max_tokens=int(_get_nested(json_config, "models.llm_max_tokens", 120)),
        llm_temperature=float(_get_nested(json_config, "models.llm_temperature", 0.7)),
        llm_top_p=float(_get_nested(json_config, "models.llm_top_p", 0.9)),
        llm_top_k=int(_get_nested(json_config, "models.llm_top_k", 50)),
        llm_seed=int(_get_nested(json_config, "models.llm_seed", 42)),
        tts_provider=_get_nested(json_config, "models.tts_provider", "xtts"),
        tts_voice=_get_nested(json_config, "models.tts_voice"),
        tts_speed=float(_get_nested(json_config, "models.tts_speed", 1.08)),

        # API settings from JSON
        rate_limit=_get_nested(json_config, "api.rate_limit", "30/minute"),
        allowed_origins=allowed_origins,

        # Search settings from JSON
        search_providers=_get_nested(json_config, "search.providers", "serpapi,brave,duckduckgo,google"),

        # Home Assistant from JSON + secrets from env
        ha_base_url=_get_nested(json_config, "home_assistant.base_url"),
        ha_verify_ssl=bool(_get_nested(json_config, "home_assistant.verify_ssl", True)),
        ha_timeout=float(_get_nested(json_config, "home_assistant.timeout", 10.0)),
        ha_token=os.getenv("HA_TOKEN"),  # SECRET from env
        ha_secret=os.getenv("HA_SECRET"),  # SECRET from env
        ha_entity_map=None,

        # Ollama from JSON + secrets from env
        ollama_base_url=_get_nested(json_config, "ollama.base_url", "http://localhost:11434"),
        ollama_use_cloud=bool(_get_nested(json_config, "ollama.use_cloud", False)),
        ollama_api_key=os.getenv("OLLAMA_API_KEY"),  # SECRET from env

        # OpenAI from JSON + secrets from env
        openai_model=_get_nested(json_config, "openai.model"),
        openai_base_url=_get_nested(json_config, "openai.base_url"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # SECRET from env

        # All secrets from env only
        brave_api_key=os.getenv("BRAVE_API_KEY"),
        speak_api_key=os.getenv("REX_SPEAK_API_KEY"),

        # Logging from JSON
        debug_logging=_get_nested(json_config, "runtime.log_level", "INFO").upper() == "DEBUG",
        file_logging_enabled=bool(_get_nested(json_config, "runtime.file_logging_enabled", False)),
        memory_max_bytes=int(_get_nested(json_config, "runtime.memory_max_bytes", 131072)),

        # Profile metadata
        active_profile=_get_nested(json_config, "active_profile", "default"),
        capabilities=capabilities,

        # Conversational followups
        followups_enabled=bool(_get_nested(json_config, "conversation.followups.enabled", False)),
        followups_max_per_session=int(_get_nested(json_config, "conversation.followups.max_per_session", 2)),
        followups_lookback_hours=int(_get_nested(json_config, "conversation.followups.lookback_hours", 72)),
        followups_expire_hours=int(_get_nested(json_config, "conversation.followups.expire_hours", 168)),
    )

    return config


def load_config(*, env_path: Optional[Path] = None, reload: bool = False, json_config: Optional[dict] = None) -> AppConfig:
    """Load configuration from rex_config.json and .env secrets.

    Args:
        env_path: Path to .env file (default: repo root .env)
        reload: Force reload instead of using cached config
        json_config: Pre-loaded JSON config dict (if None, loads from rex/config_manager)

    Returns:
        AppConfig with runtime settings from JSON and secrets from .env

    Note:
        Non-secret environment variables are now ignored. Use rex_config.json instead.
    """
    global _cached_config
    if not reload and json_config is None:
        config_module = sys.modules.get("config")
        if config_module is not None:
            cached = getattr(config_module, "_cached_config", None)
            if cached is not None:
                _cached_config = cached
                return _cached_config
        if _cached_config is not None:
            return _cached_config

    # Load .env for secrets only
    load_dotenv(env_path or ENV_PATH, override=False)

    # Load JSON config for runtime settings
    if json_config is None:
        from rex.config_manager import load_config as load_json_config, get_legacy_env_warnings
        json_config = load_json_config()

    # Warn about legacy environment variables
    warnings = get_legacy_env_warnings()
    if warnings:
        for warning in warnings[:3]:  # Limit to first 3 to avoid spam
            print(warning, file=sys.stderr)
        if len(warnings) > 3:
            print(
                f"... and {len(warnings) - 3} more legacy env vars. "
                f"Run 'rex-config migrate-legacy-env' to migrate all.",
                file=sys.stderr,
            )

        try:
            json_config = _merge_profile_config(json_config)
        except Exception as exc:
            raise ConfigurationError(f"Profile loading failed: {exc}") from exc

    config = build_app_config(json_config)

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


def reload_settings(*, env_path: Optional[Path] = None, json_config: Optional[dict] = None) -> AppConfig:
    """Reload configuration, optionally with new JSON config."""
    return load_config(env_path=env_path, reload=True, json_config=json_config)


def show_config(config: Optional[AppConfig] = None) -> None:
    """Print the resolved configuration to stdout in stable JSON format."""
    cfg = config or load_config()
    print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str))


def _cmd_show(args: argparse.Namespace) -> int:
    """Print the current configuration."""
    cfg = load_config(env_path=ENV_PATH, reload=True)
    show_config(cfg)
    return 0


def _cmd_migrate_legacy_env(args: argparse.Namespace) -> int:
    """Migrate legacy environment variables to rex_config.json."""
    from rex.config_manager import migrate_legacy_env_to_config

    env_path = Path(args.env_path) if args.env_path else ENV_PATH
    notes = migrate_legacy_env_to_config(
        env_path=env_path,
        config_path=args.config_path,
        dry_run=args.dry_run,
    )
    for note in notes:
        print(note)
    return 0


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rex-config",
        description="Configure Rex Assistant",
    )
    subparsers = parser.add_subparsers(dest="command")

    # show
    show_parser = subparsers.add_parser(
        "show",
        help="Print current configuration",
    )
    show_parser.set_defaults(func=_cmd_show)

    # migrate-legacy-env
    migrate_parser = subparsers.add_parser(
        "migrate-legacy-env",
        help="Migrate legacy environment variables into config/rex_config.json",
        description=(
            "Reads legacy non-secret environment variables (e.g. OPENAI_BASE_URL) "
            "and writes their values into config/rex_config.json. Existing non-default "
            "config values are never overwritten."
        ),
    )
    migrate_parser.add_argument(
        "--config-path",
        default="config/rex_config.json",
        help="Path to rex_config.json (default: config/rex_config.json)",
    )
    migrate_parser.add_argument(
        "--env-path",
        default=None,
        help=(
            "Path to .env file to read legacy variables from "
            "(default: repo root .env)"
        ),
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing any changes",
    )
    migrate_parser.set_defaults(func=_cmd_migrate_legacy_env)

    # Backward compat: --show and --reload still work
    parser.add_argument("--show", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reload", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    ENV_PATH.touch(exist_ok=True)

    # Handle legacy flags
    if args.show or args.reload:
        cfg = load_config(env_path=ENV_PATH, reload=True)
        show_config(cfg)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


settings = load_config()
Settings = AppConfig

if __name__ == "__main__":
    import sys
    try:
        raise SystemExit(cli())
    except ConfigurationError as exc:
        LOGGER.error("Config error: %s", exc)
        raise SystemExit(1) from exc
