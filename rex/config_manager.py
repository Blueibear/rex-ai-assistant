"""Configuration manager for Rex AI Assistant.

Separates secrets (.env) from runtime configuration (rex_config.json).
Provides migration from legacy environment-based configuration.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rex.logging_utils import get_logger

logger = get_logger(__name__)

# Default configuration structure
DEFAULT_CONFIG: Dict[str, Any] = {
    "active_profile": "default",
    "profiles_dir": "profiles",
    "audio": {
        "input_device_index": None,
        "output_device_index": None,
        "sample_rate": 16000,
    },
    "wake_word": {
        "backend": "openwakeword",
        "wakeword": "rex",
        "keyword": None,
        "threshold": 0.5,
        "window": 1.0,
        "poll_interval": 0.01,
        "wake_sound_path": "assets/wake_acknowledgment.wav",
    },
    "models": {
        "llm_provider": "transformers",
        "llm_backend": None,
        "llm_model": "sshleifer/tiny-gpt2",
        "llm_max_tokens": 120,
        "llm_temperature": 0.7,
        "llm_top_p": 0.9,
        "llm_top_k": 50,
        "llm_seed": 42,
        "stt_model": "base",
        "stt_device": "cpu",
        "tts_provider": "xtts",
        "tts_model": None,
        "tts_voice": None,
        "windows_tts_voice_index": None,
    },
    "runtime": {
        "log_level": "INFO",
        "file_logging_enabled": False,
        "memory_max_bytes": 131072,
        "transcripts_enabled": True,
        "transcripts_dir": "transcripts",
        "active_user": None,
        "user_id": "default",
        "memory_max_turns": 50,
        "command_duration": 5.0,
        "capture_seconds": 5.0,
        "detection_frame_seconds": 1.0,
        "conversation_export": True,
        "speak_language": "en",
    },
    "search": {
        "providers": "serpapi,brave,duckduckgo,google",
    },
    "home_assistant": {
        "base_url": None,
        "verify_ssl": True,
        "timeout": 10.0,
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "use_cloud": False,
    },
    "openai": {
        "model": None,
        "base_url": None,
    },
    "api": {
        "rate_limit": "30/minute",
        "allowed_origins": ["*"],
    },
    "ui": {
        "start_minimized": False,
    },
}

# Map of legacy environment variable names to config paths
ENV_TO_CONFIG_MAPPING: Dict[str, str] = {
    # Audio settings
    "REX_SAMPLE_RATE": "audio.sample_rate",
    "REX_INPUT_DEVICE": "audio.input_device_index",
    "REX_OUTPUT_DEVICE": "audio.output_device_index",
    "REX_AUDIO_INPUT_DEVICE": "audio.input_device_index",
    "REX_AUDIO_OUTPUT_DEVICE": "audio.output_device_index",
    "REX_DEVICE": "audio.input_device_index",
    # Wake word settings
    "REX_WAKEWORD_BACKEND": "wake_word.backend",
    "REX_WAKEWORD": "wake_word.wakeword",
    "REX_WAKEWORD_KEYWORD": "wake_word.keyword",
    "REX_WAKEWORD_THRESHOLD": "wake_word.threshold",
    "REX_WAKEWORD_WINDOW": "wake_word.window",
    "REX_WAKEWORD_POLL_INTERVAL": "wake_word.poll_interval",
    "REX_WAKE_SOUND": "wake_word.wake_sound_path",
    # Model settings
    "REX_LLM_PROVIDER": "models.llm_provider",
    "REX_LLM_BACKEND": "models.llm_backend",
    "REX_LLM_MODEL": "models.llm_model",
    "REX_LLM_MAX_TOKENS": "models.llm_max_tokens",
    "REX_LLM_TEMPERATURE": "models.llm_temperature",
    "REX_LLM_TOP_P": "models.llm_top_p",
    "REX_LLM_TOP_K": "models.llm_top_k",
    "REX_LLM_SEED": "models.llm_seed",
    "REX_WHISPER_MODEL": "models.stt_model",
    "REX_WHISPER_DEVICE": "models.stt_device",
    "REX_TTS_PROVIDER": "models.tts_provider",
    "REX_TTS_MODEL": "models.tts_model",
    "REX_TTS_VOICE": "models.tts_voice",
    "REX_WINDOWS_TTS_VOICE_INDEX": "models.windows_tts_voice_index",
    # Runtime settings
    "REX_LOG_LEVEL": "runtime.log_level",
    "REX_FILE_LOGGING_ENABLED": "runtime.file_logging_enabled",
    "REX_MEMORY_MAX_BYTES": "runtime.memory_max_bytes",
    "REX_TRANSCRIPTS_ENABLED": "runtime.transcripts_enabled",
    "REX_TRANSCRIPTS_DIR": "runtime.transcripts_dir",
    "REX_ACTIVE_USER": "runtime.active_user",
    "REX_USER_ID": "runtime.user_id",
    "REX_MEMORY_MAX_TURNS": "runtime.memory_max_turns",
    "REX_COMMAND_DURATION": "runtime.command_duration",
    "REX_CAPTURE_SECONDS": "runtime.capture_seconds",
    "REX_DETECTION_FRAME_SECONDS": "runtime.detection_frame_seconds",
    "REX_CONVERSATION_EXPORT": "runtime.conversation_export",
    "REX_SPEAK_LANGUAGE": "runtime.speak_language",
    "REX_DEBUG_LOGGING": "runtime.log_level",  # Special: maps "true" to "DEBUG"
    # Search settings
    "REX_SEARCH_PROVIDERS": "search.providers",
    # Home Assistant
    "HA_BASE_URL": "home_assistant.base_url",
    "HA_VERIFY_SSL": "home_assistant.verify_ssl",
    "HA_TIMEOUT": "home_assistant.timeout",
    # Ollama
    "OLLAMA_HOST": "ollama.base_url",
    "OLLAMA_BASE_URL": "ollama.base_url",
    "OLLAMA_USE_CLOUD": "ollama.use_cloud",
    # OpenAI
    "OPENAI_MODEL": "openai.model",
    "OPENAI_BASE_URL": "openai.base_url",
    # API settings
    "REX_RATE_LIMIT": "api.rate_limit",
    "REX_ALLOWED_ORIGINS": "api.allowed_origins",
}

# Secret environment variables that should stay in .env
SECRET_ENV_VARS = {
    "OPENAI_API_KEY",
    "BRAVE_API_KEY",
    "SERPAPI_KEY",
    "GOOGLE_API_KEY",
    "BROWSERLESS_API_KEY",
    "REX_PROXY_TOKEN",
    "REX_SPEAK_API_KEY",
    "HASS_SECRET",
    "HA_SECRET",
    "HA_TOKEN",
    "FLASK_LIMITER_STORAGE_URI",
    "REX_SPEAK_STORAGE_URI",
    "SERPAPI_URL",
    "BRAVE_URL",
    "GOOGLE_URL",
    "BROWSERLESS_URL",
    "GOOGLE_CSE_ID",
    "OLLAMA_API_KEY",
}


def _get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get value from nested dict using dot notation path."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def _set_nested(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set value in nested dict using dot notation path."""
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in data:
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: str) -> Optional[int]:
    """Parse integer from string, handling float-formatted strings."""
    try:
        float_val = float(value)
        if float_val.is_integer():
            return int(float_val)
    except (ValueError, OverflowError):
        pass
    return None


def _parse_float(value: str) -> Optional[float]:
    """Parse float from string."""
    try:
        return float(value)
    except (ValueError, OverflowError):
        return None


def load_config(path: str | Path = "config/rex_config.json") -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        path: Path to config file (default: config/rex_config.json)

    Returns:
        Configuration dict with all settings

    Behavior:
        - Creates directory if missing
        - Creates default file if missing
        - If invalid JSON, renames to .invalid.<timestamp>.json and recreates defaults
    """
    config_path = Path(path)

    # Ensure config directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist, create it with defaults
    if not config_path.exists():
        logger.info(f"Config file not found, creating default: {config_path}")
        save_config(DEFAULT_CONFIG.copy(), path)
        return DEFAULT_CONFIG.copy()

    # Try to load existing config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Merge with defaults to ensure all keys exist
        merged = _deep_merge(DEFAULT_CONFIG.copy(), config)
        logger.debug(f"Loaded config from {config_path}")
        return merged

    except json.JSONDecodeError as exc:
        # Invalid JSON - backup and recreate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f".invalid.{timestamp}.json")
        shutil.copy2(config_path, backup_path)
        logger.error(
            f"Invalid JSON in config file: {exc}. "
            f"Backed up to {backup_path} and recreating with defaults."
        )

        save_config(DEFAULT_CONFIG.copy(), path)
        return DEFAULT_CONFIG.copy()

    except Exception as exc:
        logger.error(f"Failed to load config from {config_path}: {exc}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], path: str | Path = "config/rex_config.json") -> None:
    """Save configuration to JSON file.

    Args:
        config: Configuration dict to save
        path: Path to config file (default: config/rex_config.json)

    Output format:
        - Pretty printed JSON with 2-space indentation
        - Sorted keys for stable ordering
        - Newline at end of file
    """
    config_path = Path(path)

    # Ensure config directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")  # Newline at EOF
        logger.info(f"Saved config to {config_path}")
    except Exception as exc:
        logger.error(f"Failed to save config to {config_path}: {exc}")
        raise


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts, with overlay taking precedence."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def migrate_legacy_env_to_config(
    env_path: str | Path = ".env",
    config_path: str | Path = "config/rex_config.json",
) -> List[str]:
    """Migrate legacy non-secret environment variables to rex_config.json.

    Args:
        env_path: Path to .env file
        config_path: Path to rex_config.json

    Returns:
        List of migration notes and warnings

    Behavior:
        - Only migrates non-secret settings from .env
        - Only migrates if config value is null or default
        - Never migrates secrets (they stay in .env)
        - Returns list of human-readable migration notes
    """
    env_file = Path(env_path)
    if not env_file.exists():
        return ["No .env file found, no migration needed"]

    notes: List[str] = []

    # Load current config
    config = load_config(config_path)

    # Read .env file
    env_vars: Dict[str, str] = {}
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except Exception as exc:
        notes.append(f"Failed to read .env file: {exc}")
        return notes

    # Check for legacy non-secret variables
    migrated_count = 0
    for env_key, config_path_str in ENV_TO_CONFIG_MAPPING.items():
        if env_key in env_vars and env_key not in SECRET_ENV_VARS:
            env_value = env_vars[env_key]
            current_value = _get_nested(config, config_path_str)

            # Get default value for comparison
            default_value = _get_nested(DEFAULT_CONFIG, config_path_str)

            # Only migrate if current config value is null or matches default
            should_migrate = current_value is None or current_value == default_value

            if should_migrate and env_value:
                # Parse value based on type
                parsed_value: Any = env_value

                # Special handling for specific keys
                if env_key == "REX_DEBUG_LOGGING":
                    # Map debug_logging boolean to log_level
                    if _parse_bool(env_value):
                        parsed_value = "DEBUG"
                        config_path_str = "runtime.log_level"
                    else:
                        continue  # Don't migrate if false

                elif "THRESHOLD" in env_key or "WINDOW" in env_key or "INTERVAL" in env_key or "DURATION" in env_key or "TIMEOUT" in env_key:
                    parsed_value = _parse_float(env_value) or parsed_value

                elif "DEVICE" in env_key and "INDEX" in env_key or env_key in ["REX_INPUT_DEVICE", "REX_OUTPUT_DEVICE", "REX_DEVICE", "REX_AUDIO_INPUT_DEVICE", "REX_AUDIO_OUTPUT_DEVICE"]:
                    parsed_value = _parse_int(env_value)

                elif any(x in env_key for x in ["MAX_TOKENS", "TOP_K", "SEED", "MAX_TURNS", "SAMPLE_RATE"]):
                    parsed_value = _parse_int(env_value) or parsed_value

                elif any(x in env_key for x in ["ENABLED", "EXPORT", "USE_CLOUD", "VERIFY_SSL"]):
                    parsed_value = _parse_bool(env_value)

                elif env_key == "REX_ALLOWED_ORIGINS":
                    # Parse comma-separated list
                    parsed_value = [x.strip() for x in env_value.split(",") if x.strip()]

                # Set value in config
                _set_nested(config, config_path_str, parsed_value)
                notes.append(f"Migrated {env_key} -> {config_path_str} = {parsed_value}")
                migrated_count += 1

    # Save config if anything was migrated
    if migrated_count > 0:
        save_config(config, config_path)
        notes.append(f"\nMigrated {migrated_count} settings from .env to {config_path}")
        notes.append("These environment variables are now ignored. Use rex_config.json for runtime settings.")
    else:
        notes.append("No legacy environment variables found that needed migration")

    return notes


def get_legacy_env_warnings() -> List[str]:
    """Check for legacy non-secret environment variables and return warnings.

    Returns:
        List of warning messages about legacy env vars that should be in config
    """
    warnings = []

    for env_key in ENV_TO_CONFIG_MAPPING.keys():
        if env_key not in SECRET_ENV_VARS and os.getenv(env_key):
            config_path = ENV_TO_CONFIG_MAPPING[env_key]
            warnings.append(
                f"Legacy setting {env_key} found in environment. "
                f"Use {config_path} in rex_config.json instead."
            )

    return warnings


__all__ = [
    "DEFAULT_CONFIG",
    "load_config",
    "save_config",
    "migrate_legacy_env_to_config",
    "get_legacy_env_warnings",
    "ENV_TO_CONFIG_MAPPING",
    "SECRET_ENV_VARS",
]
