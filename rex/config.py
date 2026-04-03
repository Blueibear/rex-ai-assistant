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

from rex.exception_handler import wrap_entrypoint
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv, set_key
except ImportError:

    def load_dotenv(*args, **kwargs):  # type: ignore[misc]
        return False

    def set_key(env_path: str, key: str, value: str):  # type: ignore[misc]
        path = Path(env_path)
        lines = (
            [line for line in path.read_text().splitlines() if not line.startswith(f"{key}=")]
            if path.exists()
            else []
        )
        lines.append(f"{key}={value}")
        path.write_text("\n".join(lines) + "\n")
        return key, value, True


from rex.assistant_errors import ConfigurationError
from rex.config_manager import get_legacy_env_warnings
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
class ModelRoutingConfig:
    """Maps task categories to LLM model identifiers.

    Each field accepts a model identifier string (e.g. ``"gpt-4o"``,
    ``"llama3"``) or an empty string to fall back to the global
    ``AppConfig.llm_model`` setting.  All fields are optional.
    """

    default: str = ""
    coding: str = ""
    reasoning: str = ""
    search: str = ""
    vision: str = ""
    fast: str = ""


@dataclass
class EmailAccountConfig:
    """Configuration for a single email account (IMAP read + SMTP send)."""

    id: str
    address: str
    imap_host: str
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 587
    credential_ref: str = ""
    use_starttls: bool = True


@dataclass
class UserEmailAccount:
    """Per-user email account entry (US-ME-001).

    Lightweight descriptor that names an email account, its backend type, and
    the ``.env`` key where credentials are stored.  Full connection details
    (host, port, etc.) are resolved at runtime from the credentials key.
    """

    account_id: str
    display_name: str = ""
    backend: str = "imap"  # "imap" | "gmail" | "outlook"
    credentials_key: str = ""  # e.g. "EMAIL_ALICE_WORK" in .env


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
    whisper_device: str = "auto"
    whisper_language: Optional[str] = "en"
    llm_provider: str = "transformers"
    llm_model: str = "sshleifer/tiny-gpt2"
    llm_max_tokens: int = 120
    voice_max_tokens: int = 150
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
    session_ttl_hours: int = 8
    default_user: Optional[str] = None
    wake_sound_path: Optional[str] = None
    acknowledgment_sound: str = "chime"  # "chime", a .wav path, or a spoken filler phrase
    response_cache_ttl: float = 300.0  # seconds; 0 disables response caching

    active_profile: str = "default"
    capabilities: List[str] = field(default_factory=list)

    audio_input_device: Optional[int] = None
    audio_output_device: Optional[int | str] = None

    debug_logging: bool = False
    file_logging_enabled: bool = False
    memory_max_bytes: int = 131072
    conversation_export: bool = True

    brave_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None

    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None

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

    # Integration credential detection
    email_provider: str = "none"  # none | gmail | outlook

    # Multi-account email config (US-208)
    email_accounts: List[EmailAccountConfig] = field(default_factory=list)
    email_default_account_id: str = ""

    # Per-user multi-email accounts (US-ME-001)
    # Keyed by user_id; each value is the list of email accounts for that user.
    user_email_accounts: Dict[str, List[UserEmailAccount]] = field(default_factory=dict)

    # Location and weather
    default_location: Optional[str] = None
    default_timezone: Optional[str] = None
    openweathermap_api_key: Optional[str] = None

    # Conversation history persistence
    persist_history: bool = True
    history_db_path: Path = field(default_factory=lambda: Path("data/history.db"))
    history_retention_days: int = 30

    # Autonomy budget limits (0 = unlimited)
    autonomy_budget_per_plan_usd: float = 0.0
    autonomy_budget_per_step_usd: float = 0.0

    # OpenClaw integration
    use_openclaw_tools: bool = False
    use_openclaw_voice_backend: bool = False
    openclaw_gateway_url: str = ""
    openclaw_gateway_timeout: int = 30
    openclaw_gateway_max_retries: int = 3
    openclaw_gateway_token: Optional[str] = None

    # Model routing
    model_routing: ModelRoutingConfig = field(default_factory=ModelRoutingConfig)

    # Voice identity
    speaker_id_threshold: float = 0.75

    # Tool dispatch
    tool_timeout_seconds: float = 10.0

    # Local file access allowlist (US-WIN-001)
    allowed_file_roots: List[str] = field(default_factory=lambda: [str(Path.home())])

    # Windows settings — require user confirmation before applying system changes (US-WIN-003)
    require_confirm_system_changes: bool = True

    # Web UI (US-UI-001)
    ui_enabled: bool = True

    # Shopping list PWA (US-SL-004) — optional PIN; empty/None means no auth
    shopping_pwa_pin: Optional[str] = None

    # Smart speaker TTS output (US-SP-002) — name of discovered speaker, or None for local audio
    tts_output_device: Optional[str] = None

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
        if provider == "anthropic" and not self.anthropic_model:
            raise ValueError("anthropic.model must be set when llm_provider is 'anthropic'.")
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


def _coerce_float(json_config: dict, path: str, default: float) -> float:
    """Get a float config value, warning if the raw value is a string.

    Pydantic/dataclass coercion silently accepts string-typed floats from
    JSON, which can hide misconfigured ``rex_config.json`` files.  This
    helper logs a WARNING so operators know to fix the source file.
    """
    raw = _get_nested(json_config, path, default)
    if isinstance(raw, str):
        LOGGER.warning(
            "Config field %r has string value %r — expected float; "
            "coercing automatically.  Fix the value in rex_config.json.",
            path,
            raw,
        )
    return float(raw)


def _coerce_int(json_config: dict, path: str, default: int) -> int:
    """Get an int config value, warning if the raw value is a string.

    See :func:`_coerce_float` for rationale.
    """
    raw = _get_nested(json_config, path, default)
    if isinstance(raw, str):
        LOGGER.warning(
            "Config field %r has string value %r — expected int; "
            "coercing automatically.  Fix the value in rex_config.json.",
            path,
            raw,
        )
    return int(float(raw))


def _parse_email_accounts(raw: object) -> List[EmailAccountConfig]:
    """Parse ``email.accounts`` from JSON config into a list of EmailAccountConfig."""
    if not isinstance(raw, list):
        return []
    accounts: List[EmailAccountConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            accounts.append(
                EmailAccountConfig(
                    id=str(item["id"]),
                    address=str(item["address"]),
                    imap_host=str(item.get("imap_host", item.get("imap", {}).get("host", ""))),
                    imap_port=int(item.get("imap_port", item.get("imap", {}).get("port", 993))),
                    smtp_host=str(item.get("smtp_host", item.get("smtp", {}).get("host", ""))),
                    smtp_port=int(item.get("smtp_port", item.get("smtp", {}).get("port", 587))),
                    credential_ref=str(item.get("credential_ref", "")),
                    use_starttls=bool(item.get("use_starttls", True)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping malformed email account entry: %s", exc)
    return accounts


def _parse_user_email_account(raw: dict) -> UserEmailAccount:
    """Parse a single user email account dict."""
    return UserEmailAccount(
        account_id=str(raw["account_id"]),
        display_name=str(raw.get("display_name", "")),
        backend=str(raw.get("backend", "imap")).lower(),
        credentials_key=str(raw.get("credentials_key", "")),
    )


def _parse_user_email_accounts(
    users_block: object,
    legacy_email_accounts: object,
) -> Dict[str, List[UserEmailAccount]]:
    """Parse ``users.{user_id}.email_accounts`` into a per-user dict.

    Migration shim: if the new ``users`` block is absent or empty but the
    legacy ``email.accounts`` list is present, its entries are migrated to the
    ``"default"`` user using their ``id`` as ``account_id`` and ``credential_ref``
    as ``credentials_key``.
    """
    result: Dict[str, List[UserEmailAccount]] = {}

    # Parse new format: users.{user_id}.email_accounts
    if isinstance(users_block, dict):
        for user_id, user_data in users_block.items():
            if not isinstance(user_data, dict):
                continue
            accounts_raw = user_data.get("email_accounts", [])
            if not isinstance(accounts_raw, list):
                continue
            parsed: List[UserEmailAccount] = []
            for entry in accounts_raw:
                if not isinstance(entry, dict) or "account_id" not in entry:
                    continue
                try:
                    parsed.append(_parse_user_email_account(entry))
                except (KeyError, TypeError, ValueError) as exc:
                    LOGGER.warning("Skipping malformed user email account: %s", exc)
            if parsed:
                result[str(user_id)] = parsed

    # Migration shim: promote legacy email.accounts to user "default" if no new entries
    if not result and isinstance(legacy_email_accounts, list):
        migrated: List[UserEmailAccount] = []
        for item in legacy_email_accounts:
            if not isinstance(item, dict):
                continue
            account_id = str(item.get("id", ""))
            if not account_id:
                continue
            migrated.append(
                UserEmailAccount(
                    account_id=account_id,
                    display_name=str(item.get("address", "")),
                    backend="imap",
                    credentials_key=str(item.get("credential_ref", "")),
                )
            )
        if migrated:
            result["default"] = migrated

    return result


def _parse_model_routing(raw: object) -> ModelRoutingConfig:
    """Parse ``model_routing`` block from JSON config."""
    if not isinstance(raw, dict):
        return ModelRoutingConfig()
    return ModelRoutingConfig(
        default=str(raw.get("default", "")),
        coding=str(raw.get("coding", "")),
        reasoning=str(raw.get("reasoning", "")),
        search=str(raw.get("search", "")),
        vision=str(raw.get("vision", "")),
        fast=str(raw.get("fast", "")),
    )


def _parse_allowed_file_roots(raw: object) -> list[str]:
    """Parse ``file_ops.allowed_roots`` list from JSON config.

    Returns a list of root path strings.  Falls back to the user home directory
    when *raw* is not a non-empty list.
    """
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if item]
    return [str(Path.home())]


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
        wakeword_threshold=_coerce_float(json_config, "wake_word.threshold", 0.5),
        wakeword_window=_coerce_float(json_config, "wake_word.window", 1.0),
        wakeword_poll_interval=_coerce_float(json_config, "wake_word.poll_interval", 0.01),
        wake_sound_path=_get_nested(json_config, "wake_word.wake_sound_path"),
        acknowledgment_sound=_get_nested(json_config, "acknowledgment.sound", "chime"),
        response_cache_ttl=_coerce_float(json_config, "response_cache.ttl", 300.0),
        wakeword_model_path=_get_nested(json_config, "wake_word.model_path"),
        wakeword_embedding_path=_get_nested(json_config, "wake_word.embedding_path"),
        wakeword_fallback_to_builtin=bool(
            _get_nested(json_config, "wake_word.fallback_to_builtin", True)
        ),
        wakeword_fallback_keyword=_get_nested(
            json_config, "wake_word.fallback_keyword", "hey jarvis"
        ),
        # Runtime settings from JSON
        command_duration=_coerce_float(json_config, "runtime.command_duration", 5.0),
        detection_frame_seconds=_coerce_float(json_config, "runtime.detection_frame_seconds", 1.0),
        capture_seconds=_coerce_float(json_config, "runtime.capture_seconds", 5.0),
        memory_max_turns=_coerce_int(json_config, "runtime.memory_max_turns", 50),
        transcripts_enabled=bool(_get_nested(json_config, "runtime.transcripts_enabled", True)),
        transcripts_dir=Path(_get_nested(json_config, "runtime.transcripts_dir", "transcripts")),
        session_ttl_hours=_coerce_int(json_config, "runtime.session_ttl_hours", 8),
        default_user=_get_nested(json_config, "runtime.active_user"),
        conversation_export=bool(_get_nested(json_config, "runtime.conversation_export", True)),
        speak_language=_get_nested(json_config, "runtime.speak_language", "en"),
        user_id=_get_nested(json_config, "runtime.user_id", "default"),
        # Audio settings from JSON
        sample_rate=_coerce_int(json_config, "audio.sample_rate", 16000),
        audio_input_device=_get_nested(json_config, "audio.input_device_index"),
        audio_output_device=_get_nested(json_config, "audio.output_device_index"),
        tts_output_device=_get_nested(json_config, "audio.tts_output_device"),
        # Model settings from JSON
        whisper_model=_get_nested(json_config, "models.stt_model", "base"),
        whisper_device=_get_nested(json_config, "models.stt_device", "auto"),
        whisper_language=_get_nested(json_config, "models.stt_language", "en"),
        llm_provider=_get_nested(json_config, "models.llm_provider", "transformers"),
        llm_model=_get_nested(json_config, "models.llm_model", "sshleifer/tiny-gpt2"),
        llm_max_tokens=_coerce_int(json_config, "models.llm_max_tokens", 120),
        llm_temperature=_coerce_float(json_config, "models.llm_temperature", 0.7),
        llm_top_p=_coerce_float(json_config, "models.llm_top_p", 0.9),
        llm_top_k=_coerce_int(json_config, "models.llm_top_k", 50),
        llm_seed=_coerce_int(json_config, "models.llm_seed", 42),
        tts_provider=_get_nested(json_config, "models.tts_provider", "xtts"),
        tts_voice=_get_nested(json_config, "models.tts_voice"),
        tts_speed=_coerce_float(json_config, "models.tts_speed", 1.08),
        # API settings from JSON
        rate_limit=_get_nested(json_config, "api.rate_limit", "30/minute"),
        allowed_origins=allowed_origins,
        # Search settings from JSON
        search_providers=_get_nested(
            json_config, "search.providers", "serpapi,brave,duckduckgo,google"
        ),
        # Home Assistant from JSON + secrets from env
        ha_base_url=_get_nested(json_config, "home_assistant.base_url"),
        ha_verify_ssl=bool(_get_nested(json_config, "home_assistant.verify_ssl", True)),
        ha_timeout=_coerce_float(json_config, "home_assistant.timeout", 10.0),
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
        # Anthropic from JSON + secrets from env
        anthropic_model=_get_nested(json_config, "anthropic.model"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),  # SECRET from env
        # All secrets from env only
        brave_api_key=os.getenv("BRAVE_API_KEY"),
        speak_api_key=os.getenv("REX_SPEAK_API_KEY"),
        # Logging from JSON
        debug_logging=_get_nested(json_config, "runtime.log_level", "INFO").upper() == "DEBUG",
        file_logging_enabled=bool(_get_nested(json_config, "runtime.file_logging_enabled", False)),
        memory_max_bytes=_coerce_int(json_config, "runtime.memory_max_bytes", 131072),
        # Profile metadata
        active_profile=_get_nested(json_config, "active_profile", "default"),
        capabilities=capabilities,
        # Location and weather (location from JSON, API key from env)
        default_location=_get_nested(json_config, "location.default_location"),
        default_timezone=_get_nested(json_config, "location.default_timezone"),
        openweathermap_api_key=os.getenv("OPENWEATHERMAP_API_KEY"),
        # Conversational followups
        followups_enabled=bool(_get_nested(json_config, "conversation.followups.enabled", False)),
        followups_max_per_session=_coerce_int(
            json_config, "conversation.followups.max_per_session", 2
        ),
        followups_lookback_hours=_coerce_int(
            json_config, "conversation.followups.lookback_hours", 72
        ),
        followups_expire_hours=_coerce_int(json_config, "conversation.followups.expire_hours", 168),
        # OpenClaw integration
        use_openclaw_tools=bool(_get_nested(json_config, "openclaw.use_tools", False)),
        use_openclaw_voice_backend=bool(
            _get_nested(json_config, "openclaw.use_voice_backend", False)
        ),
        openclaw_gateway_url=_get_nested(json_config, "openclaw.gateway_url", ""),
        openclaw_gateway_timeout=_coerce_int(json_config, "openclaw.gateway_timeout", 30),
        openclaw_gateway_max_retries=_coerce_int(json_config, "openclaw.gateway_max_retries", 3),
        openclaw_gateway_token=os.getenv("OPENCLAW_GATEWAY_TOKEN"),  # SECRET from env
        # Multi-account email (US-208)
        email_accounts=_parse_email_accounts(_get_nested(json_config, "email.accounts", [])),
        email_default_account_id=_get_nested(json_config, "email.default_account_id", ""),
        # Per-user multi-email accounts (US-ME-001)
        user_email_accounts=_parse_user_email_accounts(
            _get_nested(json_config, "users", {}),
            _get_nested(json_config, "email.accounts", []),
        ),
        # History persistence
        persist_history=bool(_get_nested(json_config, "runtime.persist_history", True)),
        history_db_path=Path(
            _get_nested(json_config, "runtime.history_db_path", "data/history.db")
        ),
        history_retention_days=_coerce_int(json_config, "runtime.history_retention_days", 30),
        # Model routing
        model_routing=_parse_model_routing(_get_nested(json_config, "model_routing", {})),
        # Voice identity
        speaker_id_threshold=_coerce_float(
            json_config, "voice_identity.speaker_id_threshold", 0.75
        ),
        # Local file access allowlist
        allowed_file_roots=_parse_allowed_file_roots(
            _get_nested(json_config, "file_ops.allowed_roots", [])
        ),
        # Windows settings confirmation
        require_confirm_system_changes=bool(
            _get_nested(json_config, "windows.require_confirm_system_changes", True)
        ),
    )

    return config


def load_config(
    *, env_path: Optional[Path] = None, reload: bool = False, json_config: Optional[dict] = None
) -> AppConfig:
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
        from rex.config_manager import load_config as load_json_config

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


def reload_settings(
    *, env_path: Optional[Path] = None, json_config: Optional[dict] = None
) -> AppConfig:
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


@wrap_entrypoint
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
        help=("Path to .env file to read legacy variables from " "(default: repo root .env)"),
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

    return args.func(args)  # type: ignore[no-any-return]


settings = load_config()
Settings = AppConfig

if __name__ == "__main__":
    import sys

    try:
        raise SystemExit(cli())
    except ConfigurationError as exc:
        LOGGER.error("Config error: %s", exc)
        raise SystemExit(1) from exc
