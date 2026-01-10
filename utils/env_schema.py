"""Parser for .env.example to extract schema, descriptions, and defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EnvVariable:
    """Represents a single environment variable."""
    key: str
    default_value: str
    description: str
    section: str
    is_required: bool = False
    is_secret: bool = False
    is_advanced: bool = False

    # UI control hints
    control_type: str = "entry"  # entry, dropdown, spinbox, checkbox, path
    dropdown_options: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    path_type: Optional[str] = None  # file, directory

    # Tooltip enhancements for numeric ranges
    tooltip_low_effect: Optional[str] = None
    tooltip_high_effect: Optional[str] = None
    units: Optional[str] = None

    # Active engine/module grouping
    active_group: Optional[str] = None  # e.g., "llm_provider", "tts_provider"

    def __post_init__(self):
        """Auto-detect control types and constraints."""
        self._detect_secret()
        self._detect_advanced()
        self._detect_control_type()
        self._detect_tooltip_info()
        self._detect_active_group()

    def _detect_secret(self):
        """Detect if this is a secret/API key field."""
        secret_keywords = [
            'api_key', 'token', 'secret', 'password', 'key'
        ]
        key_lower = self.key.lower()
        for keyword in secret_keywords:
            if keyword in key_lower:
                self.is_secret = True
                break

    def _detect_control_type(self):
        """Auto-detect the appropriate UI control type."""
        key_lower = self.key.lower()
        desc_lower = self.description.lower()

        # Boolean values
        if self.default_value.lower() in ('true', 'false', '1', '0'):
            self.control_type = 'checkbox'
            return

        # Known dropdowns - Log level
        if 'log_level' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            return

        # Device selection
        if 'device' in key_lower and 'audio' not in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['cpu', 'cuda']
            return

        # Whisper model
        if 'whisper_model' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['tiny', 'base', 'small', 'medium', 'large']
            return

        # TTS provider
        if 'tts_provider' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['xtts', 'edge', 'pyttsx3', 'piper']
            return

        # Wakeword backend
        if 'wakeword_backend' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['onnx', 'openwakeword', 'porcupine', 'none']
            return

        # LLM provider/backend
        if 'llm_provider' in key_lower or 'llm_backend' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['transformers', 'openai', 'ollama']
            return

        # Search providers
        if 'search_provider' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['serpapi', 'brave', 'duckduckgo', 'google']
            return

        # Language selection
        if 'language' in key_lower and 'speak' in key_lower:
            self.control_type = 'dropdown'
            self.dropdown_options = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'ko']
            return

        # Threshold (0.0-1.0)
        if 'threshold' in key_lower:
            self.control_type = 'spinbox'
            self.min_value = 0.0
            self.max_value = 1.0
            return

        # Numeric ranges
        if any(word in key_lower for word in ['duration', 'seconds', 'rate', 'window', 'interval']):
            try:
                float(self.default_value)
                self.control_type = 'spinbox'
                self.min_value = 0.0
                self.max_value = 3600.0  # Reasonable max
                return
            except ValueError:
                pass

        # Integer values
        if any(word in key_lower for word in ['max_', 'limit', 'tokens', 'items', 'turns', 'bytes', 'chars', 'timeout']):
            try:
                int(self.default_value)
                self.control_type = 'spinbox'
                self.min_value = 0
                self.max_value = 999999
                return
            except ValueError:
                pass

        # Paths and directories
        if any(word in key_lower for word in ['path', 'dir', 'file']) or any(word in desc_lower for word in ['path', 'directory', 'file']):
            self.control_type = 'path'
            if 'dir' in key_lower or 'directory' in desc_lower:
                self.path_type = 'directory'
            else:
                self.path_type = 'file'
            return

        # Default to entry
        self.control_type = 'entry'

    def _detect_advanced(self):
        """Detect if this is an advanced/rarely-used setting."""
        key_lower = self.key.lower()

        # Advanced settings keywords
        advanced_keywords = [
            'poll_interval', 'detection_frame', 'wakeword_window', 'seed',
            'top_k', 'top_p', 'verify_ssl', 'storage_uri', 'limiter',
            'proxy', 'rate_window', 'bytes', 'output_limit', 'frame_seconds',
            'browserless', 'cse', 'engine_id'
        ]

        for keyword in advanced_keywords:
            if keyword in key_lower:
                self.is_advanced = True
                return

        # File logging and debug options are advanced
        if 'debug' in key_lower or 'file_logging' in key_lower:
            self.is_advanced = True

    def _detect_tooltip_info(self):
        """Detect tooltip information for numeric ranges."""
        key_lower = self.key.lower()

        # Temperature settings
        if 'temperature' in key_lower:
            self.tooltip_low_effect = "More focused and deterministic responses"
            self.tooltip_high_effect = "More creative and varied responses"
            self.units = None
            if not self.min_value:
                self.min_value = 0.0
            if not self.max_value:
                self.max_value = 2.0

        # Threshold settings
        elif 'threshold' in key_lower:
            self.tooltip_low_effect = "More sensitive, may trigger on noise"
            self.tooltip_high_effect = "Less sensitive, requires clearer signal"
            self.units = None

        # Duration/time settings
        elif any(word in key_lower for word in ['duration', 'seconds', 'window', 'interval']):
            self.tooltip_low_effect = "Shorter time period"
            self.tooltip_high_effect = "Longer time period"
            self.units = "seconds"

        # Token limits
        elif 'token' in key_lower or 'max_' in key_lower:
            if 'token' in key_lower:
                self.tooltip_low_effect = "Shorter, more concise responses"
                self.tooltip_high_effect = "Longer, more detailed responses"
                self.units = "tokens"
            else:
                self.tooltip_low_effect = "Lower limit"
                self.tooltip_high_effect = "Higher limit"
                self.units = "items" if 'items' in key_lower or 'turns' in key_lower else None

        # Sample rate
        elif 'sample_rate' in key_lower or 'rate' in key_lower:
            self.tooltip_low_effect = "Lower audio quality"
            self.tooltip_high_effect = "Higher audio quality"
            self.units = "Hz" if 'sample' in key_lower else None

        # Top_p and top_k
        elif 'top_p' in key_lower:
            self.tooltip_low_effect = "More focused on likely words"
            self.tooltip_high_effect = "Considers more word possibilities"
            self.units = None

        elif 'top_k' in key_lower:
            self.tooltip_low_effect = "Fewer word choices considered"
            self.tooltip_high_effect = "More word choices considered"
            self.units = None

        # Timeout settings
        elif 'timeout' in key_lower:
            self.tooltip_low_effect = "Faster timeout, may fail on slow operations"
            self.tooltip_high_effect = "More time allowed, better for slow connections"
            self.units = "seconds"

    def _detect_active_group(self):
        """Detect if this setting belongs to an active engine group."""
        key_lower = self.key.lower()

        # LLM provider group
        if 'llm' in key_lower or 'openai' in key_lower or 'ollama' in key_lower:
            if 'provider' in key_lower or 'backend' in key_lower:
                self.active_group = 'llm_provider_selector'
            elif any(word in key_lower for word in ['model', 'api_key', 'base_url', 'temperature', 'max_tokens', 'top_']):
                # Determine which provider this belongs to
                if 'openai' in key_lower:
                    self.active_group = 'openai'
                elif 'ollama' in key_lower:
                    self.active_group = 'ollama'
                elif 'llm' in key_lower:
                    self.active_group = 'transformers'

        # TTS provider group
        elif 'tts' in key_lower or 'piper' in key_lower or 'edge' in key_lower or 'speak' in key_lower:
            if 'provider' in key_lower:
                self.active_group = 'tts_provider_selector'
            elif 'voice' in key_lower or 'model' in key_lower or 'piper' in key_lower:
                # Determine TTS engine
                if 'edge' in key_lower:
                    self.active_group = 'edge'
                elif 'piper' in key_lower:
                    self.active_group = 'piper'
                else:
                    self.active_group = 'xtts'

        # Wakeword backend group
        elif 'wakeword' in key_lower:
            if 'backend' in key_lower:
                self.active_group = 'wakeword_backend_selector'

        # Whisper settings
        elif 'whisper' in key_lower:
            self.active_group = 'whisper'


@dataclass
class EnvSection:
    """Represents a section in .env.example."""
    name: str
    variables: List[EnvVariable] = field(default_factory=list)


@dataclass
class EnvSchema:
    """Complete schema parsed from .env.example."""
    sections: List[EnvSection] = field(default_factory=list)

    def get_all_variables(self) -> List[EnvVariable]:
        """Get a flat list of all variables."""
        all_vars = []
        for section in self.sections:
            all_vars.extend(section.variables)
        return all_vars

    def get_variable(self, key: str) -> Optional[EnvVariable]:
        """Find a variable by key."""
        for var in self.get_all_variables():
            if var.key == key:
                return var
        return None


def parse_env_example(path: Path) -> EnvSchema:
    """Parse .env.example file into structured schema.

    Args:
        path: Path to .env.example file

    Returns:
        EnvSchema with all sections and variables
    """
    if not path.exists():
        raise FileNotFoundError(f".env.example not found at {path}")

    content = path.read_text(encoding='utf-8')
    lines = content.splitlines()

    schema = EnvSchema()
    current_section = None
    current_comments = []

    # Regex patterns
    section_pattern = re.compile(r'^#\s*={3,}')
    section_name_pattern = re.compile(r'^#\s+(.+?)(?:\s*#.*)?$')
    key_value_pattern = re.compile(r'^([A-Z_][A-Z0-9_]*)=(.*)$')
    comment_pattern = re.compile(r'^#\s+(.+)$')

    for line in lines:
        line = line.rstrip()

        # Skip empty lines (they reset comment accumulation)
        if not line.strip():
            # Only reset if we're not in the middle of a comment block
            if current_comments and not any(c.strip() for c in current_comments):
                current_comments = []
            continue

        # Section divider
        if section_pattern.match(line):
            # Next non-empty, non-divider line will be the section name
            continue

        # Section name (comment line after divider)
        if line.startswith('#') and '=' not in line:
            match = comment_pattern.match(line)
            if match:
                text = match.group(1).strip()

                # Check if this looks like a section name (title case, no lowercase text)
                if text and (text[0].isupper() or text.startswith('(')):
                    # Could be a section name if we don't have one yet or it looks like a title
                    if not current_section or (len(text.split()) <= 5 and not any(c in text for c in [',', '.', ':'])):
                        # Start new section
                        current_section = EnvSection(name=text)
                        schema.sections.append(current_section)
                        current_comments = []
                        continue

                # Otherwise it's a comment/description
                # Clean up dividers and extra symbols
                if not all(c in '=-_' for c in text):
                    current_comments.append(text)
            continue

        # Key=Value line
        match = key_value_pattern.match(line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()

            # Ensure we have a section
            if current_section is None:
                current_section = EnvSection(name="General")
                schema.sections.append(current_section)

            # Build description from accumulated comments
            description = ' '.join(current_comments).strip()

            # Check if required
            is_required = '(REQUIRED)' in description.upper() or '(required)' in description
            description = re.sub(r'\(REQUIRED\)', '', description, flags=re.IGNORECASE).strip()

            # Create variable
            var = EnvVariable(
                key=key,
                default_value=value,
                description=description or f"{key} setting",
                section=current_section.name,
                is_required=is_required
            )

            current_section.variables.append(var)
            current_comments = []

    return schema


def get_restart_required_keys() -> set:
    """Return set of environment keys that require restart when changed."""
    return {
        'REX_LLM_PROVIDER', 'REX_LLM_BACKEND', 'REX_LLM_MODEL',
        'REX_WHISPER_MODEL', 'REX_WHISPER_DEVICE', 'WHISPER_MODEL', 'WHISPER_DEVICE',
        'REX_WAKEWORD_BACKEND', 'REX_WAKEWORD',
        'REX_INPUT_DEVICE', 'REX_OUTPUT_DEVICE',
        'REX_AUDIO_INPUT_DEVICE', 'REX_AUDIO_OUTPUT_DEVICE',
        'REX_TTS_PROVIDER', 'REX_TTS_MODEL',
        'REX_DEVICE', 'REX_SAMPLE_RATE',
        'OPENAI_API_KEY', 'OLLAMA_HOST', 'OLLAMA_API_KEY',
        'REX_SPEAK_API_KEY'
    }


def is_restart_required(key: str) -> bool:
    """Check if changing this key requires restart."""
    return key in get_restart_required_keys()
