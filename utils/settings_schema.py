"""Authoritative schema for Rex environment variables.

Defines canonical settings with explicit types to avoid heuristic detection failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class SettingDef:
    """Authoritative definition for a single setting."""
    key: str
    type: str  # bool, int, float, string, secret, file_path, dir_path, enum, multi_enum
    default: str
    description: str
    section: str
    required: bool = False
    advanced: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None  # For enum/multi_enum
    active_group: Optional[str] = None
    tooltip_low_effect: Optional[str] = None
    tooltip_high_effect: Optional[str] = None
    units: Optional[str] = None
    aliases: Optional[List[str]] = None  # Legacy keys that map to this canonical key


# Canonical key aliases - maps legacy key to canonical key
KEY_ALIASES: Dict[str, str] = {
    'WHISPER_MODEL': 'REX_WHISPER_MODEL',
    'WHISPER_DEVICE': 'REX_WHISPER_DEVICE',
    'REX_LLM_BACKEND': 'REX_LLM_PROVIDER',  # Prefer PROVIDER over BACKEND
    'REX_MODEL': 'OLLAMA_MODEL',  # REX_MODEL is used for Ollama
}

# Keys to hide from UI (handled via their canonical key)
HIDDEN_KEYS: Set[str] = set(KEY_ALIASES.keys())


def get_canonical_key(key: str) -> str:
    """Get the canonical key for a given key (or return the key if already canonical)."""
    return KEY_ALIASES.get(key, key)


def is_hidden_key(key: str) -> bool:
    """Check if this key should be hidden from the UI."""
    return key in HIDDEN_KEYS


# Authoritative schema - defines all known settings explicitly
AUTHORITATIVE_SETTINGS: List[SettingDef] = [
    # Core Settings
    SettingDef(
        key='REX_ACTIVE_USER',
        type='string',
        default='default',
        description='Active user profile (maps to Memory/<user_key>/core.json)',
        section='Core Settings',
    ),
    SettingDef(
        key='REX_LOG_LEVEL',
        type='enum',
        default='INFO',
        description='Logging level',
        section='Core Settings',
        options=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    ),
    SettingDef(
        key='REX_LOG_PATH',
        type='file_path',
        default='logs/rex.log',
        description='Path to log file',
        section='Core Settings',
    ),
    SettingDef(
        key='REX_ERROR_LOG_PATH',
        type='file_path',
        default='logs/error.log',
        description='Path to error log file',
        section='Core Settings',
    ),
    SettingDef(
        key='REX_DEBUG_LOGGING',
        type='bool',
        default='false',
        description='Enable debug logging',
        section='Core Settings',
        advanced=True,
    ),
    SettingDef(
        key='REX_FILE_LOGGING_ENABLED',
        type='bool',
        default='true',
        description='Enable file logging (set to false to only log to stdout)',
        section='Core Settings',
        advanced=True,
    ),

    # Wakeword Detection
    SettingDef(
        key='REX_WAKEWORD',
        type='string',
        default='rex',
        description='Wakeword phrase (e.g., "rex", "jarvis", "computer")',
        section='Wakeword Detection',
        required=True,
    ),
    SettingDef(
        key='REX_WAKEWORD_KEYWORD',
        type='string',
        default='hey_jarvis',
        description='Alternative wakeword keyword for openwakeword engine',
        section='Wakeword Detection',
    ),
    SettingDef(
        key='REX_WAKEWORD_BACKEND',
        type='enum',
        default='onnx',
        description='Wakeword backend',
        section='Wakeword Detection',
        options=['onnx', 'openwakeword', 'porcupine', 'none'],
        active_group='wakeword_backend_selector',
    ),
    SettingDef(
        key='REX_WAKEWORD_THRESHOLD',
        type='float',
        default='0.5',
        description='Detection threshold (0.0-1.0, higher = more strict)',
        section='Wakeword Detection',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        tooltip_low_effect='More sensitive, may trigger on noise',
        tooltip_high_effect='Less sensitive, requires clearer signal',
    ),
    SettingDef(
        key='REX_WAKEWORD_WINDOW',
        type='float',
        default='1.0',
        description='Wakeword detection window',
        section='Wakeword Detection',
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        units='seconds',
        advanced=True,
    ),
    SettingDef(
        key='REX_WAKEWORD_POLL_INTERVAL',
        type='float',
        default='0.05',
        description='Polling interval for wakeword detection',
        section='Wakeword Detection',
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        units='seconds',
        advanced=True,
    ),
    SettingDef(
        key='REX_DETECTION_FRAME_SECONDS',
        type='float',
        default='0.5',
        description='Frame duration for detection',
        section='Wakeword Detection',
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        units='seconds',
        advanced=True,
    ),

    # Audio Configuration
    SettingDef(
        key='REX_SAMPLE_RATE',
        type='int',
        default='16000',
        description='Audio sample rate',
        section='Audio Configuration',
        min_value=8000,
        max_value=48000,
        units='Hz',
        tooltip_low_effect='Lower audio quality',
        tooltip_high_effect='Higher audio quality',
    ),
    SettingDef(
        key='REX_COMMAND_DURATION',
        type='float',
        default='5.0',
        description='Command recording duration',
        section='Audio Configuration',
        min_value=1.0,
        max_value=30.0,
        step=0.5,
        units='seconds',
    ),
    SettingDef(
        key='REX_CAPTURE_SECONDS',
        type='float',
        default='5.0',
        description='Capture duration',
        section='Audio Configuration',
        min_value=1.0,
        max_value=30.0,
        step=0.5,
        units='seconds',
    ),
    SettingDef(
        key='REX_INPUT_DEVICE',
        type='string',
        default='',
        description='Audio input device index (leave empty for default)',
        section='Audio Configuration',
    ),
    SettingDef(
        key='REX_OUTPUT_DEVICE',
        type='string',
        default='',
        description='Audio output device index (leave empty for default)',
        section='Audio Configuration',
    ),
    SettingDef(
        key='REX_AUDIO_INPUT_DEVICE',
        type='string',
        default='',
        description='Audio input device (alternative key)',
        section='Audio Configuration',
    ),
    SettingDef(
        key='REX_AUDIO_OUTPUT_DEVICE',
        type='string',
        default='',
        description='Audio output device (alternative key)',
        section='Audio Configuration',
    ),
    SettingDef(
        key='REX_WAKE_SOUND',
        type='file_path',
        default='',
        description='Wake acknowledgment sound path (leave empty to disable)',
        section='Audio Configuration',
    ),

    # Speech Recognition (Whisper)
    SettingDef(
        key='REX_WHISPER_MODEL',
        type='enum',
        default='base',
        description='Whisper model size (larger models are more accurate but slower)',
        section='Speech Recognition',
        options=['tiny', 'base', 'small', 'medium', 'large'],
        active_group='whisper',
        aliases=['WHISPER_MODEL'],
    ),
    SettingDef(
        key='REX_WHISPER_DEVICE',
        type='enum',
        default='cpu',
        description='Device for Whisper',
        section='Speech Recognition',
        options=['cpu', 'cuda'],
        active_group='whisper',
        aliases=['WHISPER_DEVICE'],
    ),

    # Language Model (LLM)
    SettingDef(
        key='REX_LLM_PROVIDER',
        type='enum',
        default='transformers',
        description='LLM backend',
        section='Language Model',
        options=['transformers', 'openai', 'ollama'],
        active_group='llm_provider_selector',
        aliases=['REX_LLM_BACKEND'],
    ),
    SettingDef(
        key='REX_LLM_MODEL',
        type='string',
        default='distilgpt2',
        description='Model name or path',
        section='Language Model',
        active_group='transformers',
    ),
    SettingDef(
        key='REX_LLM_TEMPERATURE',
        type='float',
        default='0.7',
        description='LLM temperature',
        section='Language Model',
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        tooltip_low_effect='More focused and deterministic responses',
        tooltip_high_effect='More creative and varied responses',
    ),
    SettingDef(
        key='REX_LLM_MAX_TOKENS',
        type='int',
        default='120',
        description='Maximum tokens to generate',
        section='Language Model',
        min_value=10,
        max_value=4096,
        units='tokens',
        tooltip_low_effect='Shorter, more concise responses',
        tooltip_high_effect='Longer, more detailed responses',
    ),
    SettingDef(
        key='REX_LLM_TOP_P',
        type='float',
        default='0.9',
        description='Top-p sampling parameter',
        section='Language Model',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        advanced=True,
        tooltip_low_effect='More focused on likely words',
        tooltip_high_effect='Considers more word possibilities',
    ),
    SettingDef(
        key='REX_LLM_TOP_K',
        type='int',
        default='50',
        description='Top-k sampling parameter',
        section='Language Model',
        min_value=1,
        max_value=100,
        advanced=True,
        tooltip_low_effect='Fewer word choices considered',
        tooltip_high_effect='More word choices considered',
    ),
    SettingDef(
        key='REX_LLM_SEED',
        type='int',
        default='42',
        description='Random seed for reproducibility',
        section='Language Model',
        min_value=0,
        max_value=999999,
        advanced=True,
    ),

    # OpenAI API
    SettingDef(
        key='OPENAI_API_KEY',
        type='secret',
        default='',
        description='OpenAI API key (required if REX_LLM_PROVIDER=openai)',
        section='OpenAI API',
        active_group='openai',
    ),
    SettingDef(
        key='OPENAI_MODEL',
        type='string',
        default='gpt-3.5-turbo',
        description='OpenAI model name',
        section='OpenAI API',
        active_group='openai',
    ),
    SettingDef(
        key='OPENAI_BASE_URL',
        type='string',
        default='',
        description='OpenAI base URL (for custom endpoints, e.g., Azure OpenAI)',
        section='OpenAI API',
        active_group='openai',
    ),

    # Ollama API
    SettingDef(
        key='OLLAMA_HOST',
        type='string',
        default='http://127.0.0.1:11434',
        description='Ollama server host URL',
        section='Ollama API',
        active_group='ollama',
    ),
    SettingDef(
        key='OLLAMA_MODEL',
        type='string',
        default='llama3.1:8b',
        description='Ollama model name',
        section='Ollama API',
        active_group='ollama',
        aliases=['REX_MODEL'],
    ),
    SettingDef(
        key='OLLAMA_API_KEY',
        type='secret',
        default='',
        description='Ollama API key (only needed for Ollama cloud, not local)',
        section='Ollama API',
        active_group='ollama',
        advanced=True,
    ),
    SettingDef(
        key='OLLAMA_USE_CLOUD',
        type='bool',
        default='false',
        description='Use Ollama cloud instead of local server',
        section='Ollama API',
        active_group='ollama',
        advanced=True,
    ),

    # Text-to-Speech
    SettingDef(
        key='REX_TTS_PROVIDER',
        type='enum',
        default='xtts',
        description='TTS provider',
        section='Text-to-Speech',
        options=['xtts', 'edge', 'pyttsx3', 'piper'],
        active_group='tts_provider_selector',
    ),
    SettingDef(
        key='REX_TTS_MODEL',
        type='string',
        default='tts_models/multilingual/multi-dataset/xtts_v2',
        description='TTS model name (for Coqui TTS)',
        section='Text-to-Speech',
        active_group='xtts',
    ),
    SettingDef(
        key='REX_TTS_VOICE',
        type='string',
        default='en-US-AndrewNeural',
        description='TTS voice name (for edge-tts)',
        section='Text-to-Speech',
        active_group='edge',
    ),
    SettingDef(
        key='REX_PIPER_MODEL',
        type='file_path',
        default='voices/en_US-lessac-medium.onnx',
        description='Piper model path',
        section='Text-to-Speech',
        active_group='piper',
    ),
    SettingDef(
        key='REX_SPEAK_LANGUAGE',
        type='enum',
        default='en',
        description='TTS language code',
        section='Text-to-Speech',
        options=['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'ko'],
    ),

    # Memory & Context
    SettingDef(
        key='REX_MEMORY_MAX_ITEMS',
        type='int',
        default='50',
        description='Maximum conversation history items to retain',
        section='Memory & Context',
        min_value=1,
        max_value=1000,
        units='items',
    ),
    SettingDef(
        key='REX_MEMORY_MAX_TURNS',
        type='int',
        default='50',
        description='Maximum memory turns',
        section='Memory & Context',
        min_value=1,
        max_value=1000,
        units='turns',
    ),
    SettingDef(
        key='REX_MEMORY_MAX_BYTES',
        type='int',
        default='131072',
        description='Maximum memory file size',
        section='Memory & Context',
        min_value=1024,
        max_value=10485760,
        units='bytes',
        advanced=True,
    ),
    SettingDef(
        key='REX_TRANSCRIPTS_ENABLED',
        type='bool',
        default='true',
        description='Enable transcript saving',
        section='Memory & Context',
    ),
    SettingDef(
        key='REX_TRANSCRIPTS_DIR',
        type='dir_path',
        default='transcripts',
        description='Transcripts directory',
        section='Memory & Context',
    ),
    SettingDef(
        key='REX_CONVERSATION_EXPORT',
        type='bool',
        default='true',
        description='Enable conversation export',
        section='Memory & Context',
    ),

    # Web Search Plugins
    SettingDef(
        key='REX_SEARCH_PROVIDERS',
        type='multi_enum',
        default='serpapi,brave,duckduckgo',
        description='Search providers (comma-separated list)',
        section='Web Search Plugins',
        options=['serpapi', 'brave', 'duckduckgo', 'google'],
    ),
    SettingDef(
        key='SERPAPI_KEY',
        type='secret',
        default='',
        description='SerpAPI key',
        section='Web Search Plugins',
    ),
    SettingDef(
        key='SERPAPI_ENGINE',
        type='enum',
        default='google',
        description='SerpAPI search engine',
        section='Web Search Plugins',
        options=['google', 'bing', 'yahoo', 'baidu'],
        advanced=True,
    ),
    SettingDef(
        key='BRAVE_API_KEY',
        type='secret',
        default='',
        description='Brave Search API key',
        section='Web Search Plugins',
    ),
    SettingDef(
        key='GOOGLE_API_KEY',
        type='secret',
        default='',
        description='Google Custom Search API key',
        section='Web Search Plugins',
    ),
    SettingDef(
        key='GOOGLE_CSE_ID',
        type='string',
        default='',
        description='Google Custom Search engine ID',
        section='Web Search Plugins',
        advanced=True,
    ),
    SettingDef(
        key='BROWSERLESS_API_KEY',
        type='secret',
        default='',
        description='Browserless API key (for web scraping)',
        section='Web Search Plugins',
        advanced=True,
    ),
]


def get_authoritative_schema() -> Dict[str, SettingDef]:
    """Get the authoritative schema as a dictionary."""
    return {s.key: s for s in AUTHORITATIVE_SETTINGS}
