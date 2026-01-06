"""Tests for env_schema parser."""

import tempfile
from pathlib import Path

import pytest

from utils.env_schema import EnvVariable, parse_env_example, is_restart_required


@pytest.fixture
def sample_env_example(tmp_path):
    """Create a sample .env.example file for testing."""
    content = """# ================================
# Rex AI Assistant Configuration
# ================================
# Copy this file to .env and fill in your values

# ================================
# Core Settings
# ================================

# Active user profile (maps to Memory/<user_key>/core.json)
REX_ACTIVE_USER=default

# Logging configuration
REX_LOG_LEVEL=INFO
REX_DEBUG_LOGGING=false

# ================================
# Wakeword Detection
# ================================

# Wakeword phrase (e.g., "rex", "jarvis", "computer")
REX_WAKEWORD=rex

# Wakeword backend: onnx, openwakeword
REX_WAKEWORD_BACKEND=onnx

# Detection threshold (0.0-1.0, higher = more strict)
REX_WAKEWORD_THRESHOLD=0.5

# ================================
# Language Model (LLM)
# ================================

# LLM backend: transformers, openai, ollama
REX_LLM_BACKEND=transformers

# Model name or path (REQUIRED if REX_LLM_BACKEND=transformers)
# For transformers: distilgpt2, gpt2, sshleifer/tiny-gpt2
# For OpenAI: gpt-3.5-turbo, gpt-4
# For Ollama: llama2, mistral, codellama
REX_LLM_MODEL=distilgpt2

# LLM generation parameters
REX_LLM_TEMPERATURE=0.7
REX_LLM_MAX_TOKENS=120

# ================================
# OpenAI API (if using OpenAI backend)
# ================================

# OpenAI API key (REQUIRED if REX_LLM_BACKEND=openai)
OPENAI_API_KEY=

# Whisper model size: tiny, base, small, medium, large
REX_WHISPER_MODEL=base
"""
    env_example = tmp_path / ".env.example"
    env_example.write_text(content)
    return env_example


def test_parse_env_example_basic(sample_env_example):
    """Test basic parsing of .env.example."""
    schema = parse_env_example(sample_env_example)

    assert len(schema.sections) > 0
    all_vars = schema.get_all_variables()
    assert len(all_vars) > 0

    # Check specific variables
    active_user = schema.get_variable("REX_ACTIVE_USER")
    assert active_user is not None
    assert active_user.default_value == "default"
    assert "Active user profile" in active_user.description


def test_parse_env_example_sections(sample_env_example):
    """Test that sections are properly identified."""
    schema = parse_env_example(sample_env_example)

    section_names = [s.name for s in schema.sections]
    assert "Core Settings" in section_names
    assert "Wakeword Detection" in section_names
    assert "Language Model (LLM)" in section_names


def test_parse_env_example_descriptions(sample_env_example):
    """Test that descriptions are extracted correctly."""
    schema = parse_env_example(sample_env_example)

    # Test description extraction
    wakeword = schema.get_variable("REX_WAKEWORD")
    assert wakeword is not None
    assert "Wakeword phrase" in wakeword.description

    threshold = schema.get_variable("REX_WAKEWORD_THRESHOLD")
    assert threshold is not None
    assert "threshold" in threshold.description.lower()


def test_parse_env_example_required_fields(sample_env_example):
    """Test detection of required fields."""
    schema = parse_env_example(sample_env_example)

    # OPENAI_API_KEY is marked as REQUIRED
    openai_key = schema.get_variable("OPENAI_API_KEY")
    assert openai_key is not None
    assert openai_key.is_required


def test_control_type_detection():
    """Test automatic control type detection."""
    # Boolean
    var = EnvVariable(
        key="REX_DEBUG",
        default_value="true",
        description="Enable debug mode",
        section="Test"
    )
    assert var.control_type == "checkbox"

    # Dropdown - Log level
    var = EnvVariable(
        key="REX_LOG_LEVEL",
        default_value="INFO",
        description="Log level",
        section="Test"
    )
    assert var.control_type == "dropdown"
    assert "INFO" in var.dropdown_options

    # Dropdown - Whisper model
    var = EnvVariable(
        key="REX_WHISPER_MODEL",
        default_value="base",
        description="Whisper model size",
        section="Test"
    )
    assert var.control_type == "dropdown"
    assert "tiny" in var.dropdown_options
    assert "base" in var.dropdown_options

    # Spinbox - Threshold
    var = EnvVariable(
        key="REX_WAKEWORD_THRESHOLD",
        default_value="0.5",
        description="Detection threshold",
        section="Test"
    )
    assert var.control_type == "spinbox"
    assert var.min_value == 0.0
    assert var.max_value == 1.0


def test_secret_detection():
    """Test automatic secret field detection."""
    # API key
    var = EnvVariable(
        key="OPENAI_API_KEY",
        default_value="",
        description="OpenAI API key",
        section="Test"
    )
    assert var.is_secret

    # Token
    var = EnvVariable(
        key="REX_PROXY_TOKEN",
        default_value="",
        description="Proxy token",
        section="Test"
    )
    assert var.is_secret

    # Not a secret
    var = EnvVariable(
        key="REX_LOG_LEVEL",
        default_value="INFO",
        description="Log level",
        section="Test"
    )
    assert not var.is_secret


def test_restart_required():
    """Test restart required detection."""
    assert is_restart_required("REX_LLM_PROVIDER")
    assert is_restart_required("REX_WHISPER_MODEL")
    assert is_restart_required("OPENAI_API_KEY")
    assert not is_restart_required("REX_DEBUG_LOGGING")
    assert not is_restart_required("REX_LOG_LEVEL")


def test_parse_empty_file(tmp_path):
    """Test parsing an empty .env.example file."""
    empty_file = tmp_path / ".env.example"
    empty_file.write_text("")

    schema = parse_env_example(empty_file)
    assert len(schema.sections) == 0
    assert len(schema.get_all_variables()) == 0


def test_parse_nonexistent_file(tmp_path):
    """Test parsing a non-existent file raises error."""
    nonexistent = tmp_path / "nonexistent.env"

    with pytest.raises(FileNotFoundError):
        parse_env_example(nonexistent)
