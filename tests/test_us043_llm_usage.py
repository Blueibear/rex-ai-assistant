"""Tests for US-043: Ollama cloud usage tracking."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# rex.llm_usage unit tests
# ---------------------------------------------------------------------------


def test_record_usage_creates_file(tmp_path):
    """record_usage writes a JSON line to the usage file."""
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        from rex.llm_usage import record_usage

        record_usage(model="llama3", prompt_tokens=10, completion_tokens=20)

    assert usage_file.exists()
    line = usage_file.read_text().strip()
    record = json.loads(line)
    assert record["model"] == "llama3"
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 20
    assert "timestamp" in record


def test_record_usage_appends(tmp_path):
    """record_usage appends multiple records."""
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        from rex.llm_usage import record_usage

        record_usage(model="llama3", prompt_tokens=5, completion_tokens=10)
        record_usage(model="mistral", prompt_tokens=3, completion_tokens=7)

    lines = [line for line in usage_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["model"] == "llama3"
    assert json.loads(lines[1])["model"] == "mistral"


def test_summarise_empty(tmp_path):
    """summarise returns zeros when no records exist."""
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        from rex.llm_usage import summarise

        result = summarise()

    assert result["total_requests"] == 0
    assert result["total_tokens"] == 0
    assert result["by_model"] == {}


def test_summarise_aggregates(tmp_path):
    """summarise correctly aggregates multiple records."""
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        from rex.llm_usage import record_usage, summarise

        record_usage(model="llama3", prompt_tokens=10, completion_tokens=20)
        record_usage(model="llama3", prompt_tokens=5, completion_tokens=15)
        record_usage(model="mistral", prompt_tokens=8, completion_tokens=12)

        result = summarise()

    assert result["total_requests"] == 3
    assert result["total_tokens"] == 10 + 20 + 5 + 15 + 8 + 12

    llama = result["by_model"]["llama3"]
    assert llama["requests"] == 2
    assert llama["prompt_tokens"] == 15
    assert llama["completion_tokens"] == 35
    assert llama["total_tokens"] == 50

    mistral = result["by_model"]["mistral"]
    assert mistral["requests"] == 1
    assert mistral["total_tokens"] == 20


def test_record_usage_rotation(tmp_path):
    """Large files are rotated before a new record is written."""
    usage_file = tmp_path / "llm_usage.json"
    # Write a fake "large" file (simulate >10 MB threshold via monkeypatching)
    usage_file.write_text("{}\n")

    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        with patch("rex.llm_usage._MAX_SIZE_BYTES", 0):
            from rex.llm_usage import record_usage

            record_usage(model="llama3", prompt_tokens=1, completion_tokens=1)

    rotated = usage_file.with_suffix(".json.1")
    assert rotated.exists(), "Rotated file should exist"
    assert usage_file.exists(), "New usage file should exist after rotation"


# ---------------------------------------------------------------------------
# OllamaStrategy.generate records usage
# ---------------------------------------------------------------------------


def test_ollama_generate_records_usage(tmp_path):
    """OllamaStrategy.generate calls record_usage when token counts are present."""
    usage_file = tmp_path / "llm_usage.json"

    # Build a minimal fake ollama response dict
    fake_response = {
        "message": {"content": "Hello!"},
        "prompt_eval_count": 7,
        "eval_count": 14,
    }

    # Patch the ollama module
    mock_ollama = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.return_value = fake_response
    mock_ollama.Client.return_value = fake_client

    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        # Force OLLAMA_AVAILABLE to be True
        import rex.llm_client as llm_mod

        original = llm_mod.OLLAMA_AVAILABLE
        llm_mod.OLLAMA_AVAILABLE = True
        try:
            from rex.llm_client import GenerationConfig, OllamaStrategy

            strategy = OllamaStrategy.__new__(OllamaStrategy)
            strategy.model_name = "llama3"
            strategy.base_url = "http://localhost:11434"
            strategy.use_cloud = False
            strategy.api_key = None
            strategy._ollama = mock_ollama
            strategy._client_cls = mock_ollama.Client
            strategy._client = fake_client
            strategy._retry_config = __import__("rex.retry", fromlist=["RetryConfig"]).RetryConfig()

            gen_config = GenerationConfig(
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                seed=42,
            )

            with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
                # Need to reload llm_usage with new env
                import importlib

                import rex.llm_usage as usage_mod

                importlib.reload(usage_mod)

                result = strategy.generate("Hi", gen_config)

        finally:
            llm_mod.OLLAMA_AVAILABLE = original

    assert result == "Hello!"
    assert usage_file.exists()
    record = json.loads(usage_file.read_text().strip())
    assert record["model"] == "llama3"
    assert record["prompt_tokens"] == 7
    assert record["completion_tokens"] == 14


# ---------------------------------------------------------------------------
# CLI cmd_usage — import rex.cli with version guard bypassed
# ---------------------------------------------------------------------------


def _import_cmd_usage():
    """Import cmd_usage, bypassing the Python-version guard in rex.cli."""
    import sys
    from unittest.mock import patch as _patch

    with _patch("python_compat.is_supported_python", return_value=True):
        # Remove cached module so the guarded import runs cleanly
        sys.modules.pop("rex.cli", None)
        import rex.cli  # noqa: PLC0415

        sys.modules["rex.cli"] = rex.cli
        return rex.cli.cmd_usage


def test_cmd_usage_no_records(tmp_path, capsys):
    """cmd_usage prints 'No LLM usage recorded yet.' when log is empty."""
    cmd_usage = _import_cmd_usage()
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        import importlib

        import rex.llm_usage as usage_mod

        importlib.reload(usage_mod)

        import argparse

        args = argparse.Namespace()
        rc = cmd_usage(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "No LLM usage recorded yet" in captured.out


def test_cmd_usage_with_records(tmp_path, capsys):
    """cmd_usage prints summary table when records exist."""
    cmd_usage = _import_cmd_usage()
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        import importlib

        import rex.llm_usage as usage_mod

        importlib.reload(usage_mod)

        usage_mod.record_usage(model="llama3", prompt_tokens=10, completion_tokens=20)
        usage_mod.record_usage(model="llama3", prompt_tokens=5, completion_tokens=10)

        import argparse

        args = argparse.Namespace()
        rc = cmd_usage(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "llama3" in captured.out
    assert "Total requests" in captured.out
