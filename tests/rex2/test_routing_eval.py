"""US-111 deterministic routing evaluation corpus tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

CORPUS = Path("tests/fixtures/rexbench/routing-eval.json")


def _load_rexbench_module():
    path = Path("scripts/rexbench.py").resolve()
    spec = importlib.util.spec_from_file_location("askrex_rexbench_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checked_in_routing_corpus_is_deterministic_and_content_safe():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))

    assert corpus["schema_version"] == 1
    assert corpus["evidence_class"] == "deterministic_local"
    ids = [case["id"] for case in corpus["cases"]]
    assert len(ids) == len(set(ids)) >= 4

    encoded = CORPUS.read_text(encoding="utf-8").lower()
    for forbidden in (
        "prompt",
        "response",
        "transcript",
        "memory_content",
        "credential",
        "api_key",
        "token",
        "user_id",
    ):
        assert forbidden not in encoded


def test_routing_eval_scores_selection_fallback_and_regression_without_network():
    module = _load_rexbench_module()
    report = module.run_routing_eval(2, corpus_path=CORPUS)

    assert report["profile"] == "routing-eval"
    assert report["evidence_class"] == "deterministic_local"
    assert report["live_provider_eval"] is False
    assert report["corpus_version"] == 1
    assert report["total_cases"] == 4
    assert report["passed_cases"] == 4
    assert report["selection_accuracy"] == 1.0
    assert set(report["results"]) == {
        "healthy_primary",
        "rate_limit_fallback",
        "provider_outage_fallback",
        "all_providers_unavailable",
    }

    for case_id, result in report["results"].items():
        assert result["passed"] is True, case_id
        assert result["evidence_class"] == "deterministic_local"
        assert result["iterations"] == 2
        assert result["routing_ms"]["p50"] >= 0
        assert result["routing_ms"]["p95"] >= 0

    rendered = json.dumps(report).lower()
    for forbidden in ("prompt", "transcript", "credential", "api_key", "user_id"):
        assert forbidden not in rendered


def test_routing_eval_cli_defaults_to_no_live_provider_calls(tmp_path):
    output = tmp_path / "routing-eval.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rexbench.py",
            "--profile",
            "routing-eval",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["live_provider_eval"] is False
    assert report["selection_accuracy"] == 1.0
