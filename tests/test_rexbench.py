from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rex.rexbench import BenchmarkSample, build_report


def test_build_report_groups_p50_p95_without_private_payloads() -> None:
    samples = [
        BenchmarkSample(
            request_class="typed_chat",
            warm_state="warm",
            evidence_class="deterministic_mock",
            stages_ms={"routing": 10.0, "llm": 30.0, "completion": 2.0, "total": 42.0},
        ),
        BenchmarkSample(
            request_class="typed_chat",
            warm_state="warm",
            evidence_class="deterministic_mock",
            stages_ms={"routing": 12.0, "llm": 34.0, "completion": 3.0, "total": 49.0},
        ),
    ]

    report = build_report(samples, profile="baseline")
    bucket = report["results"]["typed_chat"]["warm"]
    assert bucket["sample_count"] == 2
    assert bucket["stages_ms"]["llm"]["p50"] == pytest.approx(32.0)
    assert bucket["stages_ms"]["llm"]["p95"] == pytest.approx(33.8)
    assert bucket["evidence_class"] == "deterministic_mock"
    encoded = json.dumps(report).lower()
    for forbidden in ("prompt", "transcript", "memory_content", "credential", "user_id"):
        assert forbidden not in encoded


def test_baseline_cli_emits_all_request_classes_and_safe_evidence(tmp_path) -> None:
    output = tmp_path / "baseline.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rexbench.py",
            "--profile",
            "baseline",
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
    assert set(report["results"]) == {
        "typed_chat",
        "voice",
        "read_only_tool",
        "mutating_tool",
        "unavailable_capability",
    }
    for request_class in report["results"].values():
        assert set(request_class) == {"cold", "warm"}
        for bucket in request_class.values():
            assert bucket["evidence_class"] == "deterministic_mock"
            assert "total" in bucket["stages_ms"]
    for warm_state in ("cold", "warm"):
        typed_stages = report["results"]["typed_chat"][warm_state]["stages_ms"]
        assert "first_token" in typed_stages
    encoded = output.read_text(encoding="utf-8").lower()
    for forbidden in ("prompt", "transcript", "memory_content", "credential", "user_id"):
        assert forbidden not in encoded


def test_checked_in_baseline_covers_required_stages_and_privacy() -> None:
    baseline = Path("docs/performance/rexbench-baseline.json")
    report = json.loads(baseline.read_text(encoding="utf-8"))
    required = {
        "typed_chat": {"routing", "first_token", "llm", "completion", "total"},
        "voice": {"capture", "stt", "llm", "first_audio", "completion", "total"},
        "read_only_tool": {"routing", "tool", "llm", "completion", "total"},
        "mutating_tool": {"routing", "tool", "llm", "completion", "total"},
        "unavailable_capability": {"routing", "llm", "completion", "total"},
    }

    assert set(report["results"]) == set(required)
    for request_class, required_stages in required.items():
        assert set(report["results"][request_class]) == {"cold", "warm"}
        for bucket in report["results"][request_class].values():
            assert bucket["evidence_class"] == "deterministic_mock"
            stages = bucket["stages_ms"]
            assert required_stages <= set(stages)
            for stage in required_stages:
                assert set(stages[stage]) == {"p50", "p95"}

    encoded = baseline.read_text(encoding="utf-8").lower()
    for forbidden in ("prompt", "transcript", "memory_content", "credential", "user_id"):
        assert forbidden not in encoded
