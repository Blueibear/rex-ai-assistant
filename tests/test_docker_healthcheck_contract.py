from pathlib import Path

from rex.cli import create_parser
from rex.doctor import CheckResult, Status

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_healthcheck_flag_parses() -> None:
    args = create_parser().parse_args(["doctor", "--healthcheck"])
    assert args.healthcheck is True


def test_healthcheck_fails_when_core_runtime_check_is_not_ok(monkeypatch) -> None:
    from rex.doctor import run_healthcheck

    ok = CheckResult("Python Version", Status.OK, "ok")
    broken = CheckResult("Package Installation", Status.WARNING, "not importable")
    monkeypatch.setattr("rex.doctor.check_python_version", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_package_installation", lambda: broken)
    assert run_healthcheck() == 1


def test_healthcheck_passes_when_core_runtime_checks_are_ok(monkeypatch) -> None:
    from rex.doctor import run_healthcheck

    ok = CheckResult("core", Status.OK, "ok")
    monkeypatch.setattr("rex.doctor.check_python_version", lambda: ok)
    monkeypatch.setattr("rex.doctor.check_package_installation", lambda: ok)
    assert run_healthcheck() == 0


def test_docker_healthcheck_uses_real_doctor_liveness_probe() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "CMD python -m rex doctor --healthcheck" in dockerfile
    assert "sys.exit(0)" not in dockerfile
    assert "run_gui.py" not in dockerfile


def test_docker_is_classified_developer_only_in_required_docs() -> None:
    for rel in ("README.md", "docs/docker.md", "SURFACE-CLASSIFICATION.md"):
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        assert "docker" in text
        assert "developer-only" in text


def test_developer_container_doctor_treats_unavailable_voice_stack_as_nonfatal(monkeypatch) -> None:
    from rex.doctor import check_audio_input_device, check_audio_output_device, check_stt_backend

    monkeypatch.setenv("ASKREX_DEVELOPER_CONTAINER", "1")
    for result in (check_audio_input_device(), check_audio_output_device(), check_stt_backend()):
        assert result.status == Status.INFO
        assert "developer-only" in result.message.lower()


def test_dockerfile_does_not_set_ignored_legacy_runtime_env_vars() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ASKREX_DEVELOPER_CONTAINER=1" in dockerfile
    for name in ("REX_WAKEWORD=", "REX_DEVICE=", "REX_WHISPER_DEVICE="):
        assert name not in dockerfile
