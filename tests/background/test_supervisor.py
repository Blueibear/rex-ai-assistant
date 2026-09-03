"""Behavior tests for the persistent Rex background supervisor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rex.background.lock import AlreadyRunningError, SingleInstanceLock
from rex.background.paths import BackgroundPaths
from rex.background.supervisor import ComponentSpec, RuntimeSupervisor
from rex.background.types import HealthState


@dataclass
class _FakeChild:
    name: str
    pid: int
    events: list[str]
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.name}")
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.events.append(f"wait:{self.name}")
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.events.append(f"kill:{self.name}")
        self.returncode = -9


class _FakeProcessFactory:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.children: dict[str, list[_FakeChild]] = {"core": [], "voice_agent": []}
        self._next_pid = 1000

    def __call__(self, spec: ComponentSpec) -> _FakeChild:
        self.events.append(f"start:{spec.name}")
        child = _FakeChild(spec.name, self._next_pid, self.events)
        self._next_pid += 1
        self.children[spec.name].append(child)
        return child


def _spec(name: str, *, max_restarts: int = 3) -> ComponentSpec:
    return ComponentSpec(
        name=name,
        argv=("python", "-m", f"fake.{name}"),
        required=True,
        max_restarts=max_restarts,
        restart_window_seconds=60.0,
    )


def _supervisor(
    tmp_path: Path,
    factory: _FakeProcessFactory,
    *,
    core_restarts: int = 3,
    voice_restarts: int = 3,
) -> RuntimeSupervisor:
    return RuntimeSupervisor(
        BackgroundPaths.from_runtime_root(tmp_path),
        _spec("core", max_restarts=core_restarts),
        _spec("voice_agent", max_restarts=voice_restarts),
        poll_interval=0.01,
        process_factory=factory,
        clock=lambda: 100.0,
    )


def _write_core_endpoint(paths: BackgroundPaths, pid: int) -> None:
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.core_endpoint_file.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": 49152,
                "token": "t" * 32,
                "pid": pid,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_duplicate_supervisor_start_is_rejected(tmp_path: Path) -> None:
    paths = BackgroundPaths.from_runtime_root(tmp_path)
    factory = _FakeProcessFactory()
    supervisor = RuntimeSupervisor(
        paths,
        _spec("core"),
        _spec("voice_agent"),
        process_factory=factory,
        clock=lambda: 100.0,
    )

    with SingleInstanceLock(paths.supervisor_lock):
        with pytest.raises(AlreadyRunningError):
            supervisor.start()


def test_core_starts_before_voice_and_waits_for_endpoint(tmp_path: Path) -> None:
    factory = _FakeProcessFactory()
    supervisor = _supervisor(tmp_path, factory)

    supervisor.start()
    assert factory.events == ["start:core"]

    _write_core_endpoint(supervisor.paths, factory.children["core"][0].pid)
    supervisor.tick()

    assert factory.events[:2] == ["start:core", "start:voice_agent"]
    assert supervisor.health().core.state is HealthState.READY
    assert supervisor.health().voice_agent.state is HealthState.READY
    supervisor.stop()


def test_voice_crash_restarts_without_restarting_core(tmp_path: Path) -> None:
    factory = _FakeProcessFactory()
    supervisor = _supervisor(tmp_path, factory)
    supervisor.start()
    _write_core_endpoint(supervisor.paths, factory.children["core"][0].pid)
    supervisor.tick()

    factory.children["voice_agent"][0].returncode = 7
    supervisor.tick()

    assert len(factory.children["core"]) == 1
    assert len(factory.children["voice_agent"]) == 2
    assert supervisor.health().core.state is HealthState.READY
    supervisor.stop()


def test_repeated_voice_crash_becomes_failed_after_bounded_restarts(tmp_path: Path) -> None:
    factory = _FakeProcessFactory()
    supervisor = _supervisor(tmp_path, factory, voice_restarts=2)
    supervisor.start()
    _write_core_endpoint(supervisor.paths, factory.children["core"][0].pid)
    supervisor.tick()

    for index in range(3):
        factory.children["voice_agent"][index].returncode = 1
        supervisor.tick()

    assert len(factory.children["core"]) == 1
    assert len(factory.children["voice_agent"]) == 3
    health = supervisor.health()
    assert health.core.state is HealthState.READY
    assert health.voice_agent.state is HealthState.FAILED
    assert health.voice_agent.detail_code == "restart_limit_exceeded"
    supervisor.stop()


def test_core_crash_degrades_voice_and_restarts_core_before_voice(tmp_path: Path) -> None:
    factory = _FakeProcessFactory()
    supervisor = _supervisor(tmp_path, factory)
    supervisor.start()
    _write_core_endpoint(supervisor.paths, factory.children["core"][0].pid)
    supervisor.tick()

    first_voice = factory.children["voice_agent"][0]
    factory.children["core"][0].returncode = 9
    supervisor.tick()

    health = supervisor.health()
    assert len(factory.children["core"]) == 2
    assert len(factory.children["voice_agent"]) == 1
    assert first_voice.returncode == 0
    assert health.core.state is HealthState.STARTING
    assert health.voice_agent.state is HealthState.DEGRADED
    assert health.voice_agent.detail_code == "core_unavailable"

    _write_core_endpoint(supervisor.paths, factory.children["core"][1].pid)
    supervisor.tick()
    assert len(factory.children["voice_agent"]) == 2
    assert factory.events.index("start:core", 2) < factory.events.index("start:voice_agent", 3)
    supervisor.stop()


def test_stop_terminates_voice_before_core(tmp_path: Path) -> None:
    factory = _FakeProcessFactory()
    supervisor = _supervisor(tmp_path, factory)
    supervisor.start()
    _write_core_endpoint(supervisor.paths, factory.children["core"][0].pid)
    supervisor.tick()

    supervisor.stop()
    voice_index = factory.events.index("terminate:voice_agent")
    core_index = factory.events.index("terminate:core")
    assert voice_index < core_index
    assert supervisor.health().voice_agent.state is HealthState.STOPPED
    assert supervisor.health().core.state is HealthState.STOPPED


def test_aggregate_health_file_is_content_free(tmp_path: Path) -> None:
    factory = _FakeProcessFactory()
    supervisor = _supervisor(tmp_path, factory)
    supervisor.start()
    _write_core_endpoint(supervisor.paths, factory.children["core"][0].pid)
    supervisor.tick()

    payload = json.loads(supervisor.paths.health_file.read_text(encoding="utf-8"))
    assert set(payload) == {"core", "voice_agent", "supervisor_pid", "observed_at"}
    assert set(payload["core"]) == {
        "component",
        "state",
        "detail_code",
        "observed_at",
        "pid",
    }
    assert set(payload["voice_agent"]) == set(payload["core"])
    serialized = json.dumps(payload).lower()
    for forbidden in (
        "transcript",
        "prompt",
        "user_id",
        "credential",
        "tool_result",
        "microphone_audio",
    ):
        assert forbidden not in serialized
    supervisor.stop()
