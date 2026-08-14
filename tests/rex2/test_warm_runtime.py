from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rex.runtime.warm import (
    WarmComponentSpec,
    WarmRuntimeManager,
    WarmState,
)


def test_warm_component_loads_lazily_and_reuses_one_instance() -> None:
    now = [0.0]
    loads: list[object] = []
    manager = WarmRuntimeManager(max_cost_mb=256, clock=lambda: now[0])

    manager.register(
        WarmComponentSpec(
            name="executive",
            loader=lambda: loads.append(object()) or loads[-1],
            estimated_cost_mb=128,
            idle_timeout_s=10,
        )
    )

    assert loads == []
    first = manager.get("executive")
    now[0] = 2.0
    second = manager.get("executive")

    assert first is second
    assert len(loads) == 1
    assert manager.snapshot()["components"][0]["load_count"] == 1


def test_idle_and_budget_eviction_are_bounded_and_lru() -> None:
    now = [0.0]
    unloaded: list[str] = []
    manager = WarmRuntimeManager(max_cost_mb=220, clock=lambda: now[0])

    def spec(name: str, cost: float) -> WarmComponentSpec:
        return WarmComponentSpec(
            name=name,
            loader=lambda: {"name": name},
            unloader=lambda _value: unloaded.append(name),
            estimated_cost_mb=cost,
            idle_timeout_s=10,
        )

    manager.register(spec("executive", 120))
    manager.register(spec("stt", 100))
    manager.register(spec("tts", 100))
    manager.get("executive")
    now[0] = 1.0
    manager.get("stt")
    now[0] = 2.0
    manager.get("executive")
    now[0] = 3.0
    manager.get("tts")

    assert "stt" in unloaded
    assert manager.status("stt").state is WarmState.EVICTED
    assert manager.snapshot()["estimated_cost_mb"] <= 220

    now[0] = 20.0
    assert set(manager.evict_idle()) == {"executive", "tts"}
    assert manager.snapshot()["estimated_cost_mb"] == 0


def test_optional_component_falls_back_lazily_without_leaking_loader_error() -> None:
    manager = WarmRuntimeManager(max_cost_mb=256)
    private_error = "credential-fragment-private"
    manager.register(
        WarmComponentSpec(
            name="stt",
            loader=lambda: (_ for _ in ()).throw(ModuleNotFoundError(private_error)),
            fallback=lambda: "text-only",
            estimated_cost_mb=80,
        )
    )

    assert manager.status("stt").state is WarmState.COLD
    assert manager.get("stt") == "text-only"

    status = manager.status("stt")
    snapshot = manager.snapshot()
    assert status.state is WarmState.DEGRADED
    assert status.error_type == "ModuleNotFoundError"
    assert private_error not in repr(snapshot)


def test_concurrent_get_loads_component_once() -> None:
    loads = 0
    load_lock = threading.Lock()
    manager = WarmRuntimeManager(max_cost_mb=256)

    def loader() -> object:
        nonlocal loads
        with load_lock:
            loads += 1
        time.sleep(0.02)
        return object()

    manager.register(WarmComponentSpec(name="index", loader=loader, estimated_cost_mb=32))
    values: list[object] = []
    threads = [
        threading.Thread(target=lambda: values.append(manager.get("index"))) for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert loads == 1
    assert len({id(value) for value in values}) == 1


def test_active_lease_prevents_false_eviction_until_released() -> None:
    manager = WarmRuntimeManager(max_cost_mb=100)
    manager.register(WarmComponentSpec(name="stt", loader=object, estimated_cost_mb=100))
    manager.register(WarmComponentSpec(name="tts", loader=object, estimated_cost_mb=100))

    lease = manager.acquire("stt")
    transient_tts = manager.get("tts")
    assert transient_tts is not None
    assert manager.status("stt").state is WarmState.WARM
    assert manager.status("tts").state is not WarmState.WARM

    lease.release()
    manager.get("tts")
    assert manager.status("stt").state is WarmState.EVICTED


def test_transformers_runtime_is_shared_across_strategy_instances(monkeypatch) -> None:
    import rex.llm_client as llm
    from rex.runtime.warm import reset_global_warm_runtime

    reset_global_warm_runtime()
    loads = {"tokenizer": 0, "model": 0, "pipeline": 0}

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 0

    class FakePipeline:
        def __call__(self, prompt, **_kwargs):
            return [{"generated_text": prompt + " ok"}]

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda _name: loads.__setitem__("tokenizer", loads["tokenizer"] + 1)
            or FakeTokenizer()
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda _name: loads.__setitem__("model", loads["model"] + 1) or object()
        ),
        pipeline=lambda *_args, **_kwargs: loads.__setitem__("pipeline", loads["pipeline"] + 1)
        or FakePipeline(),
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(llm, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(llm, "TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        llm,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_transformers,
    )

    first = llm.TransformersStrategy("shared-model")
    second = llm.TransformersStrategy("shared-model")
    assert first.pipeline is second.pipeline
    assert loads == {"tokenizer": 1, "model": 1, "pipeline": 1}
    reset_global_warm_runtime()


def test_whisper_model_is_shared_without_sharing_stt_request_state(monkeypatch) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import reset_global_warm_runtime
    from rex.voice.stt import SpeechToText

    reset_global_warm_runtime()
    loads: list[tuple[str, str]] = []
    shared_model = object()
    whisper = SimpleNamespace(
        load_model=lambda name, device: loads.append((name, device)) or shared_model
    )
    monkeypatch.setattr(voice_loop, "_lazy_import_whisper", lambda: whisper)
    monkeypatch.setattr(voice_loop.settings, "whisper_initial_prompt", None, raising=False)

    first = SpeechToText("base", "cpu", language="en")
    second = SpeechToText("base", "cpu", language="es")

    assert loads == [("base", "cpu")]
    assert first._resolve_model() is shared_model
    assert second._resolve_model() is shared_model
    assert first._language == "en"
    assert second._language == "es"
    reset_global_warm_runtime()


def test_xtts_engine_is_shared_across_tts_wrappers(monkeypatch) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import reset_global_warm_runtime
    from rex.voice.tts import TextToSpeech

    reset_global_warm_runtime()
    created: list[object] = []

    class FakeTTS:
        def __init__(self, **_kwargs: object) -> None:
            created.append(self)

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(voice_loop, "_lazy_import_tts", lambda: FakeTTS)
    monkeypatch.setattr(voice_loop, "import_module", lambda _name: fake_torch)
    monkeypatch.setattr(voice_loop.settings, "tts_provider", "xtts", raising=False)
    monkeypatch.setattr(voice_loop.settings, "tts_voice", None, raising=False)
    monkeypatch.setattr(voice_loop.settings, "tts_speed", 1.0, raising=False)

    first = TextToSpeech(language="en")
    second = TextToSpeech(language="en")

    assert len(created) == 1
    assert first._tts is created[0]
    assert second._tts is created[0]
    reset_global_warm_runtime()


def test_knowledge_index_reuses_process_singleton(tmp_path) -> None:
    from rex.knowledge_base import KnowledgeBase, get_knowledge_base, set_knowledge_base

    kb = KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
    )
    set_knowledge_base(kb)

    assert get_knowledge_base() is kb
    assert get_knowledge_base() is kb
    set_knowledge_base(None)


def test_doctor_reports_warm_components_without_private_identifiers() -> None:
    from rex.doctor import Status, check_warm_runtime
    from rex.runtime.warm import (
        WarmComponentSpec,
        get_global_warm_runtime,
        reset_global_warm_runtime,
        warm_component_key,
    )

    reset_global_warm_runtime()
    private_identifier = "C:/private/models/secret-model"
    manager = get_global_warm_runtime()
    name = warm_component_key("llm", private_identifier)
    manager.register(WarmComponentSpec(name=name, loader=object, estimated_cost_mb=128))
    manager.get(name)

    result = check_warm_runtime()

    assert result.status in {Status.OK, Status.INFO}
    assert "128" in result.message or "128" in result.details
    assert "llm:" in result.details
    assert private_identifier not in result.message
    assert private_identifier not in result.details
    reset_global_warm_runtime()


def test_language_model_initializes_warm_budget_from_app_config(monkeypatch) -> None:
    import rex.llm_client as llm_client
    from rex.config import AppConfig
    from rex.llm_client import LanguageModel
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime

    reset_global_warm_runtime()
    monkeypatch.setattr(llm_client, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(llm_client, "TRANSFORMERS_AVAILABLE", False)
    cfg = AppConfig(
        llm_provider="transformers",
        llm_model="sshleifer/tiny-gpt2",
        warm_runtime_max_cost_mb=3210.0,
        warm_runtime_idle_timeout_s=123.0,
    )

    LanguageModel(config=cfg)

    assert get_global_warm_runtime().max_cost_mb == 3210.0
    reset_global_warm_runtime()


def test_normal_access_sweeps_expired_idle_components() -> None:
    now = [0.0]
    unloaded: list[str] = []
    manager = WarmRuntimeManager(max_cost_mb=500, clock=lambda: now[0])
    for name in ("executive", "stt"):
        manager.register(
            WarmComponentSpec(
                name=name,
                loader=object,
                unloader=lambda _value, n=name: unloaded.append(n),
                estimated_cost_mb=100,
                idle_timeout_s=10,
            )
        )

    manager.get("executive")
    now[0] = 20.0
    manager.get("stt")

    assert "executive" in unloaded
    assert manager.status("executive").state is WarmState.EVICTED
    assert manager.status("stt").state is WarmState.WARM


def test_acquire_expired_component_does_not_deadlock() -> None:
    now = [0.0]
    loads: list[object] = []
    manager = WarmRuntimeManager(max_cost_mb=64, clock=lambda: now[0])
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=lambda: loads.append(object()) or loads[-1],
            estimated_cost_mb=32,
            idle_timeout_s=10,
        )
    )
    manager.get("engine")
    now[0] = 20.0
    completed = threading.Event()

    def use_expired_engine() -> None:
        with manager.acquire("engine"):
            completed.set()

    worker = threading.Thread(target=use_expired_engine, daemon=True)
    worker.start()
    assert completed.wait(1.0), "acquiring an expired component deadlocked"
    worker.join(1.0)

    assert not worker.is_alive()
    assert len(loads) == 2
    assert manager.status("engine").state is WarmState.WARM


def test_idle_eviction_rechecks_fresh_usage_before_evicting(monkeypatch) -> None:
    now = [0.0]
    manager = WarmRuntimeManager(max_cost_mb=64, clock=lambda: now[0])
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=object,
            estimated_cost_mb=32,
            idle_timeout_s=10,
        )
    )
    manager.get("engine")
    now[0] = 20.0
    selected = threading.Event()
    proceed = threading.Event()
    original_evict = manager._evict

    def paused_evict(name: str, *, idle_check_at: float | None = None) -> bool:
        selected.set()
        assert proceed.wait(1.0)
        return original_evict(name, idle_check_at=idle_check_at)

    monkeypatch.setattr(manager, "_evict", paused_evict)
    evicted: list[str] = []
    worker = threading.Thread(target=lambda: evicted.extend(manager.evict_idle()), daemon=True)
    worker.start()
    assert selected.wait(1.0)

    with manager._lock:
        manager._entries["engine"].last_used = now[0]
    proceed.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert evicted == []
    assert manager.status("engine").state is WarmState.WARM


def test_transformers_zero_budget_skips_constructor_load_until_inference(monkeypatch) -> None:
    import rex.llm_client as llm
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime

    reset_global_warm_runtime()
    manager = get_global_warm_runtime()
    manager.set_budget(0)
    loads = {"tokenizer": 0, "model": 0, "pipeline": 0}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class FakePipeline:
        def __call__(self, prompt: str, **_kwargs: object):
            return [{"generated_text": prompt + " ok"}]

    def load_tokenizer(_name: str) -> FakeTokenizer:
        loads["tokenizer"] += 1
        return FakeTokenizer()

    def load_model(_name: str) -> object:
        loads["model"] += 1
        return object()

    def make_pipeline(*_args: object, **_kwargs: object) -> FakePipeline:
        loads["pipeline"] += 1
        return FakePipeline()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        pipeline=make_pipeline,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda _seed: None),
        manual_seed=lambda _seed: None,
    )
    monkeypatch.setattr(llm, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(llm, "TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        llm,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_transformers,
    )

    strategy = llm.TransformersStrategy("cold-on-demand")
    assert loads == {"tokenizer": 0, "model": 0, "pipeline": 0}

    result = strategy.generate(
        "hello",
        llm.GenerationConfig(
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            seed=1,
        ),
    )
    assert result == "ok"
    assert loads == {"tokenizer": 1, "model": 1, "pipeline": 1}
    reset_global_warm_runtime()


def test_whisper_zero_budget_skips_constructor_load_until_acquire(monkeypatch) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime
    from rex.voice.stt import SpeechToText

    reset_global_warm_runtime()
    get_global_warm_runtime().set_budget(0)
    loads: list[str] = []
    whisper = SimpleNamespace(
        load_model=lambda _name, device=None: loads.append(str(device)) or object()
    )
    monkeypatch.setattr(voice_loop, "_lazy_import_whisper", lambda: whisper)

    stt = SpeechToText("base", "cpu", language="en")
    assert loads == []

    with stt._warm_manager.acquire(stt._warm_component_name):
        pass
    assert loads == ["cpu"]
    reset_global_warm_runtime()


def test_xtts_zero_budget_skips_constructor_load_until_acquire(monkeypatch) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime
    from rex.voice.tts import TextToSpeech

    reset_global_warm_runtime()
    get_global_warm_runtime().set_budget(0)
    loads: list[str] = []

    class FakeTTS:
        def __init__(self, **_kwargs: object) -> None:
            loads.append("loaded")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(voice_loop, "_lazy_import_tts", lambda: FakeTTS)
    monkeypatch.setattr(voice_loop, "import_module", lambda _name: fake_torch)
    monkeypatch.setattr(voice_loop.settings, "tts_provider", "xtts", raising=False)

    tts = TextToSpeech(language="en")
    assert loads == []

    with tts._warm_manager.acquire(tts._warm_component_name):
        pass
    assert loads == ["loaded"]
    reset_global_warm_runtime()


def test_transformers_inference_holds_warm_lease_against_budget_eviction(monkeypatch) -> None:
    import rex.llm_client as llm
    from rex.runtime.warm import WarmComponentSpec, reset_global_warm_runtime

    reset_global_warm_runtime()
    holder: dict[str, object] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class FakePipeline:
        def __call__(self, prompt: str, **_kwargs: object):
            strategy = holder["strategy"]
            manager = strategy._warm_manager
            manager.register(
                WarmComponentSpec(name="competitor", loader=object, estimated_cost_mb=5000)
            )
            assert manager.get("competitor") is not None
            assert manager.status("competitor").state is not WarmState.WARM
            assert manager.status(strategy._warm_component_name).state is WarmState.WARM
            return [{"generated_text": prompt + " ok"}]

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _name: FakeTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda _name: object()),
        pipeline=lambda *_args, **_kwargs: FakePipeline(),
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda _seed: None,
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        manual_seed=lambda _seed: None,
    )
    monkeypatch.setattr(llm, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(llm, "TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        llm,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_transformers,
    )

    strategy = llm.TransformersStrategy("lease-model")
    holder["strategy"] = strategy
    result = strategy.generate(
        "hello",
        llm.GenerationConfig(
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            seed=1,
        ),
    )

    assert result == "ok"
    reset_global_warm_runtime()


def test_whisper_transcription_holds_warm_lease_against_budget_eviction(monkeypatch) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import WarmComponentSpec, reset_global_warm_runtime
    from rex.voice.stt import SpeechToText

    reset_global_warm_runtime()
    holder: dict[str, SpeechToText] = {}

    class FakeModel:
        def transcribe(self, _audio: object, **_kwargs: object) -> dict[str, str]:
            manager = holder["stt"]._warm_manager
            manager.register(
                WarmComponentSpec(name="stt-competitor", loader=object, estimated_cost_mb=5500)
            )
            assert manager.get("stt-competitor") is not None
            assert manager.status("stt-competitor").state is not WarmState.WARM
            assert manager.status(holder["stt"]._warm_component_name).state is WarmState.WARM
            return {"text": "hello"}

    whisper = SimpleNamespace(load_model=lambda _name, device=None: FakeModel())
    monkeypatch.setattr(voice_loop, "_lazy_import_whisper", lambda: whisper)
    stt = SpeechToText("base", "cpu", language="en")
    holder["stt"] = stt

    assert asyncio.run(stt.transcribe(audio=[], sample_rate=16000)) == "hello"
    reset_global_warm_runtime()


def test_xtts_synthesis_holds_warm_lease_against_budget_eviction(monkeypatch, tmp_path) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import WarmComponentSpec, reset_global_warm_runtime
    from rex.voice.tts import TextToSpeech

    reset_global_warm_runtime()
    holder: dict[str, TextToSpeech] = {}

    class FakeTTS:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def tts_to_file(self, **kwargs: object) -> None:
            tts = holder["tts"]
            manager = tts._warm_manager
            manager.register(
                WarmComponentSpec(name="tts-competitor", loader=object, estimated_cost_mb=5000)
            )
            assert manager.get("tts-competitor") is not None
            assert manager.status("tts-competitor").state is not WarmState.WARM
            assert manager.status(tts._warm_component_name).state is WarmState.WARM
            Path(str(kwargs["file_path"])).write_bytes(b"RIFF")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(voice_loop, "_lazy_import_tts", lambda: FakeTTS)
    monkeypatch.setattr(voice_loop, "_lazy_import_soundfile", lambda: object())
    monkeypatch.setattr(voice_loop, "import_module", lambda _name: fake_torch)
    monkeypatch.setattr(voice_loop.settings, "tts_provider", "xtts", raising=False)
    monkeypatch.setattr(voice_loop.settings, "tts_voice", None, raising=False)
    monkeypatch.setattr(voice_loop.settings, "tts_speed", 1.0, raising=False)

    tts = TextToSpeech(language="en")
    holder["tts"] = tts
    monkeypatch.setattr(tts, "_try_smart_speaker", lambda _path: True)

    asyncio.run(tts._synthesize_and_play_chunk("hello", None, object()))

    reset_global_warm_runtime()


def test_late_runtime_config_replaces_unused_default_budget(monkeypatch) -> None:
    import rex.llm_client as llm_client
    from rex.config import AppConfig
    from rex.llm_client import LanguageModel
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime

    reset_global_warm_runtime()
    assert get_global_warm_runtime().max_cost_mb == 6144.0
    monkeypatch.setattr(llm_client, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(llm_client, "TRANSFORMERS_AVAILABLE", False)

    cfg = AppConfig(
        llm_provider="transformers",
        llm_model="sshleifer/tiny-gpt2",
        warm_runtime_max_cost_mb=2048.0,
    )
    LanguageModel(config=cfg)

    assert get_global_warm_runtime().max_cost_mb == 2048.0
    reset_global_warm_runtime()


def test_component_leases_serialize_shared_engine_use() -> None:
    manager = WarmRuntimeManager(max_cost_mb=64)
    manager.register(WarmComponentSpec(name="engine", loader=object, estimated_cost_mb=32))
    active = 0
    peak = 0
    lock = threading.Lock()

    def use_engine() -> None:
        nonlocal active, peak
        with manager.acquire("engine"):
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1

    first = threading.Thread(target=use_engine)
    second = threading.Thread(target=use_engine)
    first.start()
    second.start()
    first.join()
    second.join()

    assert peak == 1


def test_degraded_fallback_is_accounted_and_idle_evictable() -> None:
    now = [0.0]
    unloaded: list[object] = []
    manager = WarmRuntimeManager(max_cost_mb=100, clock=lambda: now[0])
    fallback_value = object()
    manager.register(
        WarmComponentSpec(
            name="fallback",
            loader=lambda: (_ for _ in ()).throw(RuntimeError("missing")),
            fallback=lambda: fallback_value,
            unloader=unloaded.append,
            estimated_cost_mb=80,
            idle_timeout_s=10,
        )
    )

    assert manager.get("fallback") is fallback_value
    assert manager.snapshot()["estimated_cost_mb"] == 80
    assert manager.status("fallback").estimated_cost_mb == 80

    now[0] = 20.0
    assert manager.evict_idle() == ["fallback"]
    assert unloaded == [fallback_value]
    assert manager.snapshot()["estimated_cost_mb"] == 0


def test_unloader_runs_outside_global_bookkeeping_lock() -> None:
    manager = WarmRuntimeManager(max_cost_mb=64)
    worker_completed = threading.Event()

    def unloader(_value: object) -> None:
        def inspect_manager() -> None:
            manager.snapshot()
            worker_completed.set()

        worker = threading.Thread(target=inspect_manager)
        worker.start()
        assert worker_completed.wait(0.5)
        worker.join()

    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=object,
            unloader=unloader,
            estimated_cost_mb=32,
        )
    )
    manager.get("engine")

    assert manager.evict("engine") is True
    assert manager.status("engine").state is WarmState.EVICTED


def test_snapshot_redacts_untrusted_component_names() -> None:
    manager = WarmRuntimeManager(max_cost_mb=64)
    private_name = "/private/models/model-secret"
    manager.register(
        WarmComponentSpec(
            name=private_name,
            loader=object,
            estimated_cost_mb=32,
        )
    )
    manager.get(private_name)

    snapshot = manager.snapshot()
    rendered = repr(snapshot)

    assert private_name not in rendered
    diagnostic_name = snapshot["components"][0]["name"]
    assert diagnostic_name.startswith("component:")
    assert len(diagnostic_name.split(":", 1)[1]) == 12


def test_non_authoritative_runtime_access_does_not_rewrite_global_budget(monkeypatch) -> None:
    import rex.llm_client as llm_client
    from rex.config import AppConfig
    from rex.llm_client import LanguageModel
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime

    reset_global_warm_runtime()
    monkeypatch.setattr(llm_client, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(llm_client, "TRANSFORMERS_AVAILABLE", False)
    cfg = AppConfig(
        llm_provider="transformers",
        llm_model="tiny-model",
        warm_runtime_max_cost_mb=3072.0,
    )
    LanguageModel(config=cfg)
    manager = get_global_warm_runtime()
    assert manager.max_cost_mb == 3072.0

    get_global_warm_runtime(SimpleNamespace(warm_runtime_max_cost_mb=512.0))

    assert manager.max_cost_mb == 3072.0
    reset_global_warm_runtime()


def test_second_language_model_does_not_rewrite_established_global_policy(monkeypatch) -> None:
    import rex.llm_client as llm_client
    from rex.config import AppConfig
    from rex.llm_client import LanguageModel
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime

    reset_global_warm_runtime()
    monkeypatch.setattr(llm_client, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(llm_client, "TRANSFORMERS_AVAILABLE", False)
    first = AppConfig(
        llm_provider="transformers",
        llm_model="tiny-model",
        warm_runtime_max_cost_mb=3000.0,
        warm_runtime_idle_timeout_s=300.0,
    )
    second = AppConfig(
        llm_provider="transformers",
        llm_model="tiny-model",
        warm_runtime_max_cost_mb=500.0,
        warm_runtime_idle_timeout_s=30.0,
    )

    LanguageModel(config=first)
    manager = get_global_warm_runtime()
    LanguageModel(config=second)

    assert manager.max_cost_mb == 3000.0
    assert get_global_warm_runtime() is manager
    reset_global_warm_runtime()


def test_stt_warm_registration_does_not_retain_wrapper(monkeypatch) -> None:
    import gc
    import weakref

    import rex.voice_loop as voice_loop
    from rex.runtime.warm import reset_global_warm_runtime
    from rex.voice.stt import SpeechToText

    reset_global_warm_runtime()
    shared_model = object()
    whisper = SimpleNamespace(load_model=lambda _name, device=None: shared_model)
    monkeypatch.setattr(voice_loop, "_lazy_import_whisper", lambda: whisper)
    monkeypatch.setattr(voice_loop.settings, "whisper_initial_prompt", None, raising=False)

    stt = SpeechToText("base", "cpu", language="en")
    wrapper_ref = weakref.ref(stt)
    del stt
    gc.collect()

    assert wrapper_ref() is None
    reset_global_warm_runtime()


def test_prewarm_skips_load_when_component_cannot_fit_budget() -> None:
    manager = WarmRuntimeManager(max_cost_mb=0)
    loads: list[str] = []
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=lambda: loads.append("loaded") or object(),
            estimated_cost_mb=32,
        )
    )

    assert manager.warm("engine") is False
    assert loads == []
    assert manager.status("engine").state is WarmState.COLD


def test_zero_budget_uses_uncached_cold_load_instead_of_disabling_component() -> None:
    loads: list[object] = []
    manager = WarmRuntimeManager(max_cost_mb=0)
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=lambda: loads.append(object()) or loads[-1],
            estimated_cost_mb=64,
        )
    )

    first = manager.get("engine")
    second = manager.get("engine")

    assert first is not second
    assert len(loads) == 2
    assert manager.snapshot()["estimated_cost_mb"] == 0
    assert manager.status("engine").state is not WarmState.WARM


def test_zero_budget_acquire_returns_transient_lease_and_releases_it() -> None:
    unloaded: list[object] = []
    manager = WarmRuntimeManager(max_cost_mb=0)
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=object,
            unloader=unloaded.append,
            estimated_cost_mb=64,
        )
    )
    leases: list[object] = []

    worker = threading.Thread(target=lambda: leases.append(manager.acquire("engine")), daemon=True)
    worker.start()
    worker.join(0.5)

    assert not worker.is_alive()
    lease = leases[0]
    value = lease.value
    lease.release()
    assert unloaded == [value]
    assert manager.snapshot()["estimated_cost_mb"] == 0


def test_xtts_cancelled_coroutine_keeps_worker_lease_until_synthesis_finishes(
    monkeypatch,
) -> None:
    import rex.voice_loop as voice_loop
    from rex.runtime.warm import reset_global_warm_runtime
    from rex.voice.tts import TextToSpeech

    reset_global_warm_runtime()
    started = threading.Event()
    release_worker = threading.Event()
    finished = threading.Event()
    generated_paths: list[Path] = []

    class FakeTTS:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def tts_to_file(self, **kwargs: object) -> None:
            generated_paths.append(Path(str(kwargs["file_path"])))
            started.set()
            release_worker.wait(2.0)
            Path(str(kwargs["file_path"])).write_bytes(b"RIFF")
            finished.set()

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(voice_loop, "_lazy_import_tts", lambda: FakeTTS)
    monkeypatch.setattr(voice_loop, "_lazy_import_soundfile", lambda: object())
    monkeypatch.setattr(voice_loop, "import_module", lambda _name: fake_torch)
    monkeypatch.setattr(voice_loop.settings, "tts_provider", "xtts", raising=False)
    monkeypatch.setattr(voice_loop.settings, "tts_voice", None, raising=False)
    monkeypatch.setattr(voice_loop.settings, "tts_speed", 1.0, raising=False)

    tts = TextToSpeech(language="en")
    monkeypatch.setattr(tts, "_try_smart_speaker", lambda _path: True)

    async def scenario() -> None:
        task = asyncio.create_task(tts._synthesize_and_play_chunk("hello", None, object()))
        assert await asyncio.to_thread(started.wait, 0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        manager = tts._warm_manager
        assert manager.evict(tts._warm_component_name) is False
        release_worker.set()
        assert await asyncio.to_thread(finished.wait, 1.0)
        await asyncio.sleep(0)
        assert generated_paths
        assert not generated_paths[0].exists()

    asyncio.run(scenario())
    reset_global_warm_runtime()


def test_mutable_knowledge_base_stays_warm_but_is_excluded_from_evictable_budget(
    tmp_path,
) -> None:
    from rex.knowledge_base import KnowledgeBase, get_knowledge_base, set_knowledge_base
    from rex.runtime.warm import get_global_warm_runtime, reset_global_warm_runtime

    reset_global_warm_runtime()
    kb = KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
    )
    set_knowledge_base(kb)

    assert get_knowledge_base() is kb
    assert get_knowledge_base() is kb
    snapshot = get_global_warm_runtime().snapshot()
    assert all(not item["name"].startswith("index:") for item in snapshot["components"])

    set_knowledge_base(None)
    reset_global_warm_runtime()


def test_fallback_failure_leaves_consistent_error_state() -> None:
    manager = WarmRuntimeManager(max_cost_mb=64)
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
            fallback=lambda: (_ for _ in ()).throw(ValueError("fallback failed")),
            estimated_cost_mb=32,
        )
    )

    with pytest.raises(ValueError, match="fallback failed"):
        manager.get("engine")

    status = manager.status("engine")
    assert status.state is WarmState.ERROR
    assert status.error_type == "ValueError"
    assert manager.snapshot()["estimated_cost_mb"] == 0


def test_unloader_failure_keeps_entry_resident_and_accounted() -> None:
    manager = WarmRuntimeManager(max_cost_mb=64)
    loaded = object()
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=lambda: loaded,
            unloader=lambda _value: (_ for _ in ()).throw(RuntimeError("unload failed")),
            estimated_cost_mb=32,
        )
    )
    manager.get("engine")

    with pytest.raises(RuntimeError, match="unload failed"):
        manager.evict("engine")

    status = manager.status("engine")
    assert status.state is WarmState.WARM
    assert status.error_type == "RuntimeError"
    assert manager.peek("engine") is loaded
    assert manager.snapshot()["estimated_cost_mb"] == 32


def test_authoritative_reconfigure_updates_existing_component_idle_policy() -> None:
    from rex.runtime.warm import (
        configure_global_warm_runtime,
        default_idle_timeout,
        get_global_warm_runtime,
        reset_global_warm_runtime,
    )

    reset_global_warm_runtime()
    initial = SimpleNamespace(
        warm_runtime_max_cost_mb=128.0,
        warm_runtime_idle_timeout_s=10.0,
    )
    manager = get_global_warm_runtime(initial)
    manager.register(
        WarmComponentSpec(
            name="engine",
            loader=object,
            estimated_cost_mb=32,
            idle_timeout_s=default_idle_timeout(initial),
        )
    )
    updated = SimpleNamespace(
        warm_runtime_max_cost_mb=128.0,
        warm_runtime_idle_timeout_s=20.0,
    )
    configure_global_warm_runtime(updated)

    manager.register_if_absent(
        WarmComponentSpec(
            name="engine",
            loader=object,
            estimated_cost_mb=32,
            idle_timeout_s=default_idle_timeout(updated),
        )
    )
    reset_global_warm_runtime()


def test_uncached_fallback_does_not_claim_retained_budget() -> None:
    manager = WarmRuntimeManager(max_cost_mb=0)
    manager.register(
        WarmComponentSpec(
            name="fallback",
            loader=lambda: (_ for _ in ()).throw(RuntimeError("missing")),
            fallback=object,
            estimated_cost_mb=64,
        )
    )

    assert manager.get("fallback") is not None
    assert manager.status("fallback").state is WarmState.DEGRADED
    assert manager.status("fallback").estimated_cost_mb == 0
    assert manager.snapshot()["estimated_cost_mb"] == 0


def test_slow_loader_does_not_block_unrelated_diagnostics() -> None:
    manager = WarmRuntimeManager(max_cost_mb=128)
    started = threading.Event()
    release_loader = threading.Event()

    def slow_loader() -> object:
        started.set()
        release_loader.wait(1.0)
        return object()

    manager.register(WarmComponentSpec(name="slow", loader=slow_loader, estimated_cost_mb=32))
    worker = threading.Thread(target=lambda: manager.get("slow"), daemon=True)
    worker.start()
    assert started.wait(0.5)

    before = time.monotonic()
    snapshot = manager.snapshot()
    elapsed = time.monotonic() - before

    assert elapsed < 0.25
    assert snapshot["components"][0]["state"] is WarmState.LOADING
    release_loader.set()
    worker.join(1.0)
    assert not worker.is_alive()
