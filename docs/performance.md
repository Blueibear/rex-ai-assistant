# AskRex Performance and RexBench

## Purpose

AskRex records latency before optimizing it. Timing evidence must distinguish
framework overhead from live provider, model, network, audio, and device costs.
A synthetic or mocked benchmark is never evidence that real user latency is fixed.

US-075 establishes the baseline instrumentation used by later TurnEngine and
release-readiness work. Final production acceptance belongs to US-118.

## Privacy contract

Latency telemetry may contain only:
- trace/request timing identifiers,
- channel or transport mode,
- provider/model identifiers,
- non-secret settings/mode identifiers,
- stage durations and success/error outcome.

It must not contain prompts, transcripts, memory contents, user IDs, credentials,
tool payloads, email/message bodies, or other request content.

## Timing surfaces

Typed chat records two layers:
1. Electron main-process timing: IPC/bridge total and streaming first-token time.
2. Python Assistant timing: `generate_reply()` emits `chat_latency` for routing, LLM,
   tool execution/resolution, completion, and total; `stream_reply()` emits
   `chat_stream_latency` with routing, first user-visible token, LLM, completion, and total.

Voice records wake acceptance, capture, STT, LLM, TTS start, playback completion,
and total pipeline timing. Provider TTS implementations also emit provider-specific
synthesis/playback timing where that boundary is observable. A combined TTS/playback
scope must not be presented as separate synthesis and playback evidence.

All new generic request timings use `time.perf_counter_ns()` so short Windows stages
are not quantized by a lower-resolution wall-clock source.

## Diagnostic target budgets

These are product targets, not claims about the current live system:
| Path/stage | Target |
|---|---:|
| Typed chat first token, warm p50 | <= 1.5 s |
| Typed chat first token, warm p95 | <= 3.0 s |
| Simple typed chat total, warm p95 | <= 8.0 s |
| Read-only tool total, warm p95 | <= 5.0 s |
| Verified mutating tool total, warm p95 | <= 7.0 s |
| Unavailable capability response, warm p95 | <= 1.0 s |
| Wake accepted -> capture start | <= 250 ms |
| STT after capture completes, p95 | <= 2.0 s |
| LLM first token for voice, warm p95 | <= 3.0 s |
| TTS request -> playback start, warm p95 | <= 1.0 s |
| End of utterance -> first spoken audio, warm p95 | <= 4.0 s |

Playback duration scales with spoken answer length, so it is recorded but does not
have a single fixed duration budget. Voice capture duration is user-controlled;
only capture/endpointing overhead should be optimized independently of speech length.

## RexBench baseline profile

Run the deterministic baseline with:

```bash
python scripts/rexbench.py --profile baseline --iterations 20 \
  --output docs/performance/rexbench-baseline.json
```

The baseline exercises the real current Assistant/ActionDispatcher and VoiceLoop
orchestration while mocking external provider, model, network, audio, and hardware
work. Typed-chat samples consume the real `Assistant.stream_reply()` latency event so
`first_token` is measured at the first user-visible emitted chunk rather than inferred
from non-streaming LLM duration. It reports cold/warm p50 and p95 for:
- typed chat,
- voice,
- read-only tools,
- mutating tools,
- unavailable capabilities.

Its evidence class is `deterministic_mock`. It validates instrumentation and framework
overhead only. It must not be used to claim that live provider or physical-device
latency meets the target budgets.

## Managed warm-runtime profile

US-099 adds a deterministic local lifecycle profile:

```bash
python scripts/rexbench.py --profile warm-runtime --iterations 8
```

The profile compares cold acquisition against already-warm acquisition for four synthetic
resource classes: executive/local model, STT, TTS, and retrieval/index. It exercises the
managed cache with deterministic local loader cost, so it proves cache/lifecycle overhead
and cold-vs-warm measurement behavior, not real model loading time. The runtime budget is an
approximate **retained-cache accounting** ceiling; it is not an exact RAM/VRAM measurement or
a guarantee that CUDA/library allocators release memory immediately after eviction. A
component that cannot fit the retained budget runs cold for that use rather than becoming
unavailable. The production mutable knowledge base is process-warm but intentionally outside
the evictable cache because its live size changes and callers retain it; the synthetic index
class here measures lifecycle behavior only. The report contains timings and non-secret
runtime identifiers only and never stores request content. Live model, audio, GPU, and device
latency remains a later release-gate measurement.

## Model-routing profile

US-110 adds a deterministic fast/deep routing profile:

```bash
python scripts/rexbench.py --profile model-routing --iterations 8
```

The profile covers simple commands, ambiguous tool choice, complex reasoning, provider outage,
and an unavailable local model. It validates bounded escalation, route/model selection, and
privacy-safe timing evidence only. It does not prove live provider quality, availability, cost,
or latency; US-111 owns provider reliability feedback and routing evaluation evidence.

## Identity-safe context cache

US-105 caches only deterministic private context fragments that are expensive to rebuild
repeatedly: personality prompt, formatted profile context, formatted remembered facts, and the
raw fact pairs used by `ContextPackage`. Dynamic date/time, current history, current message,
tool context, follow-up cues, response mode, action results, and final prompts remain uncached.

Cache validity is fail-closed. Private entries require a validated USER-scoped identity and
content-free revision digests for authority, selected model, capability/config state, relevant
memory/profile content, and prompt-template schema. Any relevant revision change produces a
miss. Household reuse requires an explicit household key; the private ContextBuilder path does
not share user artifacts at household scope.

Operational cache metrics are deliberately content-free: fixed cache category, hit/miss/build/
eviction counts, entry count, and aggregate build time only. They must never contain user IDs,
cache-key material, prompts, transcripts, memory/facts, credentials, filenames, or tool data.
A cache hit is performance evidence only; it does not widen authorization or change the
canonical TurnEngine/verification path.

## Known latency work after the baseline

Manual desktop testing has observed responses taking roughly 30-60+ seconds in some
real sessions. The deterministic baseline does not reproduce external model/audio
costs, so that observation remains unresolved until live RexBench evidence exists.

The integrated production-readiness plan already assigns the likely bottlenecks:
- US-094 through US-097: one canonical TurnEngine for streamed and non-streamed paths.
- US-099: managed warm local runtime/model lifetime.
- US-100: streaming ASR and semantic endpointing.
- US-101: safe speculative read-only prefetch.
- US-102: clause/sentence TTS streaming.
- US-103: barge-in through canonical cancellation.
- US-104: progressive status events while work is still running.
- US-105: identity-safe context caching.
- US-110/US-111: fast/deep model routing and provider reliability evidence.
- US-118: final live RexBench release gate.

A stage that exceeds budget in live evidence is a release blocker until the owning
story fixes it or the production-readiness tracker documents an explicit, justified
budget change.
