# US-099 Managed Warm Local Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep heavyweight local LLM, STT, TTS, and retrieval/index resources warm under a bounded process-local lifecycle with truthful diagnostics and benchmark evidence.

**Architecture:** `rex.runtime.warm` owns a content-free, bounded cache of heavyweight resources. Consumers share only immutable/model-style heavyweight resources; request/user state stays in Assistant/voice/tool objects. Idle and LRU eviction are explicit, optional dependency failures degrade lazily, and diagnostics never load a cold component merely to inspect it.

**Tech Stack:** Python 3.11, existing Rex runtime/config/doctor modules, pytest, RexBench.

## Global Constraints

- Do not add heavy dependencies to the base install.
- Do not cache prompts, transcripts, memory contents, credentials, or per-user authorization state.
- Resource cost is approximate diagnostic metadata, never a claim about exact RSS/VRAM.
- Keep existing public constructors compatible unless a new optional keyword is required.

---
### Task 1: Bounded warm lifecycle

**Files:** Create `rex/runtime/warm.py`; test `tests/rex2/test_warm_runtime.py`.

- [ ] Prove tests fail without `rex.runtime.warm`.
- [ ] Implement lazy load, reuse, leases, LRU budget eviction, idle eviction, degraded fallback, and content-free status snapshots.
- [ ] Run `pytest tests/rex2/test_warm_runtime.py -q`.

### Task 2: Share heavyweight local resources

**Files:** Modify `rex/llm_client.py`, `rex/voice/stt.py`, `rex/voice/tts.py`, `rex/knowledge_base.py`; extend `tests/rex2/test_warm_runtime.py`.

- [ ] Add failing consumer tests proving repeated local LLM/STT/TTS/index construction reloads today.
- [ ] Route only heavyweight resource construction through the warm manager; keep request/user state outside it.
- [ ] Preserve lazy optional-dependency fallback and existing public APIs.
- [ ] Run focused LLM/voice/knowledge regression tests.

### Task 3: Diagnostics and bounded policy

**Files:** Modify `rex/config.py`, `rex/config_manager.py`, `config/rex_config.json`, `rex/doctor.py`, tests.

- [ ] Add bounded warm-runtime policy settings with conservative defaults and validation.
- [ ] Add a doctor diagnostic that reports warm component names/states/approximate cost without loading them.
- [ ] Add tests for invalid bounds and content-free diagnostics.
### Task 4: RexBench and story closeout

**Files:** Modify `scripts/rexbench.py`, `CLAUDE.md`, `docs/ARCHITECTURE.md`, `PRD-production-readiness.md`, `docs/archive/progress/progress-production-readiness.txt`; add/adjust benchmark tests.

- [ ] Add deterministic `warm-runtime` benchmark comparing cold construction with reused warm resources; benchmark fixtures contain no user content.
- [ ] Run `python scripts/rexbench.py --profile warm-runtime` and contract tests.
- [ ] Document warm-runtime ownership, privacy boundary, and resource-bound rules.
- [ ] Close US-104's GitHub criterion using PR #393 evidence; mark only US-099 local criteria complete, leaving its GitHub criterion open.
- [ ] Run focused regressions, Ruff, Black, mypy, pre-commit, security guards, and `git diff --check` before publication.
- [ ] Publish PR, verify every required check on the exact head, merge, then close US-099's final criterion in the next story tracking commit.
