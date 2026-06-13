# Skipped-Test Inventory

**Generated:** 2026-06-13
**Command:** `grep -rn "pytest.mark.skip\|pytest.skip\b" tests | wc -l`
**Raw grep hits:** 142 (2 are inside docstrings/assertion strings — not real skip calls)
**Actionable skip markers:** 140

---

## Classifications

| Label | Meaning |
|---|---|
| `optional-dep-skip` | Skipped because an optional Python package, service, or build variant is absent |
| `platform-skip` | Skipped because the test is not applicable on the current OS or Python version |
| `retired-surface-skip` | Skipped because the tested surface (`rex/dashboard`) was retired in OpenClaw migration US-P7-014 |
| `temporary-bug-skip` | Skipped because a required artifact (file, running service) has not been created yet |

## Summary

| Classification | Count | Action |
|---|---|---|
| `optional-dep-skip` | 111 | Permanent — remove only when dep is made mandatory |
| `platform-skip` | 11 | Permanent — correct by design |
| `retired-surface-skip` | 14 | Delete test files to clean up skip budget (no new story needed) |
| `temporary-bug-skip` | 4 | Resolve when the blocking story creates the missing artifact |
| **Total** | **140** | |

---

## False Positives in Raw grep

These lines match the grep pattern but are **not** real skip calls:

| File | Line | Content | Why false positive |
|---|---|---|---|
| `tests/test_us174.py` | 4 | `@pytest.mark.skip(reason="rex/dashboard/routes.py retired in OpenClaw migration").` | Inside a module-level docstring, not executable code |
| `tests/test_us129_smoke.py` | 370 | `"Smoke test file must contain pytest.skip() for graceful skips"` | String inside an assertion, not a skip call |

---

## Inventory by File

### tests/conftest.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 160 | `"No async test runner installed (anyio or pytest-asyncio required)"` — defines `skip_no_async` marker applied to individual async tests | `optional-dep-skip` | permanent |

---

### tests/test_audio_device_selection.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 9 | `f"sounddevice unavailable: {exc}"` — module-level skip when `sounddevice` import fails | `optional-dep-skip` | permanent |

---

### tests/test_calendar_service.py

All skips guard against tests that require a Pydantic-style `CalendarEvent` model or specific `CalendarService` methods not present in all build variants.

| Lines | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 131 | `"CalendarService event-bus style API not available in this build."` | `optional-dep-skip` | permanent |
| 184, 241, 251, 261, 270, 287, 299, 309, 337, 352, 367, 383, 393, 403, 422, 431, 528 | `"CalendarEvent is not a pydantic-style model in this build."` / `"Mock-file calendar service tests apply to the newer implementation only."` | `optional-dep-skip` | permanent |
| 211 | `"CalendarEvent.overlaps_with not available in this build."` | `optional-dep-skip` | permanent |
| 440, 463, 485 | `"CalendarService.find_conflicts not available in this build."` | `optional-dep-skip` | permanent |
| 495, 506 | `"CalendarService.get_upcoming_events not available in this build."` | `optional-dep-skip` | permanent |
| 517 | `"CalendarService.get_all_events not available in this build."` | `optional-dep-skip` | permanent |
| 552 | `"CalendarEvent serialization test applies to the newer implementation only."` | `optional-dep-skip` | permanent |

**Count:** 26 markers

---

### tests/test_email_service.py

All skips guard against tests that require a Pydantic-style `EmailSummary` model or the event-bus `EmailService` API variant.

| Lines | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 96 | `"EmailService event-bus style API not available in this build."` | `optional-dep-skip` | permanent |
| 187, 474, 498, 528 | `"EmailSummary pydantic-style model not available in this build."` | `optional-dep-skip` | permanent |
| 212, 222, 232, 241, 253, 263, 273, 290, 299, 308, 326, 344, 362, 380, 399, 417, 430, 440, 450, 464 | `"Mock-file email service tests apply to the newer implementation only."` | `optional-dep-skip` | permanent |

**Count:** 25 markers

---

### tests/test_event_bus.py

All skips guard against API variant mismatches for `EventBus.publish`.

| Lines | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 79, 96, 274 | `"Legacy EventBus.publish(event_type, payload) API not available in this build."` | `optional-dep-skip` | permanent |
| 123, 137, 151, 161, 173, 189, 210, 233, 259, 293 | `"Newer EventBus.publish(Event) API not available in this build."` | `optional-dep-skip` | permanent |

**Count:** 13 markers

---

### tests/test_install_scripts.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 15 | `"bash not available"` | `platform-skip` | permanent |
| 24 | `"bash not usable on Windows"` | `platform-skip` | permanent |

---

### tests/test_openclaw_root_voice_loop_flag.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 19 | `"openwakeword is not installed"` — module-level `pytestmark` | `optional-dep-skip` | permanent |

---

### tests/test_openclaw_root_voice_loop_text_mode.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 20 | `"openwakeword is not installed"` — module-level `pytestmark` | `optional-dep-skip` | permanent |

---

### tests/test_scheduler.py

All skips guard against API variant mismatches for `Scheduler.__init__` signature.

| Lines | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 90, 115 | `"Legacy Scheduler(storage_path, now_func) API not available in this build."` | `optional-dep-skip` | permanent |
| 157, 167, 191, 211, 225, 241, 254, 266, 279, 311, 337, 363, 391, 407, 427, 450, 463, 493, 502, 521 | `"Newer Scheduler(jobs_file=...) API not available in this build."` | `optional-dep-skip` | permanent |

**Count:** 22 markers

---

### tests/test_skill_loader.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 213 | `"example_weather_skill.py not found"` | `optional-dep-skip` | permanent |

---

### tests/test_transformers_shim.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 30 | `f"transformers not installed or BeamSearchScorer unavailable: {e}"` | `optional-dep-skip` | permanent |
| 48 | `"transformers not installed"` | `optional-dep-skip` | permanent |
| 63 | `"transformers not installed"` | `optional-dep-skip` | permanent |

---

### tests/test_us011_voice_deps.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 43 | `"edge-tts not installed in this environment"` | `optional-dep-skip` | permanent |
| 53 | `"pyttsx3 not installed in this environment"` | `optional-dep-skip` | permanent |

---

### tests/test_us017_stt_backend.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 72 | `"numpy required for SpeechToText.transcribe"` | `optional-dep-skip` | permanent |

---

### tests/test_us027_device_approval.py

| Lines | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 129, 161, 168, 180, 205, 230 | `"rex.cli requires Python 3.11"` | `platform-skip` | permanent |

---

### tests/test_us030_clarification.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 184 | `"HABridge not importable (requests not installed)"` | `optional-dep-skip` | permanent |
| 204 | `"HABridge not importable (requests not installed)"` | `optional-dep-skip` | permanent |

---

### tests/test_us033_acknowledgment.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 11 | `"numpy not installed"` — defines `_skip_no_numpy` marker applied to tests in this file | `optional-dep-skip` | permanent |

---

### tests/test_us034_progressive_response.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 22 | `"numpy not installed"` — defines `_needs_numpy` marker applied to tests in this file | `optional-dep-skip` | permanent |

---

### tests/test_us053_secret_management.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 125 | `".env.example not found"` | `optional-dep-skip` | permanent |

---

### tests/test_us071_debug_mode.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 18 | `"rex.cli unavailable on this Python version"` — defines `requires_cli` marker | `platform-skip` | permanent |

---

### tests/test_us074_voice_loop_pipeline.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 33 | `"numpy not installed"` — defines `_requires_numpy` marker | `optional-dep-skip` | permanent |

---

### tests/test_us096_secret_scan.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 130 | `"PyYAML not installed; skipping YAML parse test"` | `optional-dep-skip` | permanent |

---

### tests/test_us098_test_coverage.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 28 | `"coverage.txt has not been generated yet"` | `temporary-bug-skip` | resolve: run `pytest --cov` before this test, or gate on CI coverage run |
| 31 | `"coverage.txt is still being written by the current coverage run"` | `temporary-bug-skip` | same as above — fires only in isolated runs |

---

### tests/test_us120_performance_baseline.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 180 | `"docs/performance-baseline.md not yet created"` | `temporary-bug-skip` | resolve: create `docs/performance-baseline.md` (future US-120 story) |
| 187 | `"docs/performance-baseline.md not yet created"` | `temporary-bug-skip` | same |

---

### tests/test_us129_smoke.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 54 | `"Local Rex instance not running — start with 'python flask_proxy.py' to exercise live-server smoke tests"` | `optional-dep-skip` | permanent — requires a running service |
| 323 | `f"agent_server optional dependency not available: {exc}"` | `optional-dep-skip` | permanent |

---

### tests/test_us139_install_scripts.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 139 | `"executable bit not relevant on Windows"` | `platform-skip` | permanent |

---

### tests/test_us140_full_extra.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 138 | `"install.sh not present"` | `optional-dep-skip` | permanent — graceful degradation |
| 145 | `"install.ps1 not present"` | `optional-dep-skip` | permanent — graceful degradation |

---

### tests/test_us149_gui_shell.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us150_design_system.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us151_nav_state.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us152_chat_message_list.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us153_chat_input.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us157_voice_waveform.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us161_schedule_coming_up.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us163_overview_quick_actions.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us164_hover_focus_states.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us165_loading_error_states.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us166_responsive_layout.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us172_thinking_indicator.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex/dashboard retired in OpenClaw migration (US-P7-014)"` — module-level `pytestmark` | `retired-surface-skip` | delete file |

---

### tests/test_us174.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 11 | `"dashboard routes retired — see test_us174_voice_max_tokens.py"` | `retired-surface-skip` | delete file |

---

### tests/test_us174_voice_max_tokens.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 129 | `"rex/dashboard/routes.py retired in OpenClaw migration (US-P7-014)"` — single test method inside an otherwise active file | `retired-surface-skip` | remove the single skipped test method |

---

### tests/test_us304_chat_stream_electron_verification.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 15 | `"electron.cmd not found"` — module-level `pytestmark` | `optional-dep-skip` | permanent — requires packaged Electron build |
| 30 | `f"Electron verification unavailable: {exc}"` | `optional-dep-skip` | permanent |

---

### tests/test_voice_id_mvp.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 590 | `"speechbrain is installed — cannot test missing-dep path"` | `optional-dep-skip` | permanent — inverted guard; skips the "missing dep" test path when speechbrain IS present |

---

### tests/test_windows_service.py

| Line | Skip Reason | Classification | Follow-up |
|---|---|---|---|
| 10 | `"rex.windows_service is Windows-only"` — module-level skip on non-Windows | `platform-skip` | permanent |

---

## Skip Budget

**Current total: 140**

Recommended cleanup target: remove the 14 `retired-surface-skip` test files (and the 1 skipped method in `test_us174_voice_max_tokens.py`). This reduces the skip count to **125** without any behavior change — the retired dashboard surface has no production path.

The 4 `temporary-bug-skip` entries should resolve as their blocking stories complete:
- `test_us098_test_coverage.py` — resolves when `coverage.txt` is present (generated by a full `pytest --cov` run)
- `test_us120_performance_baseline.py` — resolves when `docs/performance-baseline.md` is created (US-120)
