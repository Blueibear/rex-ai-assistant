# Skipped Tests Inventory

Generated for `US-002` on 2026-06-14 from the current `tests/` tree.

## Validation Snapshot

- `pytest --collect-only -q`: passed; collected 6635 tests, with 2 module-level skips during collection.
- `rg -n "pytest\.mark\.skip|pytest\.skip\b" tests | Measure-Object`: 143 grep-style matches.
- AST-confirmed skip marker/call sites inventoried below: 140.
- Grep-style matches excluded from the inventory because they are not executable skip sites: `tests/test_us129_smoke.py:369`, `tests/test_us129_smoke.py:370`, `tests/test_us174.py:4`.

## Classification Summary

| Classification | Count | Follow-up |
|---|---:|---|
| `optional-dep-skip` | 22 | permanent: optional dependency/tool/environment guard |
| `platform-skip` | 10 | permanent: platform/runtime-specific guard |
| `retired-surface-skip` | 14 | US-039 |
| `temporary-bug-skip` | 94 | US-038 |

## Inventory

| File | Line | Skip type | Skip reason text | Classification | Follow-up |
|---|---:|---|---|---|---|
| `tests/conftest.py` | 160 | `skip` | No async test runner installed (anyio or pytest-asyncio required) | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_audio_device_selection.py` | 9 | `pytest.skip` | f"sounddevice unavailable: {exc}" | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_calendar_service.py` | 131 | `skipif` | CalendarService event-bus style API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 184 | `skipif` | CalendarEvent is not a pydantic-style model in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 211 | `skipif` | CalendarEvent.overlaps_with not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 241 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 251 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 261 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 270 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 287 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 299 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 309 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 337 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 352 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 367 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 383 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 393 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 403 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 422 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 431 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 440 | `skipif` | CalendarService.find_conflicts not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 463 | `skipif` | CalendarService.find_conflicts not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 485 | `skipif` | CalendarService.find_conflicts not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 495 | `skipif` | CalendarService.get_upcoming_events not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 506 | `skipif` | CalendarService.get_upcoming_events not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 517 | `skipif` | CalendarService.get_all_events not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 528 | `skipif` | Mock-file calendar service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_calendar_service.py` | 552 | `skipif` | CalendarEvent serialization test applies to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 96 | `skipif` | EmailService event-bus style API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 187 | `skipif` | EmailSummary pydantic-style model not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 212 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 222 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 232 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 241 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 253 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 263 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 273 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 290 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 299 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 308 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 326 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 344 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 362 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 380 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 399 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 417 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 430 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 440 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 450 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 464 | `skipif` | Mock-file email service tests apply to the newer implementation only. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 474 | `skipif` | EmailSummary pydantic-style model not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 498 | `skipif` | EmailSummary pydantic-style model not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_email_service.py` | 528 | `skipif` | EmailSummary pydantic-style model not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 79 | `skipif` | Legacy EventBus.publish(event_type, payload) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 96 | `skipif` | Legacy EventBus.publish(event_type, payload) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 123 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 137 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 151 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 161 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 173 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 189 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 210 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 233 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 259 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 274 | `skipif` | Legacy EventBus.publish(event_type, payload) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_event_bus.py` | 293 | `skipif` | Newer EventBus.publish(Event) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_install_scripts.py` | 15 | `pytest.skip` | bash not available | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_install_scripts.py` | 24 | `pytest.skip` | bash not usable on Windows | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_openclaw_root_voice_loop_flag.py` | 19 | `skipif` | openwakeword is not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_openclaw_root_voice_loop_text_mode.py` | 20 | `skipif` | openwakeword is not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_scheduler.py` | 90 | `skipif` | Legacy Scheduler(storage_path, now_func) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 115 | `skipif` | Legacy Scheduler(storage_path, now_func) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 157 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 167 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 191 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 211 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 225 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 241 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 254 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 266 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 279 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 311 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 337 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 363 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 391 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 407 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 427 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 450 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 463 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 493 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 502 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_scheduler.py` | 521 | `skipif` | Newer Scheduler(jobs_file=...) API not available in this build. | `temporary-bug-skip` | US-038 |
| `tests/test_skill_loader.py` | 213 | `pytest.skip` | example_weather_skill.py not found | `temporary-bug-skip` | US-038 |
| `tests/test_transformers_shim.py` | 30 | `pytest.skip` | f"transformers not installed or BeamSearchScorer unavailable: {e}" | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_transformers_shim.py` | 48 | `pytest.skip` | transformers not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_transformers_shim.py` | 63 | `pytest.skip` | transformers not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us011_voice_deps.py` | 43 | `pytest.skip` | edge-tts not installed in this environment | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us011_voice_deps.py` | 53 | `pytest.skip` | pyttsx3 not installed in this environment | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us017_stt_backend.py` | 72 | `skipif` | numpy required for SpeechToText.transcribe | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us027_device_approval.py` | 129 | `skipif` | rex.cli requires Python 3.11 | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us027_device_approval.py` | 161 | `skipif` | rex.cli requires Python 3.11 | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us027_device_approval.py` | 168 | `skipif` | rex.cli requires Python 3.11 | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us027_device_approval.py` | 180 | `skipif` | rex.cli requires Python 3.11 | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us027_device_approval.py` | 205 | `skipif` | rex.cli requires Python 3.11 | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us027_device_approval.py` | 230 | `skipif` | rex.cli requires Python 3.11 | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us030_clarification.py` | 184 | `pytest.skip` | HABridge not importable (requests not installed) | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us030_clarification.py` | 204 | `pytest.skip` | HABridge not importable (requests not installed) | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us033_acknowledgment.py` | 11 | `skipif` | numpy not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us034_progressive_response.py` | 22 | `skipif` | numpy not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us053_secret_management.py` | 125 | `pytest.skip` | .env.example not found | `temporary-bug-skip` | US-038 |
| `tests/test_us071_debug_mode.py` | 18 | `skipif` | rex.cli unavailable on this Python version | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us074_voice_loop_pipeline.py` | 33 | `skipif` | numpy not installed | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us096_secret_scan.py` | 130 | `pytest.skip` | PyYAML not installed; skipping YAML parse test | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us098_test_coverage.py` | 28 | `pytest.skip` | coverage.txt has not been generated yet | `temporary-bug-skip` | US-038 |
| `tests/test_us098_test_coverage.py` | 31 | `pytest.skip` | coverage.txt is still being written by the current coverage run | `temporary-bug-skip` | US-038 |
| `tests/test_us120_performance_baseline.py` | 180 | `pytest.skip` | docs/performance-baseline.md not yet created | `temporary-bug-skip` | US-038 |
| `tests/test_us120_performance_baseline.py` | 187 | `pytest.skip` | docs/performance-baseline.md not yet created | `temporary-bug-skip` | US-038 |
| `tests/test_us129_smoke.py` | 54 | `pytest.skip` | Local Rex instance not running — start with 'python flask_proxy.py' to exercise live-server smoke tests | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us129_smoke.py` | 323 | `pytest.skip` | f"agent_server optional dependency not available: {exc}" | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us139_install_scripts.py` | 139 | `skipif` | executable bit not relevant on Windows | `platform-skip` | permanent: platform/runtime-specific guard |
| `tests/test_us140_full_extra.py` | 138 | `pytest.skip` | install.sh not present | `temporary-bug-skip` | US-038 |
| `tests/test_us140_full_extra.py` | 145 | `pytest.skip` | install.ps1 not present | `temporary-bug-skip` | US-038 |
| `tests/test_us149_gui_shell.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us150_design_system.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us151_nav_state.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us152_chat_message_list.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us153_chat_input.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us157_voice_waveform.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us161_schedule_coming_up.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us163_overview_quick_actions.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us164_hover_focus_states.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us165_loading_error_states.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us166_responsive_layout.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us172_thinking_indicator.py` | 10 | `skip` | rex/dashboard retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us174.py` | 11 | `skip` | dashboard routes retired — see test_us174_voice_max_tokens.py | `retired-surface-skip` | US-039 |
| `tests/test_us174_voice_max_tokens.py` | 129 | `skip` | rex/dashboard/routes.py retired in OpenClaw migration (US-P7-014) | `retired-surface-skip` | US-039 |
| `tests/test_us304_chat_stream_electron_verification.py` | 15 | `skipif` | electron.cmd not found | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_us304_chat_stream_electron_verification.py` | 30 | `pytest.skip` | f"Electron verification unavailable: {exc}" | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_voice_id_mvp.py` | 590 | `pytest.skip` | speechbrain is installed — cannot test missing-dep path | `optional-dep-skip` | permanent: optional dependency/tool/environment guard |
| `tests/test_windows_service.py` | 10 | `pytest.skip` | rex.windows_service is Windows-only | `platform-skip` | permanent: platform/runtime-specific guard |
