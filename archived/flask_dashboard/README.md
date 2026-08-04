# Retired Flask Dashboard Tests

This directory preserves placeholder tests for the removed `rex/dashboard/` Flask UI surface.
They were moved out of the active `tests/` tree by US-039 on 2026-08-04.

The preserved files are:

- `test_us149_gui_shell.py`
- `test_us150_design_system.py`
- `test_us151_nav_state.py`
- `test_us152_chat_message_list.py`
- `test_us153_chat_input.py`
- `test_us157_voice_waveform.py`
- `test_us161_schedule_coming_up.py`
- `test_us163_overview_quick_actions.py`
- `test_us164_hover_focus_states.py`
- `test_us165_loading_error_states.py`
- `test_us166_responsive_layout.py`
- `test_us172_thinking_indicator.py`
- `test_us174.py`

The files are historical reference only. They are not collected by pytest, maintained, or used
to make claims about the current product. The supported desktop GUI is the React + Electron app
under `gui/`, with active tests under `gui/tests/` and installed-artifact validation in CI.

`tests/test_us174_voice_max_tokens.py` remains active because it contains current voice-response
length coverage; only its obsolete dashboard-route assertion was removed.
