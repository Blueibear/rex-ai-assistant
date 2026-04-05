# UI Surfaces

This document is the authoritative inventory of the UI and user-facing service surfaces that ship in this repository.

| Surface | Entry point | Status | Reason |
|---|---|---|---|
| CLI (text chat) | `rex` | **Primary — keep** | Core text interface |
| Voice loop | `python rex_loop.py` | **Primary — keep** | Core voice interface |
| Web dashboard | `rex-gui` | **Primary GUI — keep** | React, modern, canonical |
| Shopping PWA | served by `rex` or `rex-gui` | **Optional feature — keep** | Functional feature surface |
| TTS API | `rex-speak-api` | **Service component — keep** | Required by voice loop |
| Tkinter window (`gui.py`) | `python run_gui.py` | **Deprecated** | Superseded by web dashboard |

## Notes

- Historical planning text may still mention `askrex`, `askrex-gui`, or `askrex-speak-api`.
- The current packaged console scripts in `pyproject.toml` are `rex`, `rex-gui`, and `rex-speak-api`.
