# DEPRECATED: Use `askrex-gui` (web dashboard) instead.
# This Tkinter launcher will be removed in the next major release.
# See docs/UI_SURFACES.md for the canonical GUI entry point.
#!/usr/bin/env python3
"""Deprecated Tkinter GUI entrypoint for Rex AI Assistant."""

if __name__ == "__main__":
    from gui import AssistantGUI

    gui = AssistantGUI()
    gui.mainloop()
