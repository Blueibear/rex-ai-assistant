#!/usr/bin/env python3
"""GUI entrypoint for Rex AI Assistant.

This is the canonical way to launch the Rex GUI on Windows.
Simply run: python run_gui.py
"""

if __name__ == "__main__":
    from gui import AssistantGUI

    gui = AssistantGUI()
    gui.mainloop()
