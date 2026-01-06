"""Simple Tkinter dashboard for monitoring the Rex assistant."""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

import asyncio
import threading
import tkinter as tk
from tkinter import ttk

from config import load_config
from logging_utils import get_logger
from memory_utils import load_recent_history
from rex_loop import AsyncRexAssistant

LOGGER = get_logger(__name__)


class AssistantGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Rex Assistant")
        self.geometry("480x360")
        self.resizable(False, False)

        self.config = load_config()
        self.assistant: AsyncRexAssistant | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        self.status_var = tk.StringVar(value="Idle")
        self.user_var = tk.StringVar(value=f"Active user: {self.config.default_user or 'auto'}")

        # --- UI Setup ---
        ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 14)).pack(pady=10)
        ttk.Label(self, textvariable=self.user_var).pack(pady=5)
        ttk.Button(self, text="Start", command=self.start_assistant).pack(pady=5)
        ttk.Button(self, text="Stop", command=self.stop_assistant).pack(pady=5)

        ttk.Label(self, text="Recent conversation:").pack(pady=(20, 5))
        self.history_box = tk.Text(self, height=8, width=55, state="disabled", wrap="word")
        self.history_box.pack(padx=10, pady=(0, 10))

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(2000, self.refresh_history)

    def start_assistant(self) -> None:
        if self.running:
            return
        if not self.assistant:
            self.assistant = AsyncRexAssistant(self.config)

        self.running = True
        self.status_var.set("Running…")
        self.thread = threading.Thread(target=self._run_assistant, daemon=True)
        self.thread.start()

    def _run_assistant(self) -> None:
        try:
            asyncio.run(self.assistant.run())
        except Exception as exc:
            LOGGER.exception("Assistant crashed")
            self.status_var.set(f"Error: {exc}")
        finally:
            self.running = False

    def stop_assistant(self) -> None:
        if self.assistant:
            self.assistant.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.running = False
        self.status_var.set("Stopped")

    def refresh_history(self) -> None:
        if self.assistant:
            user = self.assistant.active_user
            recent = load_recent_history(user, limit=6)
            if recent:
                self.history_box.configure(state="normal")
                self.history_box.delete("1.0", tk.END)
                for entry in recent:
                    self.history_box.insert(tk.END, f"{entry['role']}: {entry['text']}\n")
                self.history_box.see(tk.END)
                self.history_box.configure(state="disabled")
                self.user_var.set(f"Active user: {user}")
        self.after(2000, self.refresh_history)

    def on_close(self) -> None:
        self.stop_assistant()
        self.destroy()


if __name__ == "__main__":  # pragma: no cover
    gui = AssistantGUI()
    gui.mainloop()
