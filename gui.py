"""Simple Tkinter dashboard for monitoring the Rex assistant."""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

import asyncio
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import ttk

# Suppress torio FFmpeg extension warnings (non-critical audio codec features)
warnings.filterwarnings("ignore", message=".*FFmpeg extension.*")
warnings.filterwarnings("ignore", message=".*libtorio.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torio")

from config import load_config
from logging_utils import get_logger
from memory_utils import load_recent_history
from voice_loop import AsyncRexAssistant

LOGGER = get_logger(__name__)


class AssistantGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Rex Assistant")
        self.geometry("800x600")
        self.resizable(True, True)

        self.config = load_config()
        self.assistant: AsyncRexAssistant | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        self.status_var = tk.StringVar(value="Idle")
        self.user_var = tk.StringVar(value=f"Active user: {self.config.default_user or 'auto'}")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create Dashboard tab
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self._create_dashboard()

        # Create Settings tab
        self._create_settings_tab()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(2000, self.refresh_history)

    def _create_dashboard(self) -> None:
        """Create the dashboard tab content."""
        # --- UI Setup ---
        ttk.Label(self.dashboard_tab, textvariable=self.status_var, font=("Segoe UI", 14)).pack(pady=10)
        ttk.Label(self.dashboard_tab, textvariable=self.user_var).pack(pady=5)

        # Button frame
        button_frame = ttk.Frame(self.dashboard_tab)
        button_frame.pack(pady=5)
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_assistant)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_assistant, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        ttk.Label(self.dashboard_tab, text="Recent conversation:").pack(pady=(20, 5))
        self.history_box = tk.Text(self.dashboard_tab, height=8, width=55, state="disabled", wrap="word")
        self.history_box.pack(padx=10, pady=(0, 10))

        # Log area for errors and status
        ttk.Label(self.dashboard_tab, text="System Log:").pack(pady=(10, 5))
        self.log_box = tk.Text(self.dashboard_tab, height=6, width=55, state="disabled", wrap="word", bg="#f0f0f0")
        self.log_box.pack(padx=10, pady=(0, 10))
        self._log_to_gui("Ready. Click Start to begin.")

    def _create_settings_tab(self) -> None:
        """Create the Settings tab."""
        try:
            from gui_settings_tab import SettingsTab
            repo_root = Path(__file__).resolve().parent
            self.settings_tab = SettingsTab(self.notebook, repo_root)
            self.notebook.add(self.settings_tab, text="Settings")
        except Exception as e:
            LOGGER.exception("Failed to create Settings tab")
            # Create error tab
            error_frame = ttk.Frame(self.notebook)
            ttk.Label(
                error_frame,
                text=f"Failed to load Settings tab:\n{e}",
                foreground="red"
            ).pack(padx=20, pady=20)
            self.notebook.add(error_frame, text="Settings")

    def _log_to_gui(self, message: str) -> None:
        """Log a message to the GUI log box."""
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert(tk.END, f"{message}\n")
            self.log_box.see(tk.END)
            self.log_box.configure(state="disabled")
        except Exception:
            pass  # Don't crash if logging fails

    def start_assistant(self) -> None:
        """Start the voice assistant with comprehensive error handling."""
        if self.running:
            self._log_to_gui("Already running.")
            return

        try:
            self._log_to_gui("Initializing assistant...")
            if not self.assistant:
                self.assistant = AsyncRexAssistant(self.config)

            self.running = True
            self.status_var.set("Starting…")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self._log_to_gui("Starting audio pipeline...")

            self.thread = threading.Thread(target=self._run_assistant, daemon=True)
            self.thread.start()
            self._log_to_gui("Assistant started successfully.")
            self.status_var.set("Running")

        except Exception as exc:
            import traceback
            error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            LOGGER.exception("Failed to start assistant")
            self._log_to_gui(f"ERROR starting assistant:\n{error_details}")
            self.status_var.set(f"Start Failed: {exc}")
            self.running = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _run_assistant(self) -> None:
        """Run the assistant loop with comprehensive error handling."""
        try:
            asyncio.run(self.assistant.run())
        except Exception as exc:
            import traceback
            error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            LOGGER.exception("Assistant crashed during execution")
            self._log_to_gui(f"Assistant crashed:\n{error_details}")
            self.after(0, lambda: self.status_var.set(f"Crashed: {exc}"))
        finally:
            self.running = False
            self.after(0, lambda: self.start_button.configure(state="normal"))
            self.after(0, lambda: self.stop_button.configure(state="disabled"))

    def stop_assistant(self) -> None:
        """Stop the assistant with proper cleanup."""
        self._log_to_gui("Stopping assistant...")
        if self.assistant:
            try:
                self.assistant.stop()
            except Exception as exc:
                LOGGER.exception("Error stopping assistant")
                self._log_to_gui(f"Error during stop: {exc}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        self.running = False
        self.status_var.set("Stopped")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._log_to_gui("Assistant stopped.")

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
