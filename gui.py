"""Simple Tkinter dashboard for monitoring the Rex assistant."""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

import asyncio
import logging
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
from utils.audio_device import (
    enumerate_input_devices,
    load_audio_config,
    save_audio_config,
)
from voice_loop import AsyncRexAssistant

LOGGER = get_logger(__name__)


class GUILogHandler(logging.Handler):
    """Custom logging handler that sends log messages to the GUI."""

    def __init__(self, gui_log_callback):
        super().__init__()
        self.gui_log_callback = gui_log_callback
        # Set format
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        self.setFormatter(formatter)

    def emit(self, record):
        try:
            msg = self.format(record)
            # Call the GUI callback (thread-safe)
            self.gui_log_callback(msg)
        except Exception:
            self.handleError(record)


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

        # Install logging handler to capture logs from background threads
        self._install_log_handler()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(2000, self.refresh_history)

    def _create_dashboard(self) -> None:
        """Create the dashboard tab content."""
        # --- Audio Input Section ---
        audio_frame = ttk.LabelFrame(self.dashboard_tab, text="Audio Input", padding=10)
        audio_frame.pack(padx=10, pady=10, fill="x")

        # Device selection
        device_frame = ttk.Frame(audio_frame)
        device_frame.pack(fill="x", pady=5)
        ttk.Label(device_frame, text="Input Device:").pack(side="left", padx=(0, 10))

        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly", width=40)
        self.device_combo.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(device_frame, text="Refresh", command=self._refresh_audio_devices, width=10).pack(side="left", padx=(0, 5))
        ttk.Button(device_frame, text="Test Mic", command=self._test_selected_mic, width=10).pack(side="left")

        # Populate devices
        self._refresh_audio_devices()

        # --- Status Section ---
        ttk.Label(self.dashboard_tab, textvariable=self.status_var, font=("Segoe UI", 14)).pack(pady=10)
        ttk.Label(self.dashboard_tab, textvariable=self.user_var).pack(pady=5)

        # Wake word info
        wake_keyword = self.config.wakeword_keyword or self.config.wakeword
        self.wake_var = tk.StringVar(value=f"Wake word: \"{wake_keyword}\" (threshold: {self.config.wakeword_threshold})")
        ttk.Label(self.dashboard_tab, textvariable=self.wake_var, font=("Segoe UI", 9), foreground="blue").pack(pady=2)

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

    def _install_log_handler(self) -> None:
        """Install custom log handler to capture logs from background threads."""
        # Create handler that calls our GUI logging method
        handler = GUILogHandler(self._log_to_gui_threadsafe)
        handler.setLevel(logging.DEBUG)  # Capture DEBUG and above for troubleshooting

        # Add to root logger so we get logs from voice_loop, wakeword_utils, etc.
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        # Set specific loggers to DEBUG level
        logging.getLogger("voice_loop").setLevel(logging.DEBUG)
        logging.getLogger("wakeword_utils").setLevel(logging.DEBUG)

        LOGGER.info("GUI log handler installed (level: DEBUG)")

    def _log_to_gui_threadsafe(self, message: str) -> None:
        """Thread-safe logging to GUI - can be called from any thread."""
        # Use after() to execute on main thread
        self.after(0, lambda: self._log_to_gui(message))

    def _log_to_gui(self, message: str) -> None:
        """Log a message to the GUI log box (must be called from main thread)."""
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert(tk.END, f"{message}\n")
            self.log_box.see(tk.END)
            self.log_box.configure(state="disabled")
        except Exception:
            pass  # Don't crash if logging fails

    def _refresh_audio_devices(self) -> None:
        """Refresh the list of available audio input devices."""
        try:
            devices = enumerate_input_devices()
            if not devices:
                self.device_combo["values"] = ["No input devices found"]
                self.device_var.set("No input devices found")
                self._log_to_gui("WARNING: No audio input devices found")
                return

            # Format device list for dropdown
            device_list = [d.display_name() for d in devices]
            self.device_combo["values"] = device_list

            # Load saved config and select that device
            audio_config = load_audio_config()
            saved_device_idx = audio_config.get("input_device_index")

            if saved_device_idx is not None:
                # Find the device in the list
                for i, device in enumerate(devices):
                    if device.index == saved_device_idx:
                        self.device_combo.current(i)
                        self._log_to_gui(f"Loaded saved device: {device.display_name()}")
                        return

            # No saved device or not found - select first device
            if device_list:
                self.device_combo.current(0)
                self._log_to_gui(f"Selected default device: {devices[0].display_name()}")

        except Exception as exc:
            LOGGER.exception("Failed to refresh audio devices")
            self._log_to_gui(f"ERROR refreshing devices: {exc}")
            self.device_combo["values"] = ["Error loading devices"]
            self.device_var.set("Error loading devices")

    def _get_selected_device_index(self) -> int | None:
        """Get the device index from the currently selected dropdown item."""
        try:
            selected = self.device_var.get()
            if not selected or selected in ["No input devices found", "Error loading devices"]:
                return None

            # Extract device index from "12: Device Name [API]" format
            if ":" in selected:
                idx_str = selected.split(":")[0].strip()
                return int(idx_str)
        except Exception as exc:
            LOGGER.exception("Failed to parse device index")
            self._log_to_gui(f"ERROR parsing device: {exc}")

        return None

    def _save_audio_device(self) -> None:
        """Save the currently selected audio device to config."""
        try:
            device_idx = self._get_selected_device_index()
            if device_idx is not None:
                save_audio_config(device_idx)
                self._log_to_gui(f"Saved audio device: {device_idx}")
        except Exception as exc:
            LOGGER.exception("Failed to save audio device")
            self._log_to_gui(f"ERROR saving device: {exc}")

    def _test_selected_mic(self) -> None:
        """Test the selected microphone with the same parameters the assistant will use."""
        import sounddevice as sd

        device_idx = self._get_selected_device_index()
        if device_idx is None:
            self._log_to_gui("ERROR: No device selected")
            return

        self._log_to_gui(f"Testing device {device_idx}...")

        # Get device info
        try:
            from utils.audio_device import enumerate_input_devices
            devices = enumerate_input_devices()
            device_obj = next((d for d in devices if d.index == device_idx), None)
            if device_obj:
                device_name = device_obj.name
                device_hostapi = device_obj.hostapi_name
                self._log_to_gui(f"  Device: {device_name} [{device_hostapi}]")
            else:
                device_name = f"device_{device_idx}"
                device_hostapi = "unknown"
        except Exception as exc:
            device_name = f"device_{device_idx}"
            device_hostapi = "unknown"
            self._log_to_gui(f"  Warning: Could not get device details: {exc}")

        # Try to open stream with same parameters as assistant
        sample_rate = 16000
        test_duration = 0.25  # 250ms

        # Try multiple configurations
        test_configs = [
            (sample_rate, None, None),
            (sample_rate, 1024, None),
            (sample_rate, None, "high"),
            (44100, None, None),
            (48000, None, None),
        ]

        for sr, blocksize, latency in test_configs:
            try:
                self._log_to_gui(f"  Testing: sr={sr}, blocksize={blocksize}, latency={latency or 'default'}...")

                stream_kwargs = {
                    "channels": 1,
                    "samplerate": sr,
                    "device": device_idx,
                    "dtype": "float32",
                }

                if blocksize is not None:
                    stream_kwargs["blocksize"] = blocksize

                if latency is not None:
                    stream_kwargs["latency"] = latency

                # Record a short clip
                frames = int(sr * test_duration)
                audio = sd.rec(frames, **stream_kwargs)
                sd.wait()

                # Ensure device is fully released
                sd.stop()
                import time
                time.sleep(0.5)  # Give Windows time to release device

                self._log_to_gui(f"  ✓ PASS: Device {device_idx} works at {sr} Hz")
                self._log_to_gui(f"  Recommendation: Use this device for the assistant")
                return

            except Exception as exc:
                self._log_to_gui(f"  ✗ Failed at sr={sr}: {type(exc).__name__}: {exc}")
                continue

        # All tests failed
        self._log_to_gui(f"ERROR: Device {device_idx} failed all test configurations")
        if "DirectSound" in device_hostapi:
            self._log_to_gui(f"  💡 Tip: Look for a WASAPI version of this device (more reliable than DirectSound)")
        self._log_to_gui(f"  Try selecting a different device from the dropdown")

    def start_assistant(self) -> None:
        """Start the voice assistant with comprehensive error handling."""
        if self.running:
            self._log_to_gui("Already running.")
            return

        try:
            # Save audio device selection before starting
            self._save_audio_device()

            self._log_to_gui("Initializing assistant...")
            # ALWAYS create a new assistant instance to pick up new device selection
            self.assistant = AsyncRexAssistant(self.config)

            # Update wake word display with the ACTUAL keyword being used (may differ from config due to fallback)
            actual_wake_keyword = self.assistant._wake_keyword
            self.wake_var.set(f"Wake word: \"{actual_wake_keyword}\" (threshold: {self.config.wakeword_threshold})")

            self.running = True
            self.status_var.set("Starting…")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self._log_to_gui("Starting audio pipeline...")

            self.thread = threading.Thread(target=self._run_assistant, daemon=True)
            self.thread.start()
            self._log_to_gui("Assistant thread started.")
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
