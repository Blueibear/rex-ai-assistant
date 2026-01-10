"""Settings tab for Rex GUI - comprehensive environment variable editor."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Tuple

from logging_utils import get_logger
from utils.env_schema import EnvSchema, EnvVariable, is_restart_required, parse_env_example
from utils.env_writer import (
    create_backup,
    get_backup_files,
    get_extra_keys,
    read_current_env,
    restore_from_backup,
    write_env_from_template,
)
from utils.tooltips import HelpIcon

LOGGER = get_logger(__name__)


class SettingsTab(ttk.Frame):
    """Settings tab for editing all environment variables."""

    def __init__(self, parent, repo_root: Path):
        super().__init__(parent)
        self.repo_root = repo_root
        self.env_path = repo_root / ".env"
        self.template_path = repo_root / ".env.example"
        self.backup_dir = repo_root / "backups"

        # Load schema
        try:
            self.schema = parse_env_example(self.template_path)
        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                f".env.example not found at {self.template_path}",
                parent=self
            )
            self.schema = EnvSchema()

        # Load current values
        self.current_values = read_current_env(self.env_path)
        self.extra_keys = get_extra_keys(self.env_path, self.schema)

        # Track controls
        self.controls: Dict[str, tk.Widget] = {}
        self.secret_toggles: Dict[str, tk.Button] = {}
        self.restart_indicators: Dict[str, tk.Label] = {}
        self.needs_restart = set()

        # Create UI
        self._create_ui()

    def _create_ui(self):
        """Create the settings UI."""
        # Top button bar
        button_frame = ttk.Frame(self)
        button_frame.pack(side="top", fill="x", padx=10, pady=5)

        ttk.Button(button_frame, text="Save", command=self.save_settings, width=12).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_to_defaults, width=15).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Backup", command=self.create_backup_manual, width=12).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Restore", command=self.restore_backup, width=12).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Restart App", command=self.restart_application, width=12).pack(side="left", padx=2)

        # Platform-specific "Open in editor" button
        if sys.platform == "win32":
            ttk.Button(
                button_frame,
                text="Open .env in Notepad",
                command=self.open_in_notepad,
                width=20
            ).pack(side="right", padx=2)
        else:
            ttk.Button(
                button_frame,
                text="Open .env",
                command=self.open_in_editor,
                width=12
            ).pack(side="right", padx=2)

        # Restart required indicator
        self.restart_label = ttk.Label(
            button_frame,
            text="",
            foreground="orange",
            font=("Segoe UI", 9, "bold")
        )
        self.restart_label.pack(side="right", padx=10)

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)

        # Scrollable frame for settings
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Build settings controls
        self._build_settings_controls()

    def _build_settings_controls(self):
        """Build all settings controls organized by section."""
        row = 0

        # Track advanced settings separately
        advanced_vars = []

        # Add all sections from schema (non-advanced first)
        for section in self.schema.sections:
            # Collect non-advanced variables for this section
            regular_vars = [v for v in section.variables if not v.is_advanced]

            if regular_vars:
                # Section heading
                section_label = ttk.Label(
                    self.scrollable_frame,
                    text=section.name,
                    font=("Segoe UI", 11, "bold"),
                    foreground="#0066cc"
                )
                section_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(15, 5), padx=5)
                row += 1

                # Regular variables in section
                for var in regular_vars:
                    row = self._add_variable_control(var, row)

            # Collect advanced variables
            advanced_vars.extend([v for v in section.variables if v.is_advanced])

        # Add collapsible Advanced section
        if advanced_vars or self.extra_keys:
            row = self._add_advanced_section(row, advanced_vars)

        # Add Custom/Extra keys section
        if self.extra_keys or True:  # Always show option to add custom keys
            section_label = ttk.Label(
                self.scrollable_frame,
                text="Advanced / Custom Settings",
                font=("Segoe UI", 11, "bold"),
                foreground="#cc6600"
            )
            section_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(15, 5), padx=5)
            row += 1

            info_label = ttk.Label(
                self.scrollable_frame,
                text="Custom keys not in .env.example. Be careful when modifying.",
                font=("Segoe UI", 8),
                foreground="gray"
            )
            info_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 5), padx=5)
            row += 1

            for key, value in sorted(self.extra_keys.items()):
                custom_var = EnvVariable(
                    key=key,
                    default_value=value,
                    description=f"Custom setting: {key}",
                    section="Advanced"
                )
                row = self._add_variable_control(custom_var, row, is_custom=True)

            # Add new custom key button
            add_button = ttk.Button(
                self.scrollable_frame,
                text="+ Add Custom Key",
                command=self.add_custom_key
            )
            add_button.grid(row=row, column=0, columnspan=2, sticky="w", pady=10, padx=5)
            row += 1

    def _add_advanced_section(self, row: int, advanced_vars: list) -> int:
        """Add collapsible Advanced settings section.

        Args:
            row: Current grid row
            advanced_vars: List of advanced EnvVariable objects

        Returns:
            Next available row number
        """
        # Create collapsible frame
        collapse_frame = ttk.Frame(self.scrollable_frame)
        collapse_frame.grid(row=row, column=0, columnspan=5, sticky="ew", pady=(15, 5), padx=5)
        row += 1

        # Track collapsed state
        is_collapsed = tk.BooleanVar(value=True)

        # Toggle button
        toggle_btn = ttk.Button(
            collapse_frame,
            text="▶ Advanced Settings (click to expand)",
            command=lambda: self._toggle_advanced_section(is_collapsed, toggle_btn, advanced_frame)
        )
        toggle_btn.pack(anchor="w")

        # Warning label
        warning_label = ttk.Label(
            self.scrollable_frame,
            text="⚠ Most users should not change these. Incorrect values can cause instability.",
            foreground="orange",
            font=("Segoe UI", 8)
        )
        warning_label.grid(row=row, column=0, columnspan=5, sticky="w", pady=(0, 5), padx=20)
        warning_label.grid_remove()  # Hidden by default
        row += 1

        # Advanced settings container (hidden by default)
        advanced_frame = ttk.Frame(self.scrollable_frame)
        advanced_frame.grid(row=row, column=0, columnspan=5, sticky="ew", padx=20)
        advanced_frame.grid_remove()  # Hidden by default

        # Store references for toggle
        toggle_btn.warning_label = warning_label
        toggle_btn.advanced_frame = advanced_frame

        # Add advanced variables to the frame
        adv_row = 0
        for var in advanced_vars:
            adv_row = self._add_variable_control_to_frame(var, advanced_frame, adv_row)

        row += 1
        return row

    def _toggle_advanced_section(self, is_collapsed: tk.BooleanVar, toggle_btn, advanced_frame):
        """Toggle visibility of advanced section."""
        if is_collapsed.get():
            # Expand
            toggle_btn.config(text="▼ Advanced Settings (click to collapse)")
            advanced_frame.grid()
            if hasattr(toggle_btn, 'warning_label'):
                toggle_btn.warning_label.grid()
            is_collapsed.set(False)
        else:
            # Collapse
            toggle_btn.config(text="▶ Advanced Settings (click to expand)")
            advanced_frame.grid_remove()
            if hasattr(toggle_btn, 'warning_label'):
                toggle_btn.warning_label.grid_remove()
            is_collapsed.set(True)

    def _add_variable_control_to_frame(self, var: EnvVariable, parent_frame: ttk.Frame, row: int) -> int:
        """Add a variable control to a specific frame (used for Advanced section).

        This is similar to _add_variable_control but renders into a custom frame.
        """
        # Get current value (from .env or default)
        current_value = self.current_values.get(var.key, var.default_value)

        # Label with required indicator
        label_text = var.key
        if var.is_required:
            label_text += " "

        label_frame = ttk.Frame(parent_frame)
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=3)

        label = ttk.Label(label_frame, text=label_text)
        label.pack(side="left")

        if var.is_required:
            required_label = ttk.Label(
                label_frame,
                text="Required",
                foreground="red",
                font=("Segoe UI", 7, "bold")
            )
            required_label.pack(side="left", padx=(2, 0))

        # Control
        control = self._create_control(var, current_value)
        control.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        self.controls[var.key] = control

        # Configure column weight
        parent_frame.columnconfigure(1, weight=1)

        # Help icon
        tooltip_text = self._generate_tooltip_text(var)
        help_icon = HelpIcon(
            parent_frame,
            text=tooltip_text,
            title=f"Help: {var.key}"
        )
        help_icon.grid(row=row, column=2, padx=5)

        return row + 1

    def _generate_tooltip_text(self, var: EnvVariable) -> str:
        """Generate enhanced tooltip text for a variable.

        For numeric settings, includes range and effect information.
        """
        base_text = var.description or f"{var.key} setting"

        # Add range information for numeric settings
        if var.control_type in ['spinbox'] and (var.min_value is not None or var.max_value is not None):
            range_parts = []

            # Add range
            if var.min_value is not None and var.max_value is not None:
                unit_str = f" {var.units}" if var.units else ""
                range_parts.append(f"Range: {var.min_value}{unit_str} to {var.max_value}{unit_str}")
            elif var.min_value is not None:
                unit_str = f" {var.units}" if var.units else ""
                range_parts.append(f"Minimum: {var.min_value}{unit_str}")
            elif var.max_value is not None:
                unit_str = f" {var.units}" if var.units else ""
                range_parts.append(f"Maximum: {var.max_value}{unit_str}")

            # Add effect descriptions
            if var.tooltip_low_effect:
                range_parts.append(f"Low values: {var.tooltip_low_effect}")
            if var.tooltip_high_effect:
                range_parts.append(f"High values: {var.tooltip_high_effect}")

            if range_parts:
                return f"{base_text}\n\n" + "\n".join(range_parts)

        return base_text

    def _add_variable_control(self, var: EnvVariable, row: int, is_custom: bool = False) -> int:
        """Add a control for a single variable.

        Args:
            var: EnvVariable to create control for
            row: Current grid row
            is_custom: Whether this is a custom/advanced key

        Returns:
            Next available row number
        """
        # Get current value (from .env or default)
        current_value = self.current_values.get(var.key, var.default_value)

        # Label with required indicator
        label_text = var.key
        if var.is_required:
            label_text += " "

        label_frame = ttk.Frame(self.scrollable_frame)
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=3)

        label = ttk.Label(label_frame, text=label_text)
        label.pack(side="left")

        if var.is_required:
            required_label = ttk.Label(
                label_frame,
                text="Required",
                foreground="red",
                font=("Segoe UI", 7, "bold")
            )
            required_label.pack(side="left", padx=(2, 0))

        # Control with required highlighting
        control = self._create_control(var, current_value)

        # Add red highlight frame for required fields
        if var.is_required:
            highlight_frame = tk.Frame(
                self.scrollable_frame,
                highlightbackground="red",
                highlightthickness=2,
                highlightcolor="red"
            )
            highlight_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
            control.pack(in_=highlight_frame, fill="x", expand=True)
        else:
            control.grid(row=row, column=1, sticky="ew", padx=5, pady=3)

        self.controls[var.key] = control

        # Configure column weight for expansion
        self.scrollable_frame.columnconfigure(1, weight=1)

        # Help icon with enhanced tooltip for numeric ranges
        tooltip_text = self._generate_tooltip_text(var)
        help_icon = HelpIcon(
            self.scrollable_frame,
            text=tooltip_text,
            title=f"Help: {var.key}"
        )
        help_icon.grid(row=row, column=2, padx=5)

        # Restart indicator and active engine badge
        indicator_col = 3
        if is_restart_required(var.key):
            restart_label = ttk.Label(
                self.scrollable_frame,
                text="⚠",
                foreground="orange",
                font=("Segoe UI", 10)
            )
            restart_label.grid(row=row, column=indicator_col, padx=2)
            self.restart_indicators[var.key] = restart_label

            # Add tooltip
            from utils.tooltips import ToolTip
            ToolTip(restart_label, "Restart required after changing this setting")
            indicator_col += 1

        # Active engine indication
        if var.active_group and self._is_engine_active(var):
            active_label = ttk.Label(
                self.scrollable_frame,
                text="Active",
                foreground="green",
                font=("Segoe UI", 7, "bold"),
                background="#e8f5e9",
                padding=(4, 2)
            )
            active_label.grid(row=row, column=indicator_col, padx=2)
        elif var.active_group and not self._is_engine_selector(var):
            # Inactive engine settings get a grayed badge
            inactive_label = ttk.Label(
                self.scrollable_frame,
                text="Inactive",
                foreground="gray",
                font=("Segoe UI", 7),
                padding=(4, 2)
            )
            inactive_label.grid(row=row, column=indicator_col, padx=2)

        return row + 1

    def _is_engine_selector(self, var: EnvVariable) -> bool:
        """Check if this variable is an engine selector (not a configuration for an engine)."""
        return var.active_group and 'selector' in var.active_group

    def _is_engine_active(self, var: EnvVariable) -> bool:
        """Check if this engine/module is currently active."""
        if not var.active_group:
            return False

        # Get current provider/backend selections
        llm_provider = self.current_values.get('REX_LLM_PROVIDER', '').lower()
        tts_provider = self.current_values.get('REX_TTS_PROVIDER', '').lower()
        wakeword_backend = self.current_values.get('REX_WAKEWORD_BACKEND', '').lower()

        # Check if this setting's group matches active provider
        if var.active_group == 'openai':
            return llm_provider == 'openai'
        elif var.active_group == 'ollama':
            return llm_provider == 'ollama'
        elif var.active_group == 'transformers':
            return llm_provider == 'transformers'
        elif var.active_group == 'xtts':
            return tts_provider == 'xtts'
        elif var.active_group == 'edge':
            return tts_provider == 'edge'
        elif var.active_group == 'piper':
            return tts_provider == 'piper'
        elif var.active_group == 'pyttsx3':
            return tts_provider == 'pyttsx3'
        elif var.active_group == 'whisper':
            return True  # Whisper is always active for STT
        elif 'selector' in var.active_group:
            return True  # Selectors are always "active"

        return False

    def _create_control(self, var: EnvVariable, current_value: str) -> tk.Widget:
        """Create appropriate control widget for variable."""
        # Special handling for voice selection
        if 'voice' in var.key.lower() and 'tts' in var.section.lower():
            return self._create_voice_dropdown(var, current_value)

        if var.control_type == "checkbox":
            return self._create_checkbox(var, current_value)
        elif var.control_type == "dropdown":
            return self._create_dropdown(var, current_value)
        elif var.control_type == "spinbox":
            return self._create_spinbox(var, current_value)
        elif var.control_type == "path":
            return self._create_path_control(var, current_value)
        else:
            return self._create_entry(var, current_value)

    def _create_checkbox(self, var: EnvVariable, current_value: str) -> ttk.Checkbutton:
        """Create checkbox control."""
        var_obj = tk.BooleanVar(value=current_value.lower() in ("true", "1", "yes", "on"))

        check = ttk.Checkbutton(self.scrollable_frame, variable=var_obj)
        check.var = var_obj  # Store reference
        return check

    def _create_dropdown(self, var: EnvVariable, current_value: str) -> ttk.Combobox:
        """Create dropdown control."""
        dropdown = ttk.Combobox(
            self.scrollable_frame,
            values=var.dropdown_options or [],
            state="readonly",
            width=30
        )

        # Set current value if in options
        if current_value in (var.dropdown_options or []):
            dropdown.set(current_value)
        elif var.dropdown_options:
            dropdown.set(var.dropdown_options[0])

        # Special handling for Ollama model dropdown
        if 'LLM_MODEL' in var.key and self._is_ollama_selected():
            self._setup_ollama_refresh(dropdown)

        return dropdown

    def _create_spinbox(self, var: EnvVariable, current_value: str) -> ttk.Spinbox:
        """Create spinbox control."""
        try:
            value = float(current_value) if current_value else (var.min_value or 0)
        except ValueError:
            value = var.min_value or 0

        spinbox = ttk.Spinbox(
            self.scrollable_frame,
            from_=var.min_value or 0,
            to=var.max_value or 999999,
            increment=0.1 if var.min_value and var.min_value < 1 else 1,
            width=28
        )
        spinbox.set(value)
        return spinbox

    def _create_entry(self, var: EnvVariable, current_value: str) -> tk.Entry:
        """Create text entry control."""
        entry = ttk.Entry(self.scrollable_frame, width=30)
        entry.insert(0, current_value)

        # Handle secrets
        if var.is_secret:
            entry.config(show="*")

            # Add show/hide toggle
            def toggle_secret():
                if entry.cget("show") == "*":
                    entry.config(show="")
                    toggle_btn.config(text="Hide")
                else:
                    entry.config(show="*")
                    toggle_btn.config(text="Show")

            toggle_btn = ttk.Button(
                self.scrollable_frame,
                text="Show",
                command=toggle_secret,
                width=6
            )
            self.secret_toggles[var.key] = toggle_btn

        return entry

    def _create_voice_dropdown(self, var: EnvVariable, current_value: str) -> ttk.Frame:
        """Create voice selection dropdown with preview button."""
        frame = ttk.Frame(self.scrollable_frame)

        # Discover voices based on TTS provider
        tts_provider = self.current_values.get('REX_TTS_PROVIDER', 'xtts').lower()

        try:
            from utils.voice_discovery import discover_all_voices
            voices = discover_all_voices(provider=tts_provider)
            voice_options = [v.display_name() for v in voices]

            if not voice_options:
                voice_options = [current_value or "No voices found"]
        except Exception as e:
            LOGGER.warning(f"Failed to discover voices: {e}")
            voice_options = [current_value or "Error loading voices"]

        # Dropdown
        dropdown = ttk.Combobox(
            frame,
            values=voice_options,
            state="readonly" if voice_options else "normal",
            width=25
        )

        # Set current value if in options
        if current_value in voice_options:
            dropdown.set(current_value)
        elif voice_options and voice_options[0] != "No voices found":
            dropdown.set(voice_options[0])

        dropdown.pack(side="left", fill="x", expand=True)

        # Preview button
        preview_btn = ttk.Button(
            frame,
            text="Preview",
            command=lambda: self._preview_voice(dropdown.get(), tts_provider),
            width=8
        )
        preview_btn.pack(side="left", padx=(5, 0))

        # Disable preview if no valid voices
        if not voice_options or voice_options[0] in ["No voices found", "Error loading voices"]:
            preview_btn.config(state="disabled")

        # Store dropdown reference for value retrieval
        frame.dropdown = dropdown
        return frame

    def _create_path_control(self, var: EnvVariable, current_value: str) -> ttk.Frame:
        """Create path entry with browse button."""
        frame = ttk.Frame(self.scrollable_frame)

        entry = ttk.Entry(frame, width=25)
        entry.insert(0, current_value)
        entry.pack(side="left", fill="x", expand=True)

        def browse():
            if var.path_type == "directory":
                path = filedialog.askdirectory(parent=frame, initialdir=current_value or ".")
            else:
                path = filedialog.askopenfilename(parent=frame, initialdir=current_value or ".")

            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)

        browse_btn = ttk.Button(frame, text="Browse", command=browse, width=8)
        browse_btn.pack(side="left", padx=(5, 0))

        # Store entry reference for value retrieval
        frame.entry = entry
        return frame

    def _is_ollama_selected(self) -> bool:
        """Check if Ollama is the selected LLM provider."""
        for key in ['REX_LLM_PROVIDER', 'REX_LLM_BACKEND']:
            if key in self.controls:
                control = self.controls[key]
                if hasattr(control, 'get'):
                    value = control.get()
                    if value.lower() == 'ollama':
                        return True
        return False

    def _setup_ollama_refresh(self, dropdown: ttk.Combobox):
        """Add Ollama model refresh functionality."""
        # This would query Ollama API - implement if needed
        pass

    def _normalize_value(self, key: str, value: str) -> str:
        """Normalize value to preserve integer types.

        Converts float-formatted integers (e.g., "16000.0") back to integers ("16000").
        """
        if not value:
            return value

        # Try to detect if this should be an integer
        # Common integer config keys
        integer_keys = {
            'REX_SAMPLE_RATE', 'REX_LLM_MAX_TOKENS', 'REX_LLM_TOP_K', 'REX_LLM_SEED',
            'REX_MEMORY_MAX_TURNS', 'REX_MEMORY_MAX_BYTES', 'REX_SPEAK_RATE_LIMIT',
            'REX_SPEAK_RATE_WINDOW', 'REX_SPEAK_MAX_CHARS', 'REX_PLUGIN_TIMEOUT',
            'REX_PLUGIN_OUTPUT_LIMIT', 'REX_PLUGIN_RATE_LIMIT', 'REX_INPUT_DEVICE',
            'REX_OUTPUT_DEVICE', 'REX_AUDIO_INPUT_DEVICE', 'REX_AUDIO_OUTPUT_DEVICE'
        }

        if key in integer_keys:
            try:
                # Parse as float to handle "16000.0"
                float_val = float(value)
                # Check if it's a whole number
                if float_val.is_integer():
                    return str(int(float_val))
            except (ValueError, OverflowError):
                pass

        return value

    def _preview_voice(self, voice_display_name: str, provider: str):
        """Preview a TTS voice."""
        if not voice_display_name or voice_display_name in ["No voices found", "Error loading voices"]:
            messagebox.showwarning(
                "No Voice",
                "No voice selected or voice unavailable for preview.",
                parent=self
            )
            return

        try:
            from utils.voice_preview import generate_and_play_voice_sample

            # Extract voice ID from display name (format: "Name (lang) - provider")
            voice_id = voice_display_name.split(" - ")[0]
            if " (" in voice_id:
                voice_id = voice_id.split(" (")[0]

            # Play sample in background
            success = generate_and_play_voice_sample(voice_id, provider)

            if not success:
                messagebox.showinfo(
                    "Preview Unavailable",
                    f"Preview unavailable for this voice.\nProvider: {provider}",
                    parent=self
                )

        except Exception as e:
            LOGGER.exception("Voice preview failed")
            messagebox.showerror(
                "Preview Error",
                f"Failed to preview voice:\n{e}",
                parent=self
            )

    def _get_control_value(self, key: str) -> str:
        """Get current value from control."""
        if key not in self.controls:
            return ""

        control = self.controls[key]

        if isinstance(control, ttk.Checkbutton):
            value = "true" if control.var.get() else "false"
        elif isinstance(control, ttk.Frame) and hasattr(control, 'entry'):
            value = control.entry.get()
        elif isinstance(control, ttk.Frame) and hasattr(control, 'dropdown'):
            value = control.dropdown.get()
        elif hasattr(control, 'get'):
            value = str(control.get())
        else:
            value = ""

        # Normalize to preserve integer types
        return self._normalize_value(key, value)

    def save_settings(self):
        """Save all settings to .env file."""
        try:
            # Collect all values
            values = {}
            for key in self.controls.keys():
                values[key] = self._get_control_value(key)

            # Validate required fields
            missing_required = []
            for var in self.schema.get_all_variables():
                if var.is_required:
                    value = values.get(var.key, '').strip()
                    if not value:
                        missing_required.append(var.key)

            if missing_required:
                error_msg = "The following required fields are missing:\n\n"
                error_msg += "\n".join(f"  - {key}" for key in missing_required)
                error_msg += "\n\nPlease fill in all required fields before saving."

                messagebox.showerror(
                    "Required Fields Missing",
                    error_msg,
                    parent=self
                )
                return

            # Separate custom overrides
            schema_keys = {var.key for var in self.schema.get_all_variables()}
            standard_values = {k: v for k, v in values.items() if k in schema_keys}
            custom_overrides = {k: v for k, v in values.items() if k not in schema_keys}

            # Write .env
            backup_path = write_env_from_template(
                self.env_path,
                self.template_path,
                standard_values,
                custom_overrides,
                create_backup=True
            )

            # Reload environment
            try:
                from utils.env_loader import load
                load()

                # Try to reload config
                try:
                    from rex.config import reload_settings
                    reload_settings()
                except ImportError:
                    pass
            except Exception as e:
                LOGGER.warning(f"Failed to reload environment: {e}")

            # Check if restart needed
            changed_keys = set(values.keys())
            restart_keys = {k for k in changed_keys if is_restart_required(k)}

            if restart_keys:
                message = "Settings saved!\n\nRestart required for: " + ", ".join(sorted(restart_keys))
                self.restart_label.config(text="⚠ Restart required")
            else:
                message = "Settings saved successfully!"
                self.restart_label.config(text="")

            if backup_path and backup_path != self.env_path:
                message += f"\n\nBackup created: {backup_path.name}"

            messagebox.showinfo("Success", message, parent=self)

        except Exception as e:
            LOGGER.exception("Failed to save settings")
            messagebox.showerror("Error", f"Failed to save settings:\n{e}", parent=self)

    def reset_to_defaults(self):
        """Reset all settings to defaults from .env.example."""
        if not messagebox.askyesno(
            "Reset to Defaults",
            "This will reset ALL settings to defaults from .env.example.\n\nContinue?",
            parent=self
        ):
            return

        try:
            # Create backup first
            if self.env_path.exists():
                backup_path = create_backup(self.env_path, self.backup_dir)
                LOGGER.info(f"Created backup: {backup_path}")

            # Get defaults from schema
            defaults = {}
            for var in self.schema.get_all_variables():
                defaults[var.key] = var.default_value

            # Write with defaults
            write_env_from_template(
                self.env_path,
                self.template_path,
                defaults,
                create_backup=False  # Already created above
            )

            messagebox.showinfo(
                "Success",
                "Settings reset to defaults.\n\nPlease restart the application.",
                parent=self
            )

            # Reload the UI
            self._reload_ui()

        except Exception as e:
            LOGGER.exception("Failed to reset settings")
            messagebox.showerror("Error", f"Failed to reset settings:\n{e}", parent=self)

    def create_backup_manual(self):
        """Manually create a backup of current .env."""
        if not self.env_path.exists():
            messagebox.showwarning(
                "No .env File",
                ".env file does not exist yet. Nothing to backup.",
                parent=self
            )
            return

        try:
            backup_path = create_backup(self.env_path, self.backup_dir)
            messagebox.showinfo(
                "Backup Created",
                f"Backup saved to:\n{backup_path}",
                parent=self
            )
        except Exception as e:
            LOGGER.exception("Failed to create backup")
            messagebox.showerror("Error", f"Failed to create backup:\n{e}", parent=self)

    def restore_backup(self):
        """Restore .env from a backup file."""
        # Get list of backups
        backups = get_backup_files(self.backup_dir)

        if not backups:
            messagebox.showinfo(
                "No Backups",
                "No backup files found.",
                parent=self
            )
            return

        # Show selection dialog
        dialog = tk.Toplevel(self)
        dialog.title("Restore Backup")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Select backup to restore:", font=("Segoe UI", 10)).pack(pady=10)

        listbox = tk.Listbox(dialog, width=50, height=10)
        listbox.pack(padx=10, pady=10, fill="both", expand=True)

        for backup in backups:
            listbox.insert(tk.END, backup.name)

        def do_restore():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a backup to restore.", parent=dialog)
                return

            backup_path = backups[selection[0]]

            if messagebox.askyesno(
                "Confirm Restore",
                f"Restore from:\n{backup_path.name}\n\nThis will overwrite current settings. Continue?",
                parent=dialog
            ):
                try:
                    restore_from_backup(backup_path, self.env_path)
                    messagebox.showinfo(
                        "Success",
                        "Settings restored.\n\nPlease restart the application.",
                        parent=dialog
                    )
                    dialog.destroy()
                    self._reload_ui()
                except Exception as e:
                    LOGGER.exception("Failed to restore backup")
                    messagebox.showerror("Error", f"Failed to restore:\n{e}", parent=dialog)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Restore", command=do_restore).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

    def open_in_notepad(self):
        """Open .env file in Notepad (Windows)."""
        if not self.env_path.exists():
            if messagebox.askyesno(
                ".env Not Found",
                ".env file does not exist. Create it with defaults?",
                parent=self
            ):
                self.reset_to_defaults()
            else:
                return

        try:
            subprocess.Popen(["notepad.exe", str(self.env_path)])
        except Exception as e:
            LOGGER.exception("Failed to open Notepad")
            messagebox.showerror("Error", f"Failed to open Notepad:\n{e}", parent=self)

    def open_in_editor(self):
        """Open .env file in system default editor."""
        if not self.env_path.exists():
            if messagebox.askyesno(
                ".env Not Found",
                ".env file does not exist. Create it with defaults?",
                parent=self
            ):
                self.reset_to_defaults()
            else:
                return

        try:
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", str(self.env_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(self.env_path)])
        except Exception as e:
            LOGGER.exception("Failed to open editor")
            messagebox.showerror("Error", f"Failed to open editor:\n{e}", parent=self)

    def add_custom_key(self):
        """Add a new custom environment key."""
        dialog = tk.Toplevel(self)
        dialog.title("Add Custom Key")
        dialog.geometry("400x150")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Key:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        key_entry = ttk.Entry(dialog, width=30)
        key_entry.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(dialog, text="Value:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        value_entry = ttk.Entry(dialog, width=30)
        value_entry.grid(row=1, column=1, padx=10, pady=10)

        def add_key():
            key = key_entry.get().strip().upper()
            value = value_entry.get().strip()

            if not key:
                messagebox.showwarning("Invalid Key", "Key cannot be empty.", parent=dialog)
                return

            if not key.replace('_', '').isalnum():
                messagebox.showwarning("Invalid Key", "Key must contain only letters, numbers, and underscores.", parent=dialog)
                return

            # Add to extra keys
            self.extra_keys[key] = value

            dialog.destroy()
            self._reload_ui()

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Add", command=add_key).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

    def _reload_ui(self):
        """Reload the UI after changes."""
        # Clear existing controls
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Reload data
        self.current_values = read_current_env(self.env_path)
        self.extra_keys = get_extra_keys(self.env_path, self.schema)
        self.controls.clear()
        self.secret_toggles.clear()
        self.restart_indicators.clear()

        # Rebuild UI
        self._build_settings_controls()

    def restart_application(self):
        """Restart the application.

        Handles unsaved changes and attempts to restart automatically on Windows.
        """
        # Check for unsaved changes
        try:
            current_env = read_current_env(self.env_path)
            has_changes = False

            for key in self.controls.keys():
                current_ui_value = self._get_control_value(key)
                current_file_value = current_env.get(key, '')

                if current_ui_value != current_file_value:
                    has_changes = True
                    break

            if has_changes:
                response = messagebox.askyesnocancel(
                    "Unsaved Changes",
                    "You have unsaved changes.\n\n"
                    "Do you want to save before restarting?\n\n"
                    "Yes = Save and restart\n"
                    "No = Restart without saving\n"
                    "Cancel = Cancel restart",
                    parent=self
                )

                if response is None:  # Cancel
                    return
                elif response:  # Yes - save first
                    self.save_settings()

        except Exception as e:
            LOGGER.warning(f"Failed to check for unsaved changes: {e}")

        # Attempt restart
        try:
            if sys.platform == "win32":
                # Windows: try to restart via Python
                python = sys.executable
                script = str(self.repo_root / "run_gui.py")

                # Log the restart
                LOGGER.info(f"Restarting application: {python} {script}")

                # Close current window
                root = self.winfo_toplevel()

                # Launch new instance
                subprocess.Popen([python, script], cwd=str(self.repo_root))

                # Close this instance after a short delay
                root.after(500, root.destroy)

            else:
                # For other platforms, show manual restart instructions
                messagebox.showinfo(
                    "Restart Required",
                    "Please close and restart the application manually.\n\n"
                    "Run: python run_gui.py",
                    parent=self
                )

        except Exception as e:
            LOGGER.exception("Failed to restart application")
            messagebox.showerror(
                "Restart Failed",
                f"Failed to restart automatically:\n{e}\n\n"
                "Please close and restart the application manually.\n"
                "Run: python run_gui.py",
                parent=self
            )
