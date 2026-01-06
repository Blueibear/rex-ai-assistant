"""Tooltip and help dialog implementations for Tkinter."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Optional


class ToolTip:
    """Tooltip that appears on hover over a widget."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500):
        """
        Args:
            widget: Widget to attach tooltip to
            text: Text to display in tooltip
            delay: Delay in milliseconds before showing tooltip
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window: Optional[tk.Toplevel] = None
        self.after_id: Optional[str] = None

        # Bind events
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Button-1>", self.on_leave)  # Hide on click

    def on_enter(self, event=None):
        """Schedule tooltip to appear."""
        self.schedule_tooltip()

    def on_leave(self, event=None):
        """Cancel scheduled tooltip and hide if shown."""
        self.cancel_tooltip()
        self.hide_tooltip()

    def schedule_tooltip(self):
        """Schedule tooltip to appear after delay."""
        self.cancel_tooltip()
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def cancel_tooltip(self):
        """Cancel scheduled tooltip."""
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show_tooltip(self):
        """Display the tooltip."""
        if self.tooltip_window or not self.text:
            return

        # Get widget position
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Create label with text
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=300,
            padx=5,
            pady=3,
        )
        label.pack()

    def hide_tooltip(self):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class HelpIcon:
    """Help icon button that shows tooltip on hover and dialog on click."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        title: str = "Help",
        icon_text: str = "?",
    ):
        """
        Args:
            parent: Parent widget
            text: Help text to display
            title: Title for dialog box
            icon_text: Text for the icon button
        """
        self.text = text
        self.title = title

        # Create icon button
        self.button = tk.Label(
            parent,
            text=icon_text,
            font=("Segoe UI", 10, "bold"),
            foreground="#0066cc",
            cursor="hand2",
            width=2,
        )

        # Add tooltip
        self.tooltip = ToolTip(self.button, text, delay=500)

        # Bind click event
        self.button.bind("<Button-1>", self.on_click)

        # Hover effect
        self.button.bind("<Enter>", lambda e: self.button.config(foreground="#0052a3"))
        self.button.bind("<Leave>", lambda e: self.button.config(foreground="#0066cc"))

    def on_click(self, event=None):
        """Show help dialog on click."""
        messagebox.showinfo(self.title, self.text, parent=self.button)

    def grid(self, **kwargs):
        """Grid the button."""
        self.button.grid(**kwargs)

    def pack(self, **kwargs):
        """Pack the button."""
        self.button.pack(**kwargs)
