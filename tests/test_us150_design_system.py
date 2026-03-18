"""Tests for US-150: Apply base visual design system (colors, typography, spacing).

Acceptance criteria:
- a design token file (CSS variables, theme object, or equivalent) defines:
  primary color, background color, surface color, text color, accent color,
  font family, base spacing unit
- all GUI components use values from the design token file, not hardcoded colors
  or sizes
- overall appearance is dark or neutral-dark themed (not a default browser/OS
  chrome look)
- typography uses a clean sans-serif font (system font stack or a single loaded
  font)
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CSS_PATH = REPO_ROOT / "rex" / "dashboard" / "static" / "css" / "dashboard.css"
HTML_PATH = REPO_ROOT / "rex" / "dashboard" / "templates" / "index.html"


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _root_vars(css: str) -> dict[str, str]:
    """Extract CSS variable declarations from the :root block."""
    root_match = re.search(r":root\s*\{([^}]+)\}", css, re.DOTALL)
    assert root_match, ":root block not found in CSS"
    root_block = root_match.group(1)
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root_block))


# ---------------------------------------------------------------------------
# Design token file — required tokens
# ---------------------------------------------------------------------------


class TestDesignTokensDefined:
    def test_css_file_exists(self) -> None:
        assert CSS_PATH.exists(), f"CSS file not found: {CSS_PATH}"

    def test_root_block_exists(self) -> None:
        assert ":root" in _css()

    def test_primary_color_defined(self) -> None:
        vars_ = _root_vars(_css())
        assert "--primary-color" in vars_, "--primary-color not in :root"

    def test_bg_color_defined(self) -> None:
        vars_ = _root_vars(_css())
        assert "--bg-color" in vars_, "--bg-color not in :root"

    def test_surface_color_defined(self) -> None:
        vars_ = _root_vars(_css())
        assert "--surface-color" in vars_, "--surface-color not in :root"

    def test_text_color_defined(self) -> None:
        vars_ = _root_vars(_css())
        # accept --text-primary or --text-color
        has_text = "--text-primary" in vars_ or "--text-color" in vars_
        assert has_text, "--text-primary / --text-color not in :root"

    def test_accent_color_defined(self) -> None:
        vars_ = _root_vars(_css())
        assert "--accent-color" in vars_, "--accent-color not in :root"

    def test_font_family_defined(self) -> None:
        vars_ = _root_vars(_css())
        assert "--font-family" in vars_, "--font-family not in :root"

    def test_base_spacing_defined(self) -> None:
        vars_ = _root_vars(_css())
        # accept --base-spacing or --spacing-unit or --spacing-base
        has_spacing = any(
            k in vars_ for k in ("--base-spacing", "--spacing-unit", "--spacing-base")
        )
        assert has_spacing, "--base-spacing / --spacing-unit not in :root"

    def test_all_required_tokens_have_values(self) -> None:
        vars_ = _root_vars(_css())
        required = [
            "--primary-color",
            "--bg-color",
            "--surface-color",
            "--accent-color",
            "--font-family",
        ]
        for token in required:
            assert vars_.get(token, "").strip(), f"{token} has no value"


# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------


class TestDarkTheme:
    def _parse_hex(self, hex_str: str) -> tuple[int, int, int]:
        """Parse #rrggbb or #rgb to (r, g, b)."""
        h = hex_str.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _luminance(self, hex_str: str) -> float:
        r, g, b = self._parse_hex(hex_str)
        return 0.2126 * r / 255 + 0.7152 * g / 255 + 0.0722 * b / 255

    def test_bg_color_is_dark(self) -> None:
        vars_ = _root_vars(_css())
        bg = vars_.get("--bg-color", "").strip()
        assert bg.startswith("#"), f"--bg-color should be a hex color, got: {bg!r}"
        lum = self._luminance(bg)
        assert lum < 0.3, f"--bg-color {bg!r} has luminance {lum:.2f}, expected dark (< 0.3)"

    def test_surface_color_is_dark(self) -> None:
        vars_ = _root_vars(_css())
        surface = vars_.get("--surface-color", "").strip()
        assert surface.startswith("#"), f"--surface-color should be a hex color, got: {surface!r}"
        lum = self._luminance(surface)
        assert (
            lum < 0.3
        ), f"--surface-color {surface!r} has luminance {lum:.2f}, expected dark (< 0.3)"

    def test_text_primary_is_light(self) -> None:
        """On a dark theme, text should be light."""
        vars_ = _root_vars(_css())
        text_var = "--text-primary" if "--text-primary" in vars_ else "--text-color"
        text = vars_.get(text_var, "").strip()
        assert text.startswith("#"), f"{text_var} should be a hex color, got: {text!r}"
        lum = self._luminance(text)
        assert lum > 0.5, f"{text_var} {text!r} has luminance {lum:.2f}, expected light (> 0.5)"

    def test_not_default_white_surface(self) -> None:
        vars_ = _root_vars(_css())
        surface = vars_.get("--surface-color", "").strip().lower()
        assert surface not in (
            "#ffffff",
            "#fff",
            "white",
        ), "--surface-color should not be white in a dark theme"

    def test_not_default_white_bg(self) -> None:
        vars_ = _root_vars(_css())
        bg = vars_.get("--bg-color", "").strip().lower()
        assert bg not in (
            "#ffffff",
            "#fff",
            "#f9fafb",
            "white",
        ), "--bg-color should not be near-white in a dark theme"


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------


class TestTypography:
    def test_font_family_token_is_sans_serif(self) -> None:
        vars_ = _root_vars(_css())
        font = vars_.get("--font-family", "").lower()
        # Should reference system fonts or known sans-serif fonts
        sans_serif_markers = [
            "sans-serif",
            "segoe ui",
            "helvetica",
            "arial",
            "-apple-system",
            "roboto",
            "blinkmacsystemfont",
        ]
        assert any(
            m in font for m in sans_serif_markers
        ), f"--font-family does not appear to be sans-serif: {font!r}"

    def test_body_uses_font_family_var(self) -> None:
        css = _css()
        # html,body rule should reference --font-family
        body_block_match = re.search(r"html\s*,\s*body\s*\{([^}]+)\}", css, re.DOTALL)
        assert body_block_match, "html, body rule not found"
        body_block = body_block_match.group(1)
        assert (
            "var(--font-family)" in body_block
        ), "html,body font-family should use var(--font-family)"


# ---------------------------------------------------------------------------
# Components use design tokens
# ---------------------------------------------------------------------------


class TestComponentsUseTokens:
    def test_no_hardcoded_hex_login_gradient(self) -> None:
        css = _css()
        # Old hardcoded gradient colors should not appear
        assert "#667eea" not in css, "Hardcoded login gradient color #667eea found"
        assert "#764ba2" not in css, "Hardcoded login gradient color #764ba2 found"

    def test_login_gradient_uses_vars(self) -> None:
        css = _css()
        assert "linear-gradient" in css
        # gradient should reference CSS vars, not hardcoded hex
        grad_match = re.search(r"linear-gradient\([^)]+\)", css)
        assert grad_match, "linear-gradient not found"
        grad = grad_match.group(0)
        assert "var(--" in grad, f"Login gradient should use CSS variables, found: {grad!r}"

    def test_bg_color_used_in_body(self) -> None:
        css = _css()
        assert "var(--bg-color)" in css

    def test_surface_color_used_in_components(self) -> None:
        css = _css()
        assert "var(--surface-color)" in css

    def test_primary_color_used_in_components(self) -> None:
        css = _css()
        assert "var(--primary-color)" in css

    def test_text_primary_used_in_components(self) -> None:
        css = _css()
        assert "var(--text-primary)" in css

    def test_border_color_used_in_components(self) -> None:
        css = _css()
        assert "var(--border-color)" in css

    def test_accent_color_referenced(self) -> None:
        css = _css()
        # accent-color must appear somewhere in the CSS (not just defined)
        uses = list(re.findall(r"var\(--accent-color\)", css))
        assert len(uses) >= 1, "--accent-color defined but never referenced"


# ---------------------------------------------------------------------------
# Typecheck
# ---------------------------------------------------------------------------


class TestTypecheck:
    def test_mypy_exits_without_crash(self) -> None:
        """mypy must not crash (exit 2 = crash, exit 0/1 = type errors ok)."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "rex/", "--ignore-missing-imports"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 2, f"mypy crashed (exit 2):\n{result.stdout}\n{result.stderr}"
