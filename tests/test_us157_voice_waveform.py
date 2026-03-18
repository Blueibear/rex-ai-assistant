"""Tests for US-157: Visual waveform animation during active listening."""

from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "static"
TEMPLATE_DIR = Path(__file__).parent.parent / "rex" / "dashboard" / "templates"
CSS_FILE = STATIC_DIR / "css" / "dashboard.css"
JS_FILE = STATIC_DIR / "js" / "dashboard.js"
HTML_FILE = TEMPLATE_DIR / "index.html"


def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def _js() -> str:
    return JS_FILE.read_text(encoding="utf-8")


def _html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


# ── HTML structure ────────────────────────────────────────────────────────────


class TestHTMLWaveform:
    def test_waveform_element_exists(self):
        assert 'id="voice-waveform"' in _html()

    def test_waveform_has_hidden_class_by_default(self):
        html = _html()
        idx = html.index('id="voice-waveform"')
        tag_end = html.index(">", idx)
        tag = html[idx:tag_end]
        assert "hidden" in tag

    def test_waveform_contains_bars(self):
        assert "waveform-bar" in _html()

    def test_waveform_has_five_bars(self):
        assert _html().count('class="waveform-bar"') == 5

    def test_waveform_inside_voice_mode_panel(self):
        html = _html()
        panel_start = html.index('class="voice-mode-panel"')
        # voice-waveform must appear after voice-mode-panel opening
        assert html.index('id="voice-waveform"') > panel_start

    def test_waveform_aria_hidden(self):
        html = _html()
        idx = html.index('id="voice-waveform"')
        tag_end = html.index(">", idx)
        tag = html[idx:tag_end]
        assert 'aria-hidden="true"' in tag


# ── CSS animation rules ───────────────────────────────────────────────────────


class TestCSSAnimation:
    def test_waveform_container_class_defined(self):
        assert ".voice-waveform {" in _css() or ".voice-waveform{" in _css()

    def test_waveform_hidden_rule_defined(self):
        assert ".voice-waveform.hidden" in _css()

    def test_waveform_bar_class_defined(self):
        assert ".waveform-bar" in _css()

    def test_keyframes_defined(self):
        assert "@keyframes waveform-bounce" in _css()

    def test_keyframes_has_transform_scaleY(self):
        css = _css()
        kf_start = css.index("@keyframes waveform-bounce")
        kf_end = css.index("}", kf_start) + 1
        # find the closing brace of the full @keyframes block
        depth = 0
        i = kf_start
        while i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    kf_end = i + 1
                    break
            i += 1
        kf_block = css[kf_start:kf_end]
        assert "scaleY" in kf_block

    def test_animation_applied_to_bar(self):
        css = _css()
        bar_idx = css.index(".waveform-bar {")
        bar_end = css.index("}", bar_idx)
        bar_block = css[bar_idx:bar_end]
        assert "animation" in bar_block

    def test_animation_uses_keyframe_name(self):
        css = _css()
        bar_idx = css.index(".waveform-bar {")
        bar_end = css.index("}", bar_idx)
        bar_block = css[bar_idx:bar_end]
        assert "waveform-bounce" in bar_block

    def test_animation_delays_stagger_bars(self):
        css = _css()
        assert "animation-delay" in css
        assert "nth-child" in css

    def test_hidden_rule_uses_display_none(self):
        css = _css()
        idx = css.index(".voice-waveform.hidden")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "display: none" in block or "display:none" in block

    def test_no_external_animation_library(self):
        """Ensure no import of Animate.css, GSAP, or similar external libraries."""
        css = _css()
        for lib in ["animate.css", "gsap", "motion.dev", "animejs"]:
            assert lib not in css.lower()

    def test_waveform_flex_layout(self):
        css = _css()
        idx = css.index(".voice-waveform {")
        block_end = css.index("}", idx)
        block = css[idx:block_end]
        assert "flex" in block


# ── JS logic ──────────────────────────────────────────────────────────────────


class TestJSWaveformToggle:
    def _get_update_fn_body(self) -> str:
        js = _js()
        fn_idx = js.index("function _updateVoiceModeUI")
        brace_start = js.index("{", fn_idx)
        depth = 0
        i = brace_start
        while i < len(js):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    return js[brace_start : i + 1]
            i += 1
        return js[brace_start:]

    def test_update_fn_references_voice_waveform(self):
        assert "voice-waveform" in self._get_update_fn_body()

    def test_update_fn_toggles_hidden_class(self):
        body = self._get_update_fn_body()
        assert "hidden" in body
        assert "toggle" in body

    def test_waveform_shown_only_for_listening(self):
        body = self._get_update_fn_body()
        # The condition should compare stateName to 'Listening'
        assert "Listening" in body

    def test_waveform_hidden_when_not_listening(self):
        """Toggle must pass true (hide) when state is not Listening."""
        body = self._get_update_fn_body()
        # pattern: toggle('hidden', stateName !== 'Listening')
        assert "!== 'Listening'" in body or '!=="Listening"' in body

    def test_update_fn_uses_dollar_selector_for_waveform(self):
        body = self._get_update_fn_body()
        assert "$(" in body
        assert "voice-waveform" in body
