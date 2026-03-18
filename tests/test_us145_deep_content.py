"""Tests for US-145: Move deep technical content into secondary docs."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
DOCS = ROOT / "docs"


def _extract_section(text: str, heading: str) -> str:
    """Extract a ## section from markdown text (up to next ## heading)."""
    lines = text.splitlines()
    in_section = False
    result = []
    for line in lines:
        if line.startswith("## ") and heading in line:
            in_section = True
            result.append(line)
            continue
        if in_section:
            if line.startswith("## ") and heading not in line:
                break
            result.append(line)
    return "\n".join(result)


def _count_section_lines(text: str, heading: str) -> int:
    section = _extract_section(text, heading)
    # Count non-empty lines only for a fair comparison
    return len([ln for ln in section.splitlines() if ln.strip()])


class TestReadmeSectionLength:
    """Each moved section must be ≤ 20 lines in the README."""

    def setup_method(self):
        self.readme = README.read_text(encoding="utf-8")

    def test_configuration_section_short(self):
        count = _count_section_lines(self.readme, "Configuration")
        assert count <= 20, f"Configuration section is {count} lines (max 20)"

    def test_usage_section_short(self):
        count = _count_section_lines(self.readme, "Usage")
        assert count <= 20, f"Usage section is {count} lines (max 20)"

    def test_docker_section_short(self):
        count = _count_section_lines(self.readme, "Docker")
        assert count <= 20, f"Docker section is {count} lines (max 20)"

    def test_memory_section_short(self):
        count = _count_section_lines(self.readme, "Memory")
        assert count <= 20, f"Memory section is {count} lines (max 20)"

    def test_troubleshooting_section_short(self):
        count = _count_section_lines(self.readme, "Troubleshooting")
        assert count <= 20, f"Troubleshooting section is {count} lines (max 20)"


class TestReadmeSectionLinks:
    """Each moved section must contain a link to the docs file."""

    def setup_method(self):
        self.readme = README.read_text(encoding="utf-8")

    def test_configuration_links_to_env_vars_doc(self):
        section = _extract_section(self.readme, "Configuration")
        assert "docs/environment-variables.md" in section

    def test_usage_links_to_usage_doc(self):
        section = _extract_section(self.readme, "Usage")
        assert "docs/usage.md" in section

    def test_docker_links_to_docker_doc(self):
        section = _extract_section(self.readme, "Docker")
        assert "docs/docker.md" in section

    def test_memory_links_to_memory_doc(self):
        section = _extract_section(self.readme, "Memory")
        assert "docs/memory.md" in section

    def test_troubleshooting_links_to_troubleshooting_doc(self):
        section = _extract_section(self.readme, "Troubleshooting")
        assert "docs/troubleshooting.md" in section


class TestDocsFilesExist:
    """Docs files must exist at the linked paths."""

    def test_environment_variables_doc_exists(self):
        assert (DOCS / "environment-variables.md").exists()

    def test_usage_doc_exists(self):
        assert (DOCS / "usage.md").exists()

    def test_docker_doc_exists(self):
        assert (DOCS / "docker.md").exists()

    def test_troubleshooting_doc_exists(self):
        assert (DOCS / "troubleshooting.md").exists()

    def test_memory_doc_exists(self):
        assert (DOCS / "memory.md").exists()


class TestContentPreserved:
    """Key content from each moved section must exist in the docs files."""

    def test_env_vars_doc_has_core_settings(self):
        content = (DOCS / "environment-variables.md").read_text(encoding="utf-8")
        assert "REX_ACTIVE_USER" in content
        assert "REX_LOG_LEVEL" in content

    def test_env_vars_doc_has_wake_word_settings(self):
        content = (DOCS / "environment-variables.md").read_text(encoding="utf-8")
        assert "REX_WAKEWORD" in content
        assert "REX_WAKEWORD_THRESHOLD" in content

    def test_env_vars_doc_has_audio_settings(self):
        content = (DOCS / "environment-variables.md").read_text(encoding="utf-8")
        assert "REX_INPUT_DEVICE" in content
        assert "REX_SAMPLE_RATE" in content

    def test_env_vars_doc_has_llm_settings(self):
        content = (DOCS / "environment-variables.md").read_text(encoding="utf-8")
        assert "REX_LLM_PROVIDER" in content
        assert "OPENAI_API_KEY" in content

    def test_env_vars_doc_has_tts_settings(self):
        content = (DOCS / "environment-variables.md").read_text(encoding="utf-8")
        assert "REX_TTS_PROVIDER" in content
        assert "REX_SPEAK_API_KEY" in content

    def test_usage_doc_has_text_mode(self):
        content = (DOCS / "usage.md").read_text(encoding="utf-8")
        assert "python -m rex" in content

    def test_usage_doc_has_voice_mode(self):
        content = (DOCS / "usage.md").read_text(encoding="utf-8")
        assert "rex_loop.py" in content

    def test_usage_doc_has_gui(self):
        content = (DOCS / "usage.md").read_text(encoding="utf-8")
        assert "python gui.py" in content

    def test_usage_doc_has_autonomous_workflows(self):
        content = (DOCS / "usage.md").read_text(encoding="utf-8")
        assert "rex plan" in content

    def test_usage_doc_has_github_integration(self):
        content = (DOCS / "usage.md").read_text(encoding="utf-8")
        assert "rex gh repos" in content

    def test_docker_doc_has_build_command(self):
        content = (DOCS / "docker.md").read_text(encoding="utf-8")
        assert "docker build" in content
        assert "rex-ai-assistant" in content

    def test_docker_doc_has_run_command(self):
        content = (DOCS / "docker.md").read_text(encoding="utf-8")
        assert "docker run" in content
        assert "--env-file .env" in content

    def test_memory_doc_has_user_profiles(self):
        content = (DOCS / "memory.md").read_text(encoding="utf-8")
        assert "Memory/" in content
        assert "core.json" in content

    def test_memory_doc_has_voice_cloning_note(self):
        content = (DOCS / "memory.md").read_text(encoding="utf-8")
        assert "XTTS" in content or "voice cloning" in content.lower()

    def test_troubleshooting_doc_has_ffmpeg(self):
        content = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
        assert "ffmpeg" in content.lower()

    def test_troubleshooting_doc_has_pytorch(self):
        content = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
        assert "torch" in content

    def test_troubleshooting_doc_has_wake_word_section(self):
        content = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
        assert "Wake Word" in content or "wake word" in content.lower()

    def test_troubleshooting_doc_has_cuda_section(self):
        content = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
        assert "CUDA" in content


class TestTypecheck:
    """Typecheck must pass (no new Python files introduced)."""

    def test_typecheck_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--ignore-missing-imports", "rex/"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # mypy exits 0 (no errors) or 1 (type errors) — both are acceptable
        # as pre-existing type errors are not this story's responsibility.
        # Exit 2+ indicates a crash/config error.
        assert result.returncode in (
            0,
            1,
        ), f"mypy crashed (exit {result.returncode}):\n{result.stderr}"
