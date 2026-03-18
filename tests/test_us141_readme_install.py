"""US-141: Remove or archive legacy install instructions from the main flow.

Tests verify:
- README references only the install scripts as primary install method
- Legacy steps (manual pip venv commands, multiple extras choices) moved to advanced-install.md
- docs/advanced-install.md exists and is linked from README
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"
ADVANCED_INSTALL = REPO_ROOT / "docs" / "advanced-install.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def advanced_text() -> str:
    return ADVANCED_INSTALL.read_text(encoding="utf-8")


class TestReadmePrimaryInstallMethod:
    def test_readme_references_install_sh(self, readme_text: str) -> None:
        assert "install.sh" in readme_text, "README must reference install.sh"

    def test_readme_references_install_ps1(self, readme_text: str) -> None:
        assert "install.ps1" in readme_text, "README must reference install.ps1"

    def test_readme_links_to_advanced_install(self, readme_text: str) -> None:
        assert "advanced-install.md" in readme_text, "README must link to docs/advanced-install.md"

    def test_readme_advanced_link_has_correct_label(self, readme_text: str) -> None:
        lower = readme_text.lower()
        assert (
            "advanced" in lower and "install" in lower
        ), "README must mention 'Advanced' and 'Install' near the advanced-install.md link"


class TestReadmeNoLegacySteps:
    """Legacy multi-step manual install commands must NOT appear in the Quickstart section."""

    def _quickstart_section(self, readme_text: str) -> str:
        """Extract text between ## Quickstart and the next ## heading."""
        lines = readme_text.splitlines()
        in_section = False
        section_lines: list[str] = []
        for line in lines:
            if line.startswith("## Quickstart"):
                in_section = True
                continue
            if in_section and line.startswith("## ") and not line.startswith("### "):
                break
            if in_section:
                section_lines.append(line)
        return "\n".join(section_lines)

    def test_readme_no_manual_venv_in_quickstart(self, readme_text: str) -> None:
        quickstart = self._quickstart_section(readme_text)
        assert (
            "python3 -m venv .venv" not in quickstart
        ), "Quickstart must not contain manual venv creation — use install scripts"

    def test_readme_no_requirements_cpu_in_quickstart(self, readme_text: str) -> None:
        quickstart = self._quickstart_section(readme_text)
        assert (
            "requirements-cpu.txt" not in quickstart
        ), "Quickstart must not reference requirements-cpu.txt — use install scripts"

    def test_readme_no_pip_install_ml_audio_extras(self, readme_text: str) -> None:
        quickstart = self._quickstart_section(readme_text)
        assert (
            '".[ml,audio]"' not in quickstart
        ), "Quickstart must not list individual ml,audio extras — move to advanced-install.md"

    def test_readme_no_interactive_installer_in_quickstart(self, readme_text: str) -> None:
        quickstart = self._quickstart_section(readme_text)
        assert (
            "python install.py --with-ml" not in quickstart
        ), "Quickstart must not contain interactive installer options — move to advanced-install.md"

    def test_readme_no_cuda_uninstall_in_quickstart(self, readme_text: str) -> None:
        quickstart = self._quickstart_section(readme_text)
        assert (
            "pip uninstall -y torch" not in quickstart
        ), "Quickstart must not contain CUDA uninstall commands — move to advanced-install.md"


class TestAdvancedInstallDocExists:
    def test_advanced_install_file_exists(self) -> None:
        assert ADVANCED_INSTALL.exists(), "docs/advanced-install.md must exist"

    def test_advanced_install_is_non_empty(self) -> None:
        assert (
            ADVANCED_INSTALL.stat().st_size > 500
        ), "docs/advanced-install.md must have substantive content"


class TestAdvancedInstallContainsLegacyContent:
    """Legacy content must be preserved in docs/advanced-install.md."""

    def test_advanced_install_has_manual_venv(self, advanced_text: str) -> None:
        assert (
            "venv" in advanced_text
        ), "docs/advanced-install.md must contain manual venv instructions"

    def test_advanced_install_has_gpu_section(self, advanced_text: str) -> None:
        lower = advanced_text.lower()
        assert (
            "gpu" in lower or "cuda" in lower
        ), "docs/advanced-install.md must contain GPU/CUDA instructions"

    def test_advanced_install_has_pip_extras(self, advanced_text: str) -> None:
        assert (
            "pip install" in advanced_text
        ), "docs/advanced-install.md must contain pip install instructions"

    def test_advanced_install_has_requirements_cpu(self, advanced_text: str) -> None:
        assert (
            "requirements-cpu.txt" in advanced_text
        ), "docs/advanced-install.md must document CPU requirements file"

    def test_advanced_install_has_dev_section(self, advanced_text: str) -> None:
        lower = advanced_text.lower()
        assert (
            "dev" in lower and "pytest" in lower
        ), "docs/advanced-install.md must include development setup instructions"
