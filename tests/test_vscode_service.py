"""Tests for VS Code service module."""

import pytest
import tempfile
from pathlib import Path
from rex.vscode_service import (
    VSCodeService,
    get_vscode_service,
    reset_vscode_service,
    PatchResult,
    TestResult,
)


class TestVSCodeService:
    """Tests for VSCodeService class."""

    def test_service_init(self):
        """Test service initialization."""
        service = VSCodeService()
        assert service.workspace_path == Path(".")

    def test_open_file(self):
        """Test opening and reading a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test file
            test_file = tmppath / "test.txt"
            test_file.write_text("Hello, World!\nLine 2\nLine 3")

            service = VSCodeService(workspace_path=str(tmppath))
            result = service.open_file("test.txt")

            assert result["status"] == "success"
            assert result["lines"] == 3
            assert "Hello, World!" in result["content"]

    def test_open_nonexistent_file(self):
        """Test opening nonexistent file raises error."""
        service = VSCodeService()

        with pytest.raises(RuntimeError):
            service.open_file("nonexistent_file_12345.txt")

    def test_list_files(self):
        """Test listing files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "test1.txt").write_text("file1")
            (tmppath / "test2.txt").write_text("file2")
            (tmppath / "test.py").write_text("print('hello')")

            service = VSCodeService(workspace_path=str(tmppath))
            files = service.list_files(".")

            assert len(files) >= 2
            names = [f["name"] for f in files]
            assert "test1.txt" in names
            assert "test2.txt" in names

    def test_list_files_with_pattern(self):
        """Test listing files with glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "test1.txt").write_text("file1")
            (tmppath / "test2.txt").write_text("file2")
            (tmppath / "test.py").write_text("print('hello')")

            service = VSCodeService(workspace_path=str(tmppath))
            files = service.list_files(".", pattern="*.txt")

            assert len(files) == 2
            for f in files:
                assert f["name"].endswith(".txt")

    def test_generate_diff(self):
        """Test generating a unified diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create original file
            test_file = tmppath / "test.txt"
            original = "Line 1\nLine 2\nLine 3\n"
            test_file.write_text(original)

            # Generate diff with new content
            new_content = "Line 1\nLine 2 modified\nLine 3\n"
            service = VSCodeService(workspace_path=str(tmppath))
            diff = service.generate_diff("test.txt", new_content)

            assert "---" in diff
            assert "+++" in diff
            assert "-Line 2" in diff or "Line 2 modified" in diff

    def test_parse_pytest_output(self):
        """Test parsing pytest output."""
        service = VSCodeService()

        output = """
        test_example.py::test_one PASSED
        test_example.py::test_two FAILED

        5 passed, 2 failed in 1.23s
        """

        result = service._parse_pytest_output(output, returncode=1)

        assert result.passed == 5
        assert result.failed == 2
        assert result.duration_seconds == 1.23
        assert result.success is False


class TestVSCodeServiceSingleton:
    """Tests for VS Code service singleton."""

    def test_get_vscode_service(self):
        """Test getting VS Code service singleton."""
        reset_vscode_service()
        service1 = get_vscode_service()
        service2 = get_vscode_service()
        assert service1 is service2

    def test_reset_vscode_service(self):
        """Test resetting VS Code service."""
        service1 = get_vscode_service()
        reset_vscode_service()
        service2 = get_vscode_service()
        assert service1 is not service2
