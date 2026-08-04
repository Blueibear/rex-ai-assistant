"""US-037: skipped-test budget parser and command gate."""

from __future__ import annotations

import pytest

from scripts import check_skip_budget as checker


def test_parse_summary_with_skips_and_ansi_noise() -> None:
    output = (
        "tests/test_example.py::test_optional SKIPPED [100%]\n"
        "\x1b[32m= 8297 passed, 119 skipped, 282 warnings in 434.33s (0:07:14) =\x1b[0m\n"
    )
    assert checker.parse_pytest_summary(output) == {
        "passed": 8297,
        "skipped": 119,
        "warnings": 282,
    }
    assert checker.count_skipped_tests(output) == 119


def test_summary_without_skips_counts_zero() -> None:
    assert checker.count_skipped_tests("= 12 passed in 0.42s =\n") == 0


def test_final_pytest_summary_wins() -> None:
    output = "= 4 passed, 9 skipped in 1.00s =\n= 7 passed, 2 skipped in 0.50s =\n"
    assert checker.count_skipped_tests(output) == 2


def test_missing_summary_fails_closed() -> None:
    with pytest.raises(ValueError, match="summary was not found"):
        checker.count_skipped_tests("tests/test_example.py SKIPPED [100%]\n")


def test_main_passes_at_budget(tmp_path, capsys) -> None:
    report = tmp_path / "pytest.out"
    report.write_text("= 10 passed, 3 skipped in 0.20s =\n", encoding="utf-8")
    assert checker.main([str(report), "--budget", "3"]) == 0
    assert "skipped 3 tests; budget is 3" in capsys.readouterr().out


def test_main_fails_when_budget_is_exceeded(tmp_path, capsys) -> None:
    report = tmp_path / "pytest.out"
    report.write_text("= 10 passed, 4 skipped in 0.20s =\n", encoding="utf-8")
    assert checker.main([str(report), "--budget", "3"]) == 1
    output = capsys.readouterr().out
    assert "exceeding the budget of 3 by 1" in output
    assert "Do not raise the budget" in output


def test_main_rejects_unparseable_report(tmp_path, capsys) -> None:
    report = tmp_path / "pytest.out"
    report.write_text("collection interrupted\n", encoding="utf-8")
    assert checker.main([str(report)]) == 2
    assert "unable to parse" in capsys.readouterr().err
