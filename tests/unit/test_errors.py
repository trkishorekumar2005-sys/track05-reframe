"""Unit tests for errors hierarchy and CLI exit codes."""

import subprocess
import sys
import pytest

from reframe.errors import (
    InvalidInputError,
    MissingDependencyError,
    OutputValidationError,
    ProcessingError,
    ReframeError,
)


def test_error_classes_exit_codes():
    """Each error class defines the exact exit code specified in CONTRACT §2, §16."""
    assert ReframeError.exit_code == 1
    assert InvalidInputError.exit_code == 2
    assert MissingDependencyError.exit_code == 3
    assert ProcessingError.exit_code == 4
    assert OutputValidationError.exit_code == 5

    # Test instantiation and formatting
    err1 = InvalidInputError("File does not exist", code="file_not_found")
    assert err1.exit_code == 2
    assert err1.code == "file_not_found"
    assert err1.message == "File does not exist"
    assert str(err1) == "ERROR [file_not_found] File does not exist"

    err2 = MissingDependencyError("ffmpeg not found", code="ffmpeg_missing")
    assert err2.exit_code == 3
    assert err2.code == "ffmpeg_missing"
    assert str(err2) == "ERROR [ffmpeg_missing] ffmpeg not found"

    err3 = ProcessingError(code="not_implemented")
    assert err3.exit_code == 4
    assert err3.code == "not_implemented"
    assert err3.message == "not implemented"
    assert str(err3) == "ERROR [not_implemented] not implemented"

    err4 = OutputValidationError("Validation failed", code="validation_failed")
    assert err4.exit_code == 5
    assert err4.code == "validation_failed"


def test_error_inheritance():
    """All reframe error classes inherit from ReframeError and Exception."""
    for cls in [InvalidInputError, MissingDependencyError, ProcessingError, OutputValidationError]:
        assert issubclass(cls, ReframeError)
        assert issubclass(cls, Exception)


def test_cli_help_exits_zero():
    """`python -m reframe --help` exits with code 0 and lists all 7 commands."""
    cmd = [sys.executable, "-m", "reframe", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    stdout = result.stdout
    for expected_cmd in [
        "validate",
        "analyze",
        "render",
        "run",
        "validate-output",
        "eval",
        "bench",
    ]:
        assert expected_cmd in stdout


def test_cli_command_stub_exit_code_and_output():
    """CLI command stubs raise ProcessingError(code='not_implemented') exiting with code 4."""
    cmd = [sys.executable, "-m", "reframe", "validate-output", "runs/dummy"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 4
    # Without --verbose, only ERROR [code] message is printed, no traceback
    assert "ERROR [not_implemented] not implemented" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_verbose_prints_traceback():
    """With --verbose, the error handler prints ERROR [code] message AND the traceback."""
    cmd = [sys.executable, "-m", "reframe", "validate-output", "runs/dummy", "--verbose"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 4
    assert "ERROR [not_implemented] not implemented" in result.stderr
    assert "Traceback (most recent call last):" in result.stderr
    assert "ProcessingError" in result.stderr


def test_cli_validate_error_exit_code():
    """`reframe validate` with missing file exits with code 2 and prints ERROR [file_not_found]."""
    cmd = [sys.executable, "-m", "reframe", "validate", "missing_clip.mp4"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "ERROR [file_not_found]" in result.stderr


@pytest.mark.parametrize(
    "cli_args",
    [
        ["analyze", "video.mp4", "--aspect", "9:16", "--out", "runs/test"],
        ["render", "runs/test/decision_timeline.json", "--out", "runs/test/output.mp4"],
        ["run", "video.mp4", "--aspect", "9:16", "--out", "runs/test"],
        ["validate-output", "runs/test"],
        ["eval", "--runs", "runs", "--out", "eval_out"],
        ["bench", "--suite", "configs/bench.yaml", "--out", "evidence/bench"],
    ],
)
def test_remaining_command_stubs(cli_args):
    """Remaining 6 pipeline commands are stubs raising ProcessingError(code='not_implemented')."""
    cmd = [sys.executable, "-m", "reframe"] + cli_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 4
    assert "ERROR [not_implemented] not implemented" in result.stderr
