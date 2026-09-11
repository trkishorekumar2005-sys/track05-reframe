"""CLI interface for reframe per CONTRACT §16."""

import functools
import json
from pathlib import Path
import sys
import traceback
from typing import Optional
import typer

from reframe.analyze import analyze as run_analyze
from reframe.config import load_config
from reframe.errors import ProcessingError, ReframeError
from reframe.log import setup_logging
from reframe.probe import probe


def handle_cli_error(e: Exception, verbose: bool = False) -> None:
    """Central error handler per CONTRACT §16."""
    if isinstance(e, ReframeError):
        sys.stderr.write(f"ERROR [{e.code}] {e.message}\n")
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(e.exit_code)
    else:
        sys.stderr.write(f"ERROR [unexpected_error] {e}\n")
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def cli_command(fn):
    """Decorator to catch exceptions and route to central error handler."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            verbose = kwargs.get("verbose", False) or ("--verbose" in sys.argv)
            handle_cli_error(e, verbose=verbose)

    return wrapper


class ReframeTyper(typer.Typer):
    """Custom Typer app with central error interception."""

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            verbose = "--verbose" in sys.argv
            handle_cli_error(e, verbose=verbose)


app = ReframeTyper(
    help="reframe: Offline, CPU-only intelligent video reframing.",
    no_args_is_help=True,
)


@app.command("validate", help="Validate an input video file.")
@cli_command
def validate(
    input: Path = typer.Argument(..., help="Path to input video file."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    cfg = load_config()
    info = probe(input, cfg)
    print(json.dumps(info.model_dump(), indent=2))
    if info.warnings:
        for w in info.warnings:
            sys.stderr.write(f"WARNING: {w}\n")


@app.command("analyze", help="Analyze video and generate decision timeline.")
@cli_command
def analyze(
    input: Path = typer.Argument(..., help="Path to input video file."),
    aspect: str = typer.Option(..., "--aspect", help="Target aspect ratio (e.g. 9:16 or 1:1)."),
    out: Path = typer.Option(..., "--out", help="Path to run output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to custom config YAML."),
    set_: Optional[list[str]] = typer.Option(None, "--set", help="Config override key=val."),
    start: Optional[float] = typer.Option(None, "--start", help="Start time in seconds."),
    end: Optional[float] = typer.Option(None, "--end", help="End time in seconds."),
    resume: bool = typer.Option(False, "--resume", help="Resume from stage cache if present."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    setup_logging(out, verbose)
    cfg = load_config(config, set_)
    timeline_path = run_analyze(input, aspect, out, cfg, start_s=start, end_s=end, resume=resume)
    print(json.dumps({"decision_timeline": timeline_path.as_posix()}, indent=2))


@app.command("render", help="Render reframed video from timeline.")
@cli_command
def render(
    timeline: Path = typer.Argument(..., help="Path to decision_timeline.json."),
    out: Path = typer.Option(..., "--out", help="Path to output MP4."),
    debug_overlay: bool = typer.Option(False, "--debug-overlay", help="Include debug visualization overlay."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    raise ProcessingError(code="not_implemented")


@app.command("run", help="Run full pipeline: analyze, render, validate.")
@cli_command
def run(
    input: Path = typer.Argument(..., help="Path to input video file."),
    aspect: str = typer.Option(..., "--aspect", help="Target aspect ratio (9:16 or 1:1)."),
    out: Path = typer.Option(..., "--out", help="Path to run output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to custom config YAML."),
    set_: Optional[list[str]] = typer.Option(None, "--set", help="Config override key=val."),
    start: Optional[float] = typer.Option(None, "--start", help="Start time in seconds."),
    end: Optional[float] = typer.Option(None, "--end", help="End time in seconds."),
    resume: bool = typer.Option(False, "--resume", help="Resume from stage cache if present."),
    debug_overlay: bool = typer.Option(False, "--debug-overlay", help="Include debug visualization overlay."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    raise ProcessingError(code="not_implemented")


@app.command("validate-output", help="Validate run directory outputs.")
@cli_command
def validate_output(
    run_dir: Path = typer.Argument(..., help="Path to run directory."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    raise ProcessingError(code="not_implemented")


@app.command("eval", help="Evaluate runs against ground truth labels.")
@cli_command
def eval(
    runs: Path = typer.Option(..., "--runs", help="Directory containing run directories."),
    labels: Path = typer.Option(Path("labels"), "--labels", help="Directory containing ground truth labels."),
    out: Path = typer.Option(..., "--out", help="Output directory for evaluation results."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    raise ProcessingError(code="not_implemented")


@app.command("bench", help="Run benchmark suite.")
@cli_command
def bench(
    suite: Path = typer.Option(Path("configs/bench.yaml"), "--suite", help="Path to benchmark suite YAML."),
    out: Path = typer.Option(Path("evidence/bench"), "--out", help="Output directory for benchmark results."),
    quick: bool = typer.Option(False, "--quick", help="Quick mode (repeats 1, warmup 0)."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose error output."),
) -> None:
    raise ProcessingError(code="not_implemented")


__all__ = ["app", "handle_cli_error"]
