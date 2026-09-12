# Track 05 — Active-Speaker Reframing

## Setup

uv sync --frozen
uv run python scripts/download_models.py

## Run

uv run reframe run <input.mp4> --aspect 9:16 --out <run_dir>

## Test

uv run pytest -q

## Benchmark

uv run reframe bench --suite configs/bench.yaml --out evidence/bench

## Status

[Be honest here — see step 4]
