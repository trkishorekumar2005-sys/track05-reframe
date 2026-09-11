"""Resource sampling per CONTRACT §13: PeakMemorySampler (RSS incl. children), Timer."""

from __future__ import annotations

import threading
import time
from typing import Optional

import psutil


class Timer:
    """Simple wall-clock stopwatch; use as a context manager or start()/stop()."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
            self._start = None
        return self._elapsed

    @property
    def elapsed_s(self) -> float:
        if self._start is not None:
            return time.perf_counter() - self._start
        return self._elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class PeakMemorySampler:
    """Tracks the peak RSS of a process plus all its (current and future) children.

    Samples every `interval_s` (default 50ms per CONTRACT §2) in a background
    thread, so it captures short-lived ffmpeg/child-process memory spikes that
    a single before/after measurement would miss.
    """

    def __init__(self, pid: Optional[int] = None, interval_s: float = 0.05) -> None:
        self._pid = pid if pid is not None else psutil.Process().pid
        self._interval_s = interval_s
        self._peak_bytes = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _sample_once(self) -> int:
        try:
            proc = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            return 0

        total = 0
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        try:
            children = proc.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return total

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample > self._peak_bytes:
                self._peak_bytes = sample
            self._stop_event.wait(self._interval_s)

    def start(self) -> "PeakMemorySampler":
        self._peak_bytes = self._sample_once()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int:
        """Stop sampling and return the peak RSS observed, in bytes."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # One last sample in case the peak occurred between the last tick and stop().
        sample = self._sample_once()
        if sample > self._peak_bytes:
            self._peak_bytes = sample
        return self._peak_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)

    def __enter__(self) -> "PeakMemorySampler":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


__all__ = ["Timer", "PeakMemorySampler"]
