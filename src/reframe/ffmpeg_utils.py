"""FFmpeg utilities and FrameReader implementation per CONTRACT §3, §4."""

from pathlib import Path
import shutil
import subprocess
from typing import Iterator, Optional, Union
import numpy as np

from reframe.errors import MissingDependencyError, ProcessingError


def require_tools() -> None:
    """Verify that ffmpeg and ffprobe are available on PATH per CONTRACT §2, §5."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise MissingDependencyError(
            "ffmpeg and ffprobe must be installed and available on PATH",
            code="ffmpeg_missing",
        )


def run_ffmpeg(
    args: list[str],
    log_file: Union[Path, str, None] = None,
    timeout: Optional[float] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg subprocess command with arguments list and optional logging."""
    require_tools()
    cmd: list[str] = ["ffmpeg"] if not args or args[0] != "ffmpeg" else []
    cmd.extend(args)

    stderr_f = None
    try:
        if log_file is not None:
            p_log = Path(log_file)
            p_log.parent.mkdir(parents=True, exist_ok=True)
            stderr_f = open(p_log, "a", encoding="utf-8")
            stderr_target = stderr_f
        else:
            stderr_target = subprocess.DEVNULL

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_target,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise ProcessingError(
                f"ffmpeg command failed with exit code {result.returncode}",
                code="ffmpeg_error",
            )
        return result
    finally:
        if stderr_f is not None:
            stderr_f.close()


class FrameReader:
    """Decode video frames via ffmpeg pipe per CONTRACT §4.

    Yields (index, t, frame) where frame is a (out_h, out_w, 3) uint8 BGR array.
    """

    def __init__(
        self,
        path: Union[Path, str],
        out_w: int,
        out_h: int,
        fps: float,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        log_file: Union[Path, str, None] = None,
        mode: str = "analysis",
    ):
        self.path = Path(path)
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.fps = float(fps)
        self.start_s = float(start_s) if start_s is not None else None
        self.end_s = float(end_s) if end_s is not None else None
        self.log_file = Path(log_file) if log_file is not None else None
        self.mode = mode

        if self.mode not in ("analysis", "render"):
            raise ValueError(f"Unknown mode '{self.mode}', expected 'analysis' or 'render'")

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_f = None

    def _start(self) -> None:
        require_tools()
        if self._proc is not None:
            return

        cmd: list[str] = ["ffmpeg", "-v", "error"]

        # Omit -ss/-t for whole-clip runs
        if self.start_s is not None and self.end_s is not None:
            dur = self.end_s - self.start_s
            cmd.extend(["-ss", f"{self.start_s:.3f}", "-t", f"{dur:.3f}"])
        elif self.start_s is not None:
            cmd.extend(["-ss", f"{self.start_s:.3f}"])
        elif self.end_s is not None:
            cmd.extend(["-t", f"{self.end_s:.3f}"])

        cmd.extend(["-i", self.path.as_posix()])

        if self.mode == "analysis":
            cmd.extend(["-vf", f"fps={self.fps},scale={self.out_w}:{self.out_h}"])
        else:  # render
            cmd.extend(["-fps_mode", "cfr", "-r", str(self.fps)])

        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])

        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_f = open(self.log_file, "a", encoding="utf-8")
            stderr_target = self._stderr_f
        else:
            stderr_target = subprocess.DEVNULL

        frame_bytes = self.out_w * self.out_h * 3
        bufsize = frame_bytes * 10
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            bufsize=bufsize,
        )

    def close(self) -> None:
        """Terminate the ffmpeg process and close open resources."""
        if self._proc is not None:
            try:
                if self._proc.stdout:
                    self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
                except Exception:
                    pass
            self._proc = None

        if self._stderr_f is not None:
            try:
                self._stderr_f.close()
            except Exception:
                pass
            self._stderr_f = None

    def __enter__(self) -> "FrameReader":
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if self._proc is None:
            self._start()

        frame_size = self.out_w * self.out_h * 3
        index = 0
        base_t = self.start_s if self.start_s is not None else 0.0

        try:
            while True:
                assert self._proc is not None and self._proc.stdout is not None
                raw_bytes = self._proc.stdout.read(frame_size)
                if not raw_bytes or len(raw_bytes) < frame_size:
                    break

                t = base_t + index / self.fps
                if self.end_s is not None and t >= self.end_s:
                    break

                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                    (self.out_h, self.out_w, 3)
                )
                yield (index, round(t, 3), frame)
                index += 1
        finally:
            self.close()


__all__ = ["require_tools", "run_ffmpeg", "FrameReader"]
