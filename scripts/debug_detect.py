"""Debug tool: run face detection over INPUT and save annotated frames.

Not part of the offline runtime pipeline; a developer aid for eyeballing
detector output. Runs FrameReader in analysis mode + MediaPipeDetector over
the whole clip, then re-reads 6 evenly spaced timestamps in render mode (full
original resolution) and saves annotated PNGs with the detections found at
each timestamp during the analysis pass.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reframe.config import load_config
from reframe.detect import Detection, make_detector
from reframe.ffmpeg_utils import FrameReader
from reframe.probe import VideoInfo, probe


def _analysis_dims(width: int, height: int, analysis_width: int) -> tuple[int, int, float]:
    """Analysis frame size per CONTRACT §4: never upscale width; even height."""
    out_w = min(analysis_width, width)
    scale = out_w / width
    out_h = int(round(height * scale))
    if out_h % 2 != 0:
        out_h += 1
    return out_w, out_h, scale


def _read_original_frame(input_path: Path, info: VideoInfo, t: float, log_file: Path):
    """Read a single original-resolution frame at time t via FrameReader render mode."""
    dt = 1.0 / info.fps
    end = min(t + dt, info.duration_s)
    if end <= t:
        end = t + dt
    with FrameReader(
        input_path,
        out_w=info.width,
        out_h=info.height,
        fps=info.fps,
        start_s=t,
        end_s=end,
        log_file=log_file,
        mode="render",
    ) as reader:
        for _, _, frame in reader:
            return frame
    return None


def _save_annotated(frame_bgr, detections: list[Detection], t: float, out_path: Path) -> None:
    """Draw detection boxes + confidences over an original-resolution frame and save as PNG."""
    h, w = frame_bgr.shape[:2]
    rgb = frame_bgr[:, :, ::-1]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgb)
    ax.axis("off")
    for d in detections:
        x, y, bw, bh = d.bbox
        ax.add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor="lime", facecolor="none"))
        ax.text(
            x, max(0.0, y - 4), f"{d.conf:.2f}",
            color="lime", fontsize=9, bbox=dict(facecolor="black", alpha=0.5, pad=0),
        )
    ax.text(
        4, 14, f"t={t:.3f}s",
        color="yellow", fontsize=10, bbox=dict(facecolor="black", alpha=0.5, pad=1),
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug MediaPipe face detection on a video.")
    parser.add_argument("input", type=Path, help="Path to input video.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for annotated PNGs.")
    args = parser.parse_args()

    cfg = load_config()
    info = probe(args.input, cfg)

    out_w, out_h, scale = _analysis_dims(info.width, info.height, cfg.analysis.analysis_width)

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"

    detector = make_detector(cfg, scale)
    ticks: list[tuple[float, list[Detection]]] = []
    try:
        with FrameReader(
            args.input,
            out_w=out_w,
            out_h=out_h,
            fps=cfg.analysis.sample_fps,
            start_s=0.0,
            end_s=info.duration_s,
            log_file=log_dir / "ffmpeg_analysis.log",
            mode="analysis",
        ) as reader:
            for _, t, frame in reader:
                t_ms = int(round(t * 1000))
                detections = detector.detect(frame, t_ms)
                ticks.append((t, detections))
                boxes = ", ".join(
                    f"[{d.bbox[0]:.1f},{d.bbox[1]:.1f},{d.bbox[2]:.1f},{d.bbox[3]:.1f}]@{d.conf:.2f}"
                    for d in detections
                )
                print(f"t={t:.3f}s  n_faces={len(detections)}  {boxes}")
    finally:
        detector.close()

    if not ticks:
        print("No analysis frames decoded; nothing to save.", file=sys.stderr)
        sys.exit(1)

    n = len(ticks)
    n_samples = min(6, n)
    if n_samples > 1:
        sample_indices = sorted({round(i * (n - 1) / (n_samples - 1)) for i in range(n_samples)})
    else:
        sample_indices = [0]

    render_log = log_dir / "ffmpeg_render.log"
    for i, idx in enumerate(sample_indices):
        t, detections = ticks[idx]
        frame = _read_original_frame(args.input, info, t, render_log)
        if frame is None:
            print(f"WARNING: could not read original frame at t={t:.3f}s", file=sys.stderr)
            continue
        png_path = args.out / f"frame_{i:02d}_t{t:.3f}.png"
        _save_annotated(frame, detections, t, png_path)
        print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
