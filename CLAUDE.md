# Project rules (non-negotiable)

See docs/CONTRACT.md for the full specification — read it before implementing anything.

1. Implement ONLY the phase I request in each message. Show a short plan before writing code.
2. Runtime code in src/ is fully offline: no network calls, no hosted/commercial AI APIs.
3. Never invent numbers, metrics, benchmark results, licences or test outcomes anywhere.
4. src/reframe/ (except src/reframe/eval/) must never read the labels/ folder.
5. No hardcoded paths or thresholds; everything comes from configs/default.yaml and the Config model.
6. Use subprocess argument lists, never shell=True; stderr to log files; timeouts.
7. MediaPipe Tasks API only (mediapipe.tasks.python.vision); never use mp.solutions.
8. Never install opencv-python.
9. Decoding only through FrameReader (ffmpeg pipe), never cv2.VideoCapture.
10. Every change needs tests. Finish every task by running uv run pytest -q and showing the full output.
11. Never weaken, skip or delete a failing test to make it pass.
12. Do not add dependencies without asking.
13. End each task with:
    files changed,
    assumptions,
    anything unverified,
    exact commands for me to check.
