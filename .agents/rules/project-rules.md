docs/CONTRACT.md is the specification. Follow it exactly; if impossible, stop and ask.

Implement ONLY the phase I request. Show a short plan first. Do not touch files outside the phase.

Runtime code in src/ is fully offline: no network calls, no hosted AI APIs. Only scripts/download_models.py downloads files.

Never invent numbers, metrics, benchmark results, licences or test outcomes anywhere.

src/reframe/ (except src/reframe/eval/) must never read labels/.

No hardcoded paths or thresholds; everything via configs/default.yaml and the Config model.

subprocess: argument lists, never shell=True; stderr to log files; timeouts.

MediaPipe Tasks API only (mediapipe.tasks.python.vision); mp.solutions does not exist in mediapipe 1.x. Never install opencv-python.

Decoding only through FrameReader (ffmpeg pipe), never cv2.VideoCapture.

Every change has tests. Finish every task by running uv run pytest -q and showing the full output.

Never weaken, skip or delete a failing test to make it pass; explain the failure.

Do not add dependencies without asking.

End each task with: files changed, assumptions, anything unverified, and exact commands for me to check.
