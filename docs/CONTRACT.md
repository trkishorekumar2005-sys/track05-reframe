\# CONTRACT.md — implementation contract for `reframe`

Single source of truth. All code must follow it. If something is impossible or contradictory, STOP and ask; never improvise silently.



\## 1. Product contract

\- Input: local video (MP4) + target aspect `9:16` or `1:1`. Output: reframed MP4 with the ORIGINAL audio + `decision\_timeline.json` (face tracks, selected speaker, crop coordinates, confidence, switches, fallback periods, reasons).

\- CPU only, fully offline at runtime. No hosted APIs. No manual keyframes or manual speaker intervals as input.

\- Analysis uses downscaled, sampled frames. Rendering uses the ORIGINAL full-resolution video and reads ONLY the timeline + original input (never re-runs analysis).



\## 2. Package layout (src/reframe/)

\- `\_\_init\_\_.py` (`\_\_version\_\_ = "0.1.0"`), `\_\_main\_\_.py` (`python -m reframe` → `cli.app()`)

\- `cli.py`: typer `app` with `validate, analyze, render, run, validate-output, eval, bench` (§16)

\- `config.py`: pydantic v2 models (extra="forbid", sensible ge/le bounds on every number), `load\_config(path|None, overrides: list\[str]) -> Config`, `config\_hash(cfg) -> str` (sha256 of canonical sorted JSON, first 16 hex)

\- `errors.py`: `ReframeError(message, code)` with `exit\_code`; `InvalidInputError`=2, `MissingDependencyError`=3, `ProcessingError`=4, `OutputValidationError`=5

\- `log.py`: `setup\_logging(run\_dir|None, verbose)`; console human-readable; JSON lines to `run\_dir/logs/run.jsonl` with `ts, level, stage, event, msg` + extras

\- `ffmpeg\_utils.py`: `require\_tools()`, `run\_ffmpeg(args, log\_file, timeout)`, `FrameReader` (§4)

\- `probe.py`: `probe(path, cfg) -> VideoInfo` (§5)

\- `models.py`: `ensure\_model(name) -> Path` — file in `models/` must match sha256 in `models/manifest.lock.json`, else MissingDependencyError telling the user to run `uv run python scripts/download\_models.py`

\- `detect.py`: `Detection(bbox, conf)`, `FaceDetector` Protocol `detect(frame\_bgr, t\_ms) -> list\[Detection]`, `MediaPipeDetector`, `make\_detector(cfg, scale)`

\- `track.py`: `IoUTracker` (§6)

\- `audio.py`: `compute\_audio\_features(info, cfg, tick\_times, start\_s, end\_s, log\_dir) -> AudioFeatures(vad, energy, vad\_segments)`

\- `lips.py`: `MouthAnalyzer` (MediaPipe FaceLandmarker) + `window\_features(...)`

\- `scene.py`: `SceneAnalyzer` (per tick `is\_cut`, `is\_screen\_content`)

\- `score.py`: strategy registry `STRATEGIES` + `score\_ticks(...)` (§7)

\- `decide.py`: `decide(ticks, cfg, has\_audio) -> (decisions, switches)` (§8)

\- `crop.py`: `crop\_size, target\_rect, Smoother, clamp\_rect, check\_crop\_invariants, plan\_crops` (§9)

\- `timeline.py`: pydantic `Timeline` (§10), `save\_timeline`, `load\_timeline`

\- `analyze.py`: `analyze(input\_path, aspect, out\_dir, cfg, start\_s, end\_s, resume) -> Path`

\- `render.py`: `render(timeline\_path, out\_path, cfg, debug\_overlay) -> Path` (§11)

\- `validate.py`: `validate\_output(run\_dir) -> dict` (§12)

\- `cache.py`: `StageCache` (§13); `resources.py`: `PeakMemorySampler` (psutil every 50 ms, RSS of a process + all children), `Timer`

\- `machine.py`: `machine\_info() -> dict` (CPU name from Windows registry `HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\\ProcessorNameString` on Windows, cores, RAM, OS, Python, package versions, git commit)

\- `eval/labels.py`, `eval/metrics.py`, `eval/report.py` (§14); `bench.py` (§15)

Other paths: `configs/default.yaml`, `configs/bench.yaml`, `models/manifest.json` + `models/manifest.lock.json` (committed; model files gitignored), `scripts/`, `tests/unit|integration|e2e|fixtures`, `labels/` (ONLY `src/reframe/eval/` may read it), `data/` and `runs/` (gitignored), `evidence/` (committed results).



\## 3. Conventions

\- Times: seconds from the start of the original video (absolute); JSON rounded to 3 decimals.

\- Boxes `\[x, y, w, h]` in ORIGINAL display pixels (after rotation), JSON 1 decimal. Crop rects are ints.

\- Analysis ticks: `t\_k = start\_s + k / sample\_fps` while `t\_k < end\_s`. Output frames: `t\_f = start\_s + f / fps` (input average fps), `n\_frames = round((end\_s - start\_s) \* fps)`.

\- JSON paths use forward slashes (`Path.as\_posix()`).

\- Subprocess: argument lists only, never `shell=True`; stderr to a log file in `run\_dir/logs/` (never an unread PIPE); timeouts.

\- Decode ONLY via `FrameReader` (ffmpeg pipe). Never `cv2.VideoCapture` (inconsistent on Windows; ffmpeg handles rotation and VFR).

\- MediaPipe: Tasks API only (`mediapipe.tasks.python.vision`). `mp.solutions` does not exist in mediapipe 1.x.

\- Tests launching the CLI use `\[sys.executable, "-m", "reframe", ...]`.

\- Determinism: no randomness; numpy seeded from `runtime.seed`; two identical runs give identical timelines except `created\_utc` and `timing`.



\## 4. FrameReader(path, out\_w, out\_h, fps, start\_s, end\_s, log\_file, mode)

\- analysis: `ffmpeg -v error -ss {start} -t {dur} -i IN -vf fps={fps},scale={out\_w}:{out\_h} -f rawvideo -pix\_fmt bgr24 -`

\- render: `ffmpeg -v error -ss {start} -t {dur} -i IN -fps\_mode cfr -r {fps} -f rawvideo -pix\_fmt bgr24 -`

\- Omit `-ss/-t` for whole-clip runs. Reads exactly w\*h\*3 bytes per frame; yields `(index, t, frame)`; context manager terminates ffmpeg.

\- Analysis size: width = `analysis.analysis\_width` (never upscale), height = even number preserving aspect.



\## 5. VideoInfo (probe)

Fields: `path, sha256, size\_bytes, width, height, fps, is\_vfr, duration\_s, rotation, video\_codec, has\_audio, audio\_codec, n\_frames\_expected`. Use `ffprobe -v error -show\_streams -show\_format -of json`.

InvalidInputError codes: `file\_not\_found`, `empty\_file`, `not\_media`, `no\_video\_stream`, `bad\_duration`, `too\_long` (> runtime.max\_duration\_s). Missing ffmpeg/ffprobe → MissingDependencyError `ffmpeg\_missing`. No audio → OK + warning `no\_audio`. VFR (r\_frame\_rate != avg\_frame\_rate) → OK + warning `vfr\_normalised\_to\_cfr`. Swap width/height when rotation is ±90.



\## 6. Tracking

Hungarian matching (scipy) on IoU ≥ `tracking.iou\_match`; confirmed after `tracking.min\_hits` matches; ends after `tracking.max\_missed\_s` unmatched; a re-entering face gets a NEW id. Samples `{t, bbox, det\_conf, mouth\_open|null, score|null}`. Helper `box\_at(track, t)` interpolates linearly; None outside the track or inside a gap.



\## 7. Scoring (per tick, per visible track; score ∈ \[0,1] = "confidence")

\- `lip\_activity` = std of `mouth\_open` over the last `speaker.window\_s`; `lip\_norm = min(1, lip\_activity / speaker.lip\_activity\_norm)`

\- `av\_corr` = Pearson(mouth\_open window, energy window), 0 if either std < 1e-6

\- `largest\_face`: area / largest visible area (ignores audio; naive baseline for comparison)

\- `av\_heuristic`: `vad\[k] \* lip\_norm`; `av\_corr`: `vad\[k] \* max(0, av\_corr)`; `av\_fused` (optional): logistic model from `configs/calibration.json`



\## 8. Decision state machine (decide.py) — exact rules

States `FALLBACK` and `LOCKED(track\_id)`; start `FALLBACK` reason `startup`. Per tick (dt = 1/sample\_fps): `V` visible tracks, `s` scores, `speech = vad\[k] >= audio.speech\_threshold`. "for ≥ X s" = consecutive accumulated tick time, reset when the condition breaks.

1\. Scene cut at tick → FALLBACK `scene\_cut\_reset`; reset timers.

2\. `V` empty: if LOCKED and locked track missing ≤ `tracking.max\_missed\_s` → stay `face\_lost\_grace`; else FALLBACK `screen\_content` if scene says so, else `no\_face`.

3\. `best` = argmax s (tie → lower id).

4\. LOCKED(a): (a) a not visible > `tracking.max\_missed\_s` → FALLBACK `face\_lost` (≤ → stay `face\_lost\_grace`); (b) `t - locked\_since < speaker.min\_hold\_s` → stay `hold\_time\_active`; (c) speech and `s\[best] < speaker.fallback\_conf` for ≥ `speaker.fallback\_enter\_s` → FALLBACK `off\_screen\_voice`; (d) `best != a` and `s\[best] - s\[a] ≥ speaker.switch\_margin` for ≥ `speaker.switch\_confirm\_s` → LOCKED(best) `challenger\_margin\_sustained`; (e) not speech → stay `silence\_hold`; (f) else stay `audio\_visual\_match`.

5\. FALLBACK: (a) strategy `largest\_face` → LOCKED(best) `largest\_face`; (b) no audio track and len(V)==1 → LOCKED `single\_face\_no\_audio`; (c) `s\[best] ≥ speaker.enter\_conf` for ≥ `speaker.switch\_confirm\_s` → LOCKED(best) `audio\_visual\_match`; (d) else stay: `off\_screen\_voice` if speech else `low\_confidence`.

Decision per tick: `{t, mode: "speaker"|"fallback", track\_id|null, confidence, reason, scores: {track\_id: score}}`; confidence = selected track's score, or max visible score (0 if none) in fallback.

Switch record for EVERY change of framed target (track→track, track→FALLBACK, FALLBACK→track): `{t, from, to, reason, confidence}`, from/to = track id or "FALLBACK". Overlap: no special mode (incumbent protected by hold + margin).



\## 9. Crop planning (crop.py)

\- `crop\_size(W,H,aspect)`: r = aw/ah; if W/H ≥ r: h = even\_floor(H), w = even\_floor(h\*r); else w = even\_floor(W), h = even\_floor(w/r).

\- `target\_rect(face\_box, W, H, w, h, headroom)`: x = face\_cx − w/2; y = face\_cy − headroom\*h; clamp.

\- `plan\_crops(...)`: per output frame use the decision at tick `floor((t\_f − start\_s)\*sample\_fps)` (clamped). Speaker: `box\_at(track, t\_f)` → target. Fallback: full frame `(0,0,W,H)` mode `fallback`.

\- `Smoother` per axis: SNAP on mode change, track change or scene cut; else d = target − current; |d| < `crop.deadzone\_frac`·W → no move; step = `crop.smoothing\_alpha`·d capped at `crop.max\_speed\_frac\_s`·W/fps. Safety: afterwards, if the face centre is outside the central (1 − 2·`crop.safety\_margin\_frac`) of the crop, shift minimally so it is inside (overrides the speed cap).

\- Order: target → smooth → safety → `clamp\_rect` → ints → `check\_crop\_invariants`.

\- `check\_crop\_invariants(rect, W, H, aspect, mode, face\_box=None) -> list\[str]`: `out\_of\_frame`, `odd\_dims`, `wrong\_aspect` (|w/h − r| > 0.01, speaker mode), `face\_centre\_outside` (speaker mode with face\_box), `bad\_fallback\_rect`. Violations are counted and logged, never silently fixed.



\## 10. decision\_timeline.json — exact top-level keys

`schema\_version` ("1.0"), `tool` {name, version, git\_commit|null}, `created\_utc`, `input` (VideoInfo), `target` {aspect, crop\_w, crop\_h, fps}, `segment` {start\_s, end\_s}, `config\_hash`, `config`, `models` \[{name, file, sha256, size\_mb}], `analysis` {sample\_fps, analysis\_width, analysis\_height, detector, strategy, n\_ticks}, `face\_tracks` \[{track\_id, first\_t, last\_t, samples}], `vad\_segments` \[{start, end}], `scene\_cuts` \[t], `decisions`, `speaker\_segments` \[{start, end, mode, track\_id, mean\_confidence, reason}], `switches`, `fallback\_periods` \[{start, end, reason}], `crops` \[{f, t, x, y, w, h, mode}], `invariant\_violations` {count, by\_code, examples(≤20)}, `warnings` \[str], `timing` {analysis\_s}.



\## 11. Rendering

\- Output size = crop size (no upscaling). Speaker frame = `frame\[y:y+h, x:x+w]`. Fallback (`blur\_pad`): background = frame scaled to cover (w,h) + GaussianBlur 51; foreground = frame scaled to fit inside (w,h), centred.

\- Encode: `ffmpeg -y -v error -f rawvideo -pix\_fmt bgr24 -s {w}x{h} -r {fps} -i - \[-ss S -t D] -i IN -map 0:v:0 -map 1:a:0? -c:v libx264 -preset {preset} -crf {crf} -pix\_fmt yuv420p -c:a copy -shortest -movflags +faststart OUT` (`-ss/-t` only for segment runs). On failure with audio → retry `-c:a aac -b:a 192k`, warning `audio\_reencoded`.

\- More reader frames than crops → reuse last crop; log both counts. Re-check invariants for every rendered frame.

\- `--debug-overlay` → `<out stem>\_debug.mp4`: full frame scaled to `render.debug\_width`, all track boxes with id + score, selected crop rectangle, mode + reason text, VAD bar.



\## 12. Output validation → `run\_dir/validation.json` {ok, checks{name:{ok, detail}}}

Output exists; dims == (crop\_w, crop\_h); |duration − expected| ≤ max(2/fps, 0.1 s); audio present iff input has audio; full-clip run with copied audio → decoded audio MD5 (`ffmpeg -v error -i X -map 0:a -f md5 -`) equal for input and output; 0 invariant violations. Any failure → OutputValidationError (exit 5).



\## 13. run, metrics, cache

\- `run` = analyze → render (+debug) → validate; writes `run\_dir/metrics.json` {analysis\_s, render\_s, total\_s, clip\_duration\_s, realtime\_factor, peak\_rss\_mb, model\_size\_mb, config\_hash, git\_commit}; peak RSS over the whole run incl. ffmpeg children.

\- Run dir: `decision\_timeline.json, output.mp4, output\_debug.mp4 (optional), validation.json, metrics.json, logs/run.jsonl, logs/ffmpeg\_\*.log`.

\- `StageCache`: `{cache\_dir}/{input\_sha\[:16]}/{config\_hash}\_{start}\_{end}/{stage}.json`, stages `vision`, `audio`, `scene`; `--resume` loads (log `cache\_hit`), otherwise recompute (`cache\_miss`).



\## 14. Evaluation (only src/reframe/eval/ reads labels/)

Label `labels/<clip\_stem>.json`:

{"clip": "S1\_turns.mp4", "split": "test", "people": {"A": {"x\_range": \[0.0, 0.5]}, "B": {"x\_range": \[0.5, 1.0]}}, "intervals": \[{"start": 0.0, "end": 5.2, "expect": "A"}, {"start": 5.2, "end": 11.0, "expect": "B"}, {"start": 11.0, "end": 15.0, "expect": "FALLBACK"}], "tolerance\_s": 0.5}

Matched to runs by input file name. Metrics per run:

\- `visible\_speaker\_accuracy`: ticks in person intervals excluding each interval's first tolerance\_s; correct if mode=speaker and selected box centre\_x/W in that person's x\_range.

\- `fallback\_precision/recall/f1` (positive = FALLBACK) and `confident\_wrong\_crop\_rate` (labelled FALLBACK ticks framed as speaker); tolerance windows excluded.

\- `switch\_latency\_median\_s`, `switch\_latency\_p90\_s`, `switches\_missed`: per interval start T after the first, first tick ≥ T from which the correct target holds ≥ 0.5 s, minus T; missed if none before interval end.

\- `crop\_bound\_violations` (count, % frames) recomputed from crops; `jitter\_px\_per\_frame2` = mean |x\[f+1] − 2x\[f] + x\[f−1]| (and y) over triples in the same speaker segment without snap; `jitter\_norm` = value/W×1000; `p95\_speed\_px\_per\_frame`; `n\_switches`, `switches\_per\_min`; runtime/RAM copied from metrics.json.

Report: `per\_run.csv`, `summary.md`, `summary.json` in `--out`. Every metric unit-tested on hand-made examples.



\## 15. Benchmark (`reframe bench --suite configs/bench.yaml --out evidence/bench`)

Suite keys: `repeats, warmup, clips, aspects, variants \[{name, set: \[overrides]}], labels\_dir`. Each clip × aspect × variant: warm-up runs (not counted) then timed runs, each a subprocess `python -m reframe run ...` WITHOUT --resume; sampler measures the subprocess tree. Writes `<out>/<timestamp>/results.csv` (one row per timed run), `results\_summary.md` (median/min/max runtime, realtime factor, peak RAM, model size, quality metrics via eval on the last repeat), `machine.json`. Numbers are never edited by hand. `--quick` = repeats 1, warmup 0.



\## 16. CLI (exit codes: 0 ok, 2 invalid input, 3 missing dependency, 4 processing error, 5 output validation failed)

reframe validate INPUT

reframe analyze INPUT --aspect 9:16 --out RUN\_DIR \[--config F] \[--set k=v ...] \[--start S] \[--end E] \[--resume] \[--verbose]

reframe render RUN\_DIR/decision\_timeline.json --out RUN\_DIR/output.mp4 \[--debug-overlay]

reframe run INPUT --aspect 9:16|1:1 --out RUN\_DIR \[--config F] \[--set k=v ...] \[--start S] \[--end E] \[--resume] \[--debug-overlay] \[--verbose]

reframe validate-output RUN\_DIR

reframe eval --runs RUNS\_DIR --labels labels --out OUT\_DIR

reframe bench --suite configs/bench.yaml --out evidence/bench \[--quick]

Errors print one line `ERROR \[<code>] <message>` (traceback only with --verbose). `--aspect` overrides `crop.aspect`.



\## 17. configs/default.yaml (exact keys; values are STARTING points, tuned only on dev clips)

analysis: {sample\_fps: 10, analysis\_width: 640, detector: mediapipe, min\_face\_conf: 0.5, max\_faces: 4}

tracking: {iou\_match: 0.3, min\_hits: 2, max\_missed\_s: 0.7}

audio: {vad\_aggressiveness: 2, speech\_threshold: 0.5}

speaker: {strategy: largest\_face, window\_s: 0.8, min\_hold\_s: 1.2, switch\_margin: 0.15, switch\_confirm\_s: 0.4, enter\_conf: 0.35, fallback\_conf: 0.2, fallback\_enter\_s: 0.6, lip\_activity\_norm: 0.05}

scene: {enabled: true, cut\_threshold: 0.5, screen\_edge\_density: 0.08}

crop: {aspect: "9:16", headroom: 0.38, deadzone\_frac: 0.04, max\_speed\_frac\_s: 0.5, smoothing\_alpha: 0.15, safety\_margin\_frac: 0.1}

render: {crf: 20, preset: veryfast, fallback\_style: blur\_pad, debug\_width: 960}

runtime: {seed: 1234, cache\_dir: .cache, max\_duration\_s: 900, cv\_threads: 0}

