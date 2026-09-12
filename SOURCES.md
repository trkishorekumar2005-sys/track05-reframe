\# SOURCES.md — Third-party components, licences, attribution



\## Python packages (see uv.lock for exact pinned versions)

| Package | Role | Licence | Source |

|---|---|---|---|

| mediapipe | Face detection \& landmarks | Apache 2.0 | https://github.com/google-ai-edge/mediapipe |

| opencv-contrib-python | Image processing | Apache 2.0 | https://github.com/opencv/opencv-python |

| numpy | Numerical arrays | BSD-3-Clause | https://numpy.org |

| scipy | Hungarian matching (tracker) | BSD-3-Clause | https://scipy.org |

| pydantic | Config validation | MIT | https://github.com/pydantic/pydantic |

| pyyaml | Config file parsing | MIT | https://pyyaml.org |

| typer | CLI framework | MIT | https://github.com/tiangolo/typer |

| psutil | RAM/resource measurement | BSD-3-Clause | https://github.com/giampaolo/psutil |

| webrtcvad-wheels | Voice activity detection | BSD-3-Clause | https://github.com/wiseman/py-webrtcvad |

| matplotlib | Debug plots | PSF/BSD-style | https://matplotlib.org |

| pytest | Testing | MIT | https://pytest.org |

| hypothesis | Property-based testing | MPL-2.0 | https://hypothesis.readthedocs.io |



(Run `uv tree --depth 1` for the exact list and versions actually installed; add/remove rows to match.)



\## Models

| Model | Role | Licence | Source | SHA-256 |

|---|---|---|---|---|

| blaze\_face\_short\_range.tflite | Face detection | Apache 2.0 | https://storage.googleapis.com/mediapipe-models/face\_detector/blaze\_face\_short\_range/float16/1/blaze\_face\_short\_range.tflite | see models/manifest.lock.json |

| face\_landmarker.task | Mouth/lip landmarks | Apache 2.0 | https://storage.googleapis.com/mediapipe-models/face\_landmarker/face\_landmarker/float16/1/face\_landmarker.task | see models/manifest.lock.json |



\## External tools

| Tool | Role | Licence | Note |

|---|---|---|---|

| FFmpeg | Video/audio decode, encode, mux | LGPL/GPL depending on build | Invoked as an external process only, not linked into the product; not redistributed with the repo |



\## Media

\- All video/audio in `data/clips/` and `data/raw/` was recorded by the candidate and consenting participants specifically for this assessment. Not redistributed beyond evaluation.



\## Development tools (not part of the running product)

\- Claude Code (Anthropic) — used for implementation assistance across all coding phases (see AI\_USE\_log.md for details).

\- Claude (claude.ai) — used for planning, architecture, and debugging guidance (see AI\_USE\_log.md).



\## Commercial use / offline note

All components above run fully locally/offline. No hosted or paid API is used anywhere in src/reframe/.

