"""Ground-truth label schema, loading and validation per CONTRACT §14.

labels/<clip_stem>.json ::
    {"clip": "S1_turns.mp4", "split": "test",
     "people": {"A": {"x_range": [0.0, 0.5]}, "B": {"x_range": [0.5, 1.0]}},
     "intervals": [{"start": 0.0, "end": 5.2, "expect": "A"}, ...],
     "tolerance_s": 0.5}

Runs are matched to labels by the `clip` field (the input file's name), not by
the label file's own filename -- CONTRACT §14: "Matched to runs by input file
name."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PersonRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_range: tuple[float, float]

    @field_validator("x_range")
    @classmethod
    def _validate_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"x_range must satisfy 0 <= lo < hi <= 1, got {list(v)}")
        return v


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    expect: str

    @model_validator(mode="after")
    def _validate_span(self) -> "Interval":
        if not (self.start < self.end):
            raise ValueError(f"interval start must be < end, got start={self.start} end={self.end}")
        return self


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip: str
    split: str = "test"
    people: dict[str, PersonRange] = {}
    intervals: list[Interval]
    tolerance_s: float

    @field_validator("clip")
    @classmethod
    def _clip_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("clip must be a non-empty filename")
        return v

    @field_validator("tolerance_s")
    @classmethod
    def _tolerance_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"tolerance_s must be > 0, got {v}")
        return v

    @field_validator("intervals")
    @classmethod
    def _intervals_nonempty(cls, v: list[Interval]) -> list[Interval]:
        if not v:
            raise ValueError("intervals must be non-empty")
        return v

    @model_validator(mode="after")
    def _validate_expect_refs(self) -> "Label":
        for iv in self.intervals:
            if iv.expect != "FALLBACK" and iv.expect not in self.people:
                raise ValueError(
                    f"interval expect={iv.expect!r} is neither 'FALLBACK' nor a key of "
                    f"people {sorted(self.people)}"
                )
        return self


@dataclass
class LabelSet:
    """Result of loading labels/*.json: valid labels keyed by clip filename,
    plus a map of label-filename -> error message for files that failed to
    parse or validate (never raised -- eval keeps going with what's left)."""

    by_clip: dict[str, Label] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def load_labels(labels_dir: Union[Path, str]) -> LabelSet:
    """Load and validate every labels/*.json file.

    A file that is not valid JSON, fails Label's schema validation, or claims
    a `clip` already claimed by an earlier file (alphabetically) is skipped
    and recorded in `.errors` rather than aborting the whole load.
    """
    labels_dir = Path(labels_dir)
    result = LabelSet()
    if not labels_dir.is_dir():
        return result

    for path in sorted(labels_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result.errors[path.name] = f"invalid JSON: {exc}"
            continue

        try:
            label = Label.model_validate(raw)
        except Exception as exc:  # pydantic.ValidationError, or a non-dict root
            result.errors[path.name] = f"invalid label: {exc}"
            continue

        if label.clip in result.by_clip:
            result.errors[path.name] = (
                f"duplicate clip '{label.clip}' -- already declared by another label file"
            )
            continue

        result.by_clip[label.clip] = label

    return result


__all__ = ["PersonRange", "Interval", "Label", "LabelSet", "load_labels"]
