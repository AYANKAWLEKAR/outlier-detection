"""Load synthetic datasets written by ``agentfw.synth.generate``."""

from __future__ import annotations

from pathlib import Path

import orjson

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.generate import Manifest

__all__ = ["load_dir", "load_labels", "load_traces"]


def _rows(path: str | Path) -> list[dict[str, object]]:
    return [orjson.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def load_traces(path: str | Path) -> list[RawCall]:
    return [RawCall.model_validate(d) for d in _rows(path)]


def load_labels(path: str | Path) -> dict[str, Label]:
    out: dict[str, Label] = {}
    for d in _rows(path):
        lab = Label.model_validate(d)
        out[lab.envelope_id] = lab
    return out


def load_dir(d: str | Path) -> tuple[list[RawCall], dict[str, Label], Manifest]:
    d = Path(d)
    manifest = Manifest.model_validate_json((d / "manifest.json").read_bytes())
    return load_traces(d / "traces.jsonl"), load_labels(d / "labels.jsonl"), manifest
