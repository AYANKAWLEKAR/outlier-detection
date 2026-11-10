"""Envelope sinks: where the proxy writes raw calls, labels and shadow verdicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Protocol

import orjson

from agentfw.core.envelope import DetectorResult, Label, RawCall

__all__ = ["CALLS", "LABELS", "VERDICTS", "EnvelopeSink", "JsonlSink"]

CALLS = "calls.jsonl"
LABELS = "labels.jsonl"
VERDICTS = "verdicts.jsonl"
_OPTS = orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_PASSTHROUGH_DATACLASS


class EnvelopeSink(Protocol):
    def write_call(self, raw: RawCall) -> None: ...

    def write_labels(self, labels: Sequence[Label]) -> None: ...

    def write_verdict(self, envelope_id: str, result: DetectorResult) -> None: ...


def _default(obj: object) -> object:
    # dataclasses pass through here as dicts so OPT_SORT_KEYS also orders their fields
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=str)
    return str(obj)


class JsonlSink:
    """Appends one sorted-key JSON object per line to ``calls.jsonl`` / ``labels.jsonl`` /
    ``verdicts.jsonl`` under ``dir``. Each write opens, appends and closes (so it is flushed)."""

    def __init__(self, dir: Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, rows: Sequence[object]) -> None:
        if not rows:
            return
        data = b"".join(orjson.dumps(r, default=_default, option=_OPTS) + b"\n" for r in rows)
        with (self.dir / name).open("ab") as f:
            f.write(data)

    def write_call(self, raw: RawCall) -> None:
        self._append(CALLS, [raw.model_dump(mode="json")])

    def write_labels(self, labels: Sequence[Label]) -> None:
        self._append(LABELS, [lab.model_dump(mode="json") for lab in labels])

    def write_verdict(self, envelope_id: str, result: DetectorResult) -> None:
        self._append(VERDICTS, [{"envelope_id": envelope_id, "mode": "shadow", "result": result}])
