"""Load synthetic datasets written by ``agentfw.synth.generate``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pyarrow.parquet as pq

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.generate import GenerateConfig, Manifest

__all__ = ["load_dir", "load_labels", "load_labels_list", "load_traces"]


def _rows(path: str | Path) -> list[dict[str, object]]:
    p = Path(path)
    if p.suffix == ".parquet":
        rows: list[dict[str, object]] = pq.read_table(p).to_pylist()
        return rows
    return [orjson.loads(line) for line in p.read_bytes().splitlines() if line.strip()]


def load_traces(path: str | Path) -> list[RawCall]:
    """Raw calls from ``traces.jsonl`` or ``traces.parquet`` (``args``/``response_meta`` there
    are JSON strings, written by ``agentfw.sandbox.package``)."""
    rows = _rows(path)
    if Path(path).suffix == ".parquet":
        for d in rows:
            d["args"] = orjson.loads(str(d["args"]))
            rm = d["response_meta"]
            d["response_meta"] = None if rm is None else orjson.loads(str(rm))
    return [RawCall.model_validate(d) for d in rows]


def load_labels_list(path: str | Path) -> list[Label]:
    """Labels in file order (``.jsonl`` or ``.parquet``)."""
    return [Label.model_validate(d) for d in _rows(path)]


def load_labels(path: str | Path) -> dict[str, Label]:
    return {lab.envelope_id: lab for lab in load_labels_list(path)}


def _packaged_manifest(raw: dict[str, Any]) -> Manifest:
    """A ``Manifest`` view of a packaged sandbox dataset's ``manifest.json`` (it has no synthetic
    generator config: only the seed, file hashes and the headline counts carry over)."""
    c = raw["counts"]
    cfg = GenerateConfig(
        seed=int(raw["config"]["seed"]),
        personas_dir=Path("."),
        tasks_dir=Path("."),
        catalog_dir=Path("."),
        days=0,
    )
    counts = {"sessions": c["sessions"], "envelopes": c["actions"], "attacks": c["attacks"]}
    return Manifest(config=cfg, files=raw["files"], counts=counts)


def load_dir(d: str | Path) -> tuple[list[RawCall], dict[str, Label], Manifest]:
    """Traces, labels and manifest of a synthetic or packaged dataset directory (JSONL preferred,
    Parquet when no JSONL is present)."""
    d = Path(d)
    raw = orjson.loads((d / "manifest.json").read_bytes())
    manifest = _packaged_manifest(raw) if "version" in raw else Manifest.model_validate(raw)
    ext = "jsonl" if (d / "traces.jsonl").exists() else "parquet"
    return load_traces(d / f"traces.{ext}"), load_labels(d / f"labels.{ext}"), manifest
