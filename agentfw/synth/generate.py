"""``python -m agentfw.synth.generate``: write traces / labels / manifest for a synthetic world.

Output files (all deterministic for a given config; the manifest carries no timestamps):
``traces.jsonl`` (``RawCall`` per line, sorted by ``(ts, session_id, seq_no)``), ``labels.jsonl``
(one ``Label`` per envelope, same order) and ``manifest.json`` (config, sha256 per file, counts).
Paths in the config are stored exactly as given.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import orjson
from pydantic import BaseModel

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.load import (
    CATALOG_DIR,
    PERSONAS_DIR,
    TASKS_DIR,
    load_catalog,
    load_personas,
    load_tasks,
    validate_world,
)
from agentfw.synth.sampler import sample_persona, stable_hash

__all__ = ["GenerateConfig", "Manifest", "generate", "main"]


class GenerateConfig(BaseModel):
    seed: int
    personas_dir: Path
    tasks_dir: Path
    catalog_dir: Path
    days: int
    start_ms: int = 1_767_225_600_000
    max_sessions: int | None = None  # TOTAL cap across personas (ruling R3)
    attack_rate: float = 0.015
    warmup_sessions: int = 30
    families: list[str] = []
    control_rate: float = 0.005


class Manifest(BaseModel):
    config: GenerateConfig
    files: dict[str, str]
    counts: dict[str, int]


def _split_cap(total: int | None, n: int) -> list[int | None]:
    """Even split of a total cap over ``n`` personas; remainder goes one each to the first."""
    if total is None:
        return [None] * n
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def attack_rng(seed: int) -> np.random.Generator:
    """Attack stream, independent of every persona stream (Task 0.7 draws from it)."""
    return np.random.default_rng([seed, stable_hash("attacks")])


def _inject_attacks(
    cfg: GenerateConfig, sessions: list[list[RawCall]]
) -> tuple[list[list[RawCall]], dict[str, Label]]:
    """Seam for Task 0.7: mutate benign sessions and return labels for attacked envelopes."""
    if cfg.families:
        raise NotImplementedError("attack families are wired in Task 0.7")
    return sessions, {}


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(orjson.dumps(r, option=orjson.OPT_SORT_KEYS) + b"\n" for r in rows)


def generate(cfg: GenerateConfig, out_dir: Path) -> Manifest:
    catalog, tasks = load_catalog(cfg.catalog_dir), load_tasks(cfg.tasks_dir)
    personas = sorted(load_personas(cfg.personas_dir), key=lambda p: p.agent_id)
    validate_world(catalog, personas, tasks)
    sessions: list[list[RawCall]] = []
    for p, cap in zip(personas, _split_cap(cfg.max_sessions, len(personas)), strict=True):
        sessions += sample_persona(
            p,
            catalog,
            tasks,
            days=cfg.days,
            global_seed=cfg.seed,
            start_ms=cfg.start_ms,
            max_sessions=cap,
        )
    sessions, attack_labels = _inject_attacks(cfg, sessions)
    raws = sorted(
        (c for s in sessions for c in s), key=lambda c: (c.ts_unix_ms, c.session_id, c.seq_no)
    )
    labels = [
        attack_labels.get(c.envelope_id) or Label(envelope_id=c.envelope_id, label="benign")
        for c in raws
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    blobs = {
        "traces.jsonl": _jsonl([c.model_dump(mode="json") for c in raws]),
        "labels.jsonl": _jsonl([lab.model_dump(mode="json") for lab in labels]),
    }
    for name, data in blobs.items():
        (out_dir / name).write_bytes(data)
    manifest = Manifest(
        config=cfg,
        files={n: hashlib.sha256(d).hexdigest() for n, d in blobs.items()},
        counts={
            "sessions": len(sessions),
            "envelopes": len(raws),
            "attacks": len({lab.attack_id for lab in labels if lab.attack_id is not None}),
        },
    )
    (out_dir / "manifest.json").write_bytes(
        orjson.dumps(
            manifest.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
        )
        + b"\n"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentfw.synth.generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--personas", type=Path, default=PERSONAS_DIR)
    ap.add_argument("--tasks", type=Path, default=TASKS_DIR)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DIR)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--families", type=lambda s: [f for f in s.split(",") if f], default=[])
    a = ap.parse_args(argv)
    cfg = GenerateConfig(
        seed=a.seed,
        personas_dir=a.personas,
        tasks_dir=a.tasks,
        catalog_dir=a.catalog,
        days=a.days,
        max_sessions=a.max_sessions,
        families=a.families,
    )
    m = generate(cfg, a.out)
    print(f"wrote {m.counts['envelopes']} envelopes in {m.counts['sessions']} sessions to {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
