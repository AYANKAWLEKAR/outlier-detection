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
from typing import Literal

import numpy as np
import orjson
from pydantic import BaseModel, Field

from agentfw.core.envelope import Label, RawCall
from agentfw.synth.attacks import AttackContext, inject
from agentfw.synth.dists import IdPool
from agentfw.synth.load import (
    CATALOG_DIR,
    PERSONAS_DIR,
    TASKS_DIR,
    load_catalog,
    load_personas,
    load_tasks,
    validate_world,
)
from agentfw.synth.sampler import sample_persona_tagged, stable_hash
from agentfw.synth.schema import Catalog, Persona

__all__ = ["GenerateConfig", "Manifest", "generate", "main"]


class GenerateConfig(BaseModel):
    seed: int
    personas_dir: Path
    tasks_dir: Path
    catalog_dir: Path
    days: int = Field(ge=0)
    start_ms: int = 1_767_225_600_000
    max_sessions: int | None = Field(default=None, ge=0)  # TOTAL cap across personas (ruling R3)
    attack_rate: float = 0.015
    warmup_sessions: int = 30
    families: list[str] = []
    persona_ids: list[str] | None = None  # select a subset of ``personas_dir`` (None = all)
    control_rate: float = 0.005
    # Amendment A1.4. "own" = spec 4.6 (multiply the call's own value by m). "far" = the spike
    # lands at or beyond ``magnitude_far_mult`` x the persona's benign p99 of the field
    # (``attacks.far_bounds``); 5.0 = ``attacks.CONTROL_P99_MULT``, the benign-labelled control.
    magnitude_mode: Literal["own", "far"] = "own"
    magnitude_far_mult: float = Field(default=5.0, gt=0)


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
    cfg: GenerateConfig,
    catalog: Catalog,
    personas: list[Persona],
    by_persona: dict[str, list[list[RawCall]]],
) -> tuple[dict[str, list[list[RawCall]]], dict[str, Label], dict[str, int]]:
    """Mutate benign sessions per ``cfg.families``; returns attack/control labels by envelope id
    and the magnitude census (``inject``'s ``stats``)."""
    stats = {"magnitude_first_pass": 0, "magnitude_dropped": 0, "magnitude_topup": 0}
    if not cfg.families:
        return by_persona, {}, stats
    by_id = {p.agent_id: p for p in personas}

    def ctx_factory(
        agent: str, attack_id: str, persona_sessions: Sequence[list[RawCall]], idx: int
    ) -> AttackContext:
        p = by_id[agent]
        pool_rng = np.random.default_rng([cfg.seed, stable_hash(f"attack-pools:{agent}")])
        pools = {
            n: IdPool(sp.prefix, sp.size, sp.s, pool_rng) for n, sp in sorted(p.id_pools.items())
        }
        return AttackContext(catalog, p, pools, attack_id, persona_sessions, idx)

    out, labels = inject(by_persona, cfg, attack_rng(cfg.seed), ctx_factory, stats=stats)
    return out, {lab.envelope_id: lab for lab in labels}, stats


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(orjson.dumps(r, option=orjson.OPT_SORT_KEYS) + b"\n" for r in rows)


def generate(cfg: GenerateConfig, out_dir: Path) -> Manifest:
    catalog, tasks = load_catalog(cfg.catalog_dir), load_tasks(cfg.tasks_dir)
    personas = sorted(load_personas(cfg.personas_dir), key=lambda p: p.agent_id)
    validate_world(catalog, personas, tasks)
    if cfg.persona_ids is not None:
        missing = sorted(set(cfg.persona_ids) - {p.agent_id for p in personas})
        if missing:
            raise ValueError(f"unknown persona ids: {missing}")
        personas = [p for p in personas if p.agent_id in cfg.persona_ids]
    by_persona: dict[str, list[list[RawCall]]] = {}
    quarter_end: set[str] = set()
    for p, cap in zip(personas, _split_cap(cfg.max_sessions, len(personas)), strict=True):
        sampled = sample_persona_tagged(
            p,
            catalog,
            tasks,
            days=cfg.days,
            global_seed=cfg.seed,
            start_ms=cfg.start_ms,
            max_sessions=cap,
        )
        by_persona[p.agent_id] = sampled.sessions
        quarter_end |= sampled.quarter_end_session_ids
    by_persona, attack_labels, magnitude = _inject_attacks(cfg, catalog, personas, by_persona)
    sessions = [s for a in sorted(by_persona) for s in by_persona[a]]
    raws = sorted(
        (c for s in sessions for c in s), key=lambda c: (c.ts_unix_ms, c.session_id, c.seq_no)
    )
    # Precedence per envelope: attack > control (injected, or any call of a quarter-end session)
    # > benign.
    labels = [
        attack_labels.get(c.envelope_id)
        or Label(
            envelope_id=c.envelope_id,
            label="benign",
            control_type="benign_unusual" if c.session_id in quarter_end else None,
        )
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
            **(magnitude if cfg.magnitude_mode == "far" else {}),  # dropped spikes: never silent
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
    ap.add_argument("--magnitude-mode", choices=["own", "far"], default="own")
    ap.add_argument("--magnitude-far-mult", type=float, default=5.0)
    a = ap.parse_args(argv)
    cfg = GenerateConfig(
        seed=a.seed,
        personas_dir=a.personas,
        tasks_dir=a.tasks,
        catalog_dir=a.catalog,
        days=a.days,
        max_sessions=a.max_sessions,
        families=a.families,
        magnitude_mode=a.magnitude_mode,
        magnitude_far_mult=a.magnitude_far_mult,
    )
    m = generate(cfg, a.out)
    print(f"wrote {m.counts['envelopes']} envelopes in {m.counts['sessions']} sessions to {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
