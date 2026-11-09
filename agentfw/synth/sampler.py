"""Turns a persona into timestamped sessions of ``RawCall`` (spec 4.5).

Determinism: the per-persona RNG is ``default_rng([global_seed, stable_hash(agent_id)])`` and is the
only source of randomness (pools, session counts, start times, task mix, walks, ids).

Sampling model, per simulated day (midnight UTC from ``start_ms``): the session count ``N`` is
drawn from ``persona.session_rate``; the ``N`` start times are i.i.d. uniform over the persona's
UTC ``working_hours`` window ``[start, end)`` and sorted, which is exactly a Poisson process over
the window conditioned on ``N`` events. Every task walk uses one per-persona ``weights_cache`` so
categorical Dirichlet weights are drawn once per persona, not per session.

A walk may return no calls (abandon on a non-emitted first step). Such draws are dropped: no empty
session is emitted, but the draw still happened, so the attempted session count per day keeps the
configured rate (the emitted count is slightly below it). ``max_sessions`` counts emitted sessions
and truncates in time order.
"""

from __future__ import annotations

import numpy as np
import xxhash

from agentfw.core.envelope import RawCall
from agentfw.core.ids import new_ulid
from agentfw.synth.dists import IdPool, WeightsCache, sample
from agentfw.synth.grammar import walk
from agentfw.synth.schema import Catalog, Persona, TaskSpec

__all__ = ["persona_rng", "sample_persona", "stable_hash"]

_DAY_MS = 86_400_000
_HOUR_MS = 3_600_000


def stable_hash(s: str) -> int:
    """xxhash64 of ``s`` (never Python's salted ``hash()``)."""
    return xxhash.xxh64_intdigest(s.encode())


def persona_rng(global_seed: int, agent_id: str) -> np.random.Generator:
    return np.random.default_rng([global_seed, stable_hash(agent_id)])


def sample_persona(
    persona: Persona,
    catalog: Catalog,
    tasks: dict[str, TaskSpec],
    *,
    days: int,
    global_seed: int,
    start_ms: int,
    max_sessions: int | None = None,
) -> list[list[RawCall]]:
    """Sessions for one persona in time order (each a list of calls with seq_no 0..n-1)."""
    rng = persona_rng(global_seed, persona.agent_id)
    pools = {
        name: IdPool(spec.prefix, spec.size, spec.s, rng)
        for name, spec in sorted(persona.id_pools.items())
    }
    cache: WeightsCache = {}
    names = [t.name for t in persona.tasks]
    w = np.array([t.weight for t in persona.tasks], dtype=np.float64)
    probs = w / w.sum()
    lo, hi = persona.working_hours
    sessions: list[list[RawCall]] = []
    for day in range(days):
        day_ms = start_ms + day * _DAY_MS
        n = max(0, round(float(sample(persona.session_rate, rng))))
        starts = np.sort(rng.integers(day_ms + lo * _HOUR_MS, day_ms + hi * _HOUR_MS, size=n))
        for st in starts:
            task = tasks[names[int(rng.choice(len(names), p=probs))]]
            planned = walk(task, persona, catalog, rng, pools, weights_cache=cache)
            if not planned:
                continue
            session_id = new_ulid(int(st), rng)
            ts = int(st)
            calls: list[RawCall] = []
            for i, pc in enumerate(planned):
                ts += round(pc.gap_s * 1000)
                calls.append(
                    RawCall(
                        envelope_id=new_ulid(ts, rng),
                        session_id=session_id,
                        agent_id=persona.agent_id,
                        principal_id=f"{persona.agent_id}-principal",
                        seq_no=i,
                        ts_unix_ms=ts,
                        tool_id=pc.tool_id,
                        args=pc.args,
                        response_meta=pc.response_meta,
                    )
                )
            sessions.append(calls)
            if max_sessions is not None and len(sessions) >= max_sessions:
                return sessions
    return sessions
