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

Quarter-end scheduling: a persona with ``quarter_end_days`` and a ``quarter_end_mass_update`` entry
in its task list (weight 0, so it is never drawn from the regular mix) runs that task in a session
with probability ``QUARTER_END_PROB`` on every simulated day that is a quarter-end day: the UTC
calendar day-of-month is in ``quarter_end_days`` AND the month is Mar/Jun/Sep/Dec. The extra draw
happens only on such days, so every other day's stream is untouched. ``sample_persona_tagged``
also returns the ids of those sessions (the generator labels them ``benign_unusual``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import xxhash

from agentfw.core.envelope import RawCall
from agentfw.core.ids import new_ulid
from agentfw.synth.dists import IdPool, WeightsCache, sample
from agentfw.synth.grammar import walk
from agentfw.synth.schema import Catalog, Persona, TaskSpec

__all__ = [
    "QUARTER_END_PROB",
    "QUARTER_END_TASK",
    "SampledPersona",
    "persona_rng",
    "sample_persona",
    "sample_persona_tagged",
    "stable_hash",
]

_DAY_MS = 86_400_000
_HOUR_MS = 3_600_000
QUARTER_END_TASK = "quarter_end_mass_update"
QUARTER_END_PROB = 0.5
_QUARTER_END_MONTHS = (3, 6, 9, 12)


@dataclass(slots=True)
class SampledPersona:
    sessions: list[list[RawCall]]
    quarter_end_session_ids: set[str] = field(default_factory=set)


def _is_quarter_end_day(persona: Persona, day_ms: int) -> bool:
    if not persona.quarter_end_days:
        return False
    d = dt.datetime.fromtimestamp(day_ms / 1000, dt.UTC)
    return d.month in _QUARTER_END_MONTHS and d.day in persona.quarter_end_days


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
    return sample_persona_tagged(
        persona,
        catalog,
        tasks,
        days=days,
        global_seed=global_seed,
        start_ms=start_ms,
        max_sessions=max_sessions,
    ).sessions


def sample_persona_tagged(
    persona: Persona,
    catalog: Catalog,
    tasks: dict[str, TaskSpec],
    *,
    days: int,
    global_seed: int,
    start_ms: int,
    max_sessions: int | None = None,
) -> SampledPersona:
    """``sample_persona`` plus the ids of the scheduled quarter-end sessions."""
    out = SampledPersona([])
    if max_sessions is not None and max_sessions <= 0:
        return out
    rng = persona_rng(global_seed, persona.agent_id)
    pools = {
        name: IdPool(spec.prefix, spec.size, spec.s, rng)
        for name, spec in sorted(persona.id_pools.items())
    }
    cache: WeightsCache = {}
    regular = [t for t in persona.tasks if t.weight > 0]
    names = [t.name for t in regular]
    w = np.array([t.weight for t in regular], dtype=np.float64)
    probs = w / w.sum()
    lo, hi = persona.working_hours
    sessions = out.sessions
    has_qe = any(t.name == QUARTER_END_TASK for t in persona.tasks)
    for day in range(days):
        day_ms = start_ms + day * _DAY_MS
        n = max(0, round(float(sample(persona.session_rate, rng))))
        starts = np.sort(rng.integers(day_ms + lo * _HOUR_MS, day_ms + hi * _HOUR_MS, size=n))
        for st in starts:
            qe = has_qe and _is_quarter_end_day(persona, day_ms) and rng.random() < QUARTER_END_PROB
            task = tasks[QUARTER_END_TASK if qe else names[int(rng.choice(len(names), p=probs))]]
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
            if qe:
                out.quarter_end_session_ids.add(session_id)
            if max_sessions is not None and len(sessions) >= max_sessions:
                return out
    return out
