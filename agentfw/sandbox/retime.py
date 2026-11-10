"""Retime sandbox sessions onto a simulated timeline (D-015, risk R11).

A sandbox run compresses weeks of agent work into hours of wall clock, which would make velocity
baselines meaningless. Packaging therefore moves every session onto a simulated calendar:

* **Session starts** follow the synthetic generator's model: each simulated day an agent starts
  ``Poisson(sessions_per_day)`` sessions at sorted uniform times inside ``WORKING_HOURS`` (UTC),
  i.e. a Poisson process over working hours. An agent's sessions keep their wall-clock order
  (ties broken by ``session_id``) and fill the days in turn until all are placed. As in the
  generator, sessions of one agent may overlap.
* **In-session gaps** (consecutive calls in ``seq_no`` order) are kept but clamped to
  ``[MIN_GAP_MS, MAX_GAP_MS]`` = [0.5 s, 120 s]. A negative wall-clock gap (a real call logged
  after injected burst duplicates whose timestamps lie in the future) is clamped to 0.5 s, so the
  internal order is always preserved.
* **Burst gaps** — the gap *into* any envelope listed in ``burst_ids`` (the ``velocity_burst``
  duplicates; packaging takes them from the labels) — are kept exactly, unclamped, so a burst's
  rate survives retiming. Raw calls alone cannot tell a burst duplicate from a legitimately
  repeated call, so ``burst_ids`` is a required keyword (pass an empty set when there are none).
* ``start_ms`` must be a UTC midnight: it is simulated day 0, and working hours are counted
  from it.

Only ``ts_unix_ms`` changes: envelope ids (which labels key on) are kept as they are.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np

from agentfw.core.envelope import RawCall
from agentfw.sandbox.tasks.schema import AgentDef
from agentfw.synth.sampler import stable_hash

__all__ = ["MAX_GAP_MS", "MIN_GAP_MS", "WORKING_HOURS", "retime"]

MIN_GAP_MS = 500
MAX_GAP_MS = 120_000
WORKING_HOURS = (8, 18)  # UTC [start, end), the synthetic personas' default
_HOUR_MS = 3_600_000
_DAY_MS = 24 * _HOUR_MS
_MAX_DAYS = 100_000  # guards the day loop against a pathological sessions_per_day


def _session_starts(n: int, per_day: float, rng: np.random.Generator, start_ms: int) -> list[int]:
    """``n`` session start times: a Poisson process over each simulated day's working hours."""
    lo, hi = WORKING_HOURS
    starts: list[int] = []
    day = 0
    while len(starts) < n:
        if day >= _MAX_DAYS:
            raise ValueError(f"cannot place {n} sessions at {per_day}/day within {_MAX_DAYS} days")
        day_ms = start_ms + day * _DAY_MS
        k = int(rng.poisson(per_day))
        times = np.sort(rng.integers(day_ms + lo * _HOUR_MS, day_ms + hi * _HOUR_MS, size=k))
        starts.extend(int(t) for t in times)
        day += 1
    return starts[:n]


def _retime_session(calls: list[RawCall], start: int, burst_ids: Collection[str]) -> list[RawCall]:
    out: list[RawCall] = []
    ts = start
    prev: RawCall | None = None
    for c in calls:
        if prev is not None:
            gap = c.ts_unix_ms - prev.ts_unix_ms
            if c.envelope_id not in burst_ids:
                gap = min(max(gap, MIN_GAP_MS), MAX_GAP_MS)
            ts += gap
        out.append(c.model_copy(update={"ts_unix_ms": ts}))
        prev = c
    return out


def retime(
    sessions: Mapping[str, list[RawCall]],
    agents: Sequence[AgentDef],
    seed: int,
    start_ms: int,
    *,
    burst_ids: Collection[str],
) -> list[RawCall]:
    """Every call of ``sessions`` ({session_id: calls}) on the simulated timeline, time-ordered.

    Deterministic in ``seed``: agent ``a``'s starts use ``default_rng([seed, stable_hash(a)])``.
    Empty sessions are dropped; a session whose agent is not in ``agents`` is an error.
    """
    if start_ms % _DAY_MS:
        raise ValueError(f"start_ms {start_ms} is not a UTC midnight (day 0 of the timeline)")
    rate = {a.agent_id: a.sessions_per_day for a in agents}
    burst = frozenset(burst_ids)
    by_agent: dict[str, list[list[RawCall]]] = {}
    for sid in sorted(sessions):
        calls = sorted(sessions[sid], key=lambda c: c.seq_no)
        if not calls:
            continue
        agent = calls[0].agent_id
        if agent not in rate:
            raise ValueError(f"session {sid!r}: agent {agent!r} has no AgentDef")
        by_agent.setdefault(agent, []).append(calls)
    out: list[RawCall] = []
    for agent in sorted(by_agent):
        group = sorted(by_agent[agent], key=lambda s: (s[0].ts_unix_ms, s[0].session_id))
        rng = np.random.default_rng([seed, stable_hash(agent)])
        for calls, start in zip(
            group, _session_starts(len(group), rate[agent], rng, start_ms), strict=True
        ):
            out.extend(_retime_session(calls, start, burst))
    out.sort(key=lambda c: (c.ts_unix_ms, c.session_id, c.seq_no))
    return out
