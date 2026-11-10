"""Retiming sandbox sessions onto a simulated timeline (D-015, R11)."""

from __future__ import annotations

import datetime as dt

import pytest

from agentfw.core.envelope import RawCall
from agentfw.sandbox.retime import (
    MAX_GAP_MS,
    MIN_GAP_MS,
    WORKING_HOURS,
    retime,
)
from agentfw.sandbox.tasks.schema import AgentDef

START = 1_767_225_600_000  # 2026-01-01T00:00:00Z
WALL = 1_790_000_000_000


def _agent(agent_id: str, per_day: float = 6.0) -> AgentDef:
    return AgentDef(
        agent_id=agent_id,
        system_prompt="x",
        tools=["zendesk.get_ticket"],
        template_mix={"t": 1.0},
        sessions_per_day=per_day,
    )


def _session(agent: str, sid: str, start: int, gaps_ms: list[int]) -> list[RawCall]:
    ts = [start]
    for g in gaps_ms:
        ts.append(ts[-1] + g)
    return [
        RawCall(
            envelope_id=f"{sid}-e{i:03d}",
            session_id=sid,
            agent_id=agent,
            principal_id=f"{agent}-principal",
            seq_no=i,
            ts_unix_ms=t,
            tool_id="zendesk.get_ticket",
            args={"ticket_id": f"tk_{i}"},
        )
        for i, t in enumerate(ts)
    ]


def _by_session(calls: list[RawCall]) -> dict[str, list[RawCall]]:
    out: dict[str, list[RawCall]] = {}
    for c in calls:
        out.setdefault(c.session_id, []).append(c)
    return {k: sorted(v, key=lambda c: c.seq_no) for k, v in out.items()}


def _gaps(calls: list[RawCall]) -> list[int]:
    return [b.ts_unix_ms - a.ts_unix_ms for a, b in zip(calls, calls[1:], strict=False)]


def test_in_session_gaps_kept_but_clamped() -> None:
    sessions = {"s1": _session("a", "s1", WALL, [100, 5_000, 300_000, 30_000, 120_000, 500])}
    out = _by_session(retime(sessions, [_agent("a")], seed=0, start_ms=START))["s1"]
    assert [c.envelope_id for c in out] == [c.envelope_id for c in sessions["s1"]]
    assert _gaps(out) == [MIN_GAP_MS, 5_000, MAX_GAP_MS, 30_000, 120_000, 500]
    assert (MIN_GAP_MS, MAX_GAP_MS) == (500, 120_000)


def test_internal_order_kept_even_when_wall_clock_goes_backwards() -> None:
    # burst extras are logged after their original with future timestamps, so the next real call
    # can carry an earlier wall-clock ts than the previous (seq-order) call
    calls = _session("a", "s1", WALL, [1_000, 2_000, 3_000])
    calls[3] = calls[3].model_copy(update={"ts_unix_ms": calls[1].ts_unix_ms})
    out = _by_session(retime({"s1": calls}, [_agent("a")], seed=0, start_ms=START))["s1"]
    assert [c.seq_no for c in out] == [0, 1, 2, 3]
    assert _gaps(out) == [1_000, 2_000, MIN_GAP_MS]


def test_burst_gaps_kept_exactly() -> None:
    calls = _session("a", "s1", WALL, [2_000, 1, 250_000, 40])
    burst = {calls[2].envelope_id, calls[3].envelope_id, calls[4].envelope_id}
    out = _by_session(
        retime({"s1": calls}, [_agent("a")], seed=0, start_ms=START, burst_ids=burst)
    )["s1"]
    assert _gaps(out) == [2_000, 1, 250_000, 40]


def test_session_starts_inside_working_hours_and_in_agent_order() -> None:
    sessions: dict[str, list[RawCall]] = {}
    for k in range(25):
        for agent in ("a", "b"):
            sid = f"{agent}-{k:03d}"
            # wall-clock order deliberately differs from session-id order for agent b
            start = WALL + (k if agent == "a" else 25 - k) * 60_000
            sessions[sid] = _session(agent, sid, start, [1_000, 2_000])
    out = retime(sessions, [_agent("a", 4.0), _agent("b", 9.0)], seed=3, start_ms=START)
    by = _by_session(out)
    lo, hi = WORKING_HOURS
    for sid, calls in by.items():
        t0 = dt.datetime.fromtimestamp(calls[0].ts_unix_ms / 1000, dt.UTC)
        assert lo <= t0.hour < hi, (sid, t0)
        assert calls[0].ts_unix_ms >= START
    for agent in ("a", "b"):
        mine = sorted(
            (s for s in sessions if s.startswith(agent)), key=lambda s: sessions[s][0].ts_unix_ms
        )
        starts = [by[s][0].ts_unix_ms for s in mine]
        assert starts == sorted(starts)  # agent's sessions keep their wall-clock order


@pytest.mark.parametrize("per_day", [3.0, 12.0])
def test_sessions_per_simulated_day_follow_the_agent_rate(per_day: float) -> None:
    sessions = {f"s{k:04d}": _session("a", f"s{k:04d}", WALL + k, [1_000]) for k in range(600)}
    out = retime(sessions, [_agent("a", per_day)], seed=0, start_ms=START)
    days = {c.ts_unix_ms // 86_400_000 for c in out if c.seq_no == 0}
    # every simulated day up to the last is either used or a Poisson zero; the mean is the rate
    span = max(days) - min(days) + 1
    assert 0.85 * per_day < 600 / span < 1.15 * per_day


def test_output_is_complete_time_ordered_and_deterministic() -> None:
    sessions = {
        f"{a}-{k}": _session(a, f"{a}-{k}", WALL + k * 7_000, [700, 900])
        for a in ("a", "b")
        for k in range(5)
    }
    agents = [_agent("a"), _agent("b")]
    out = retime(sessions, agents, seed=1, start_ms=START)
    assert sorted(c.envelope_id for c in out) == sorted(
        c.envelope_id for s in sessions.values() for c in s
    )
    keys = [(c.ts_unix_ms, c.session_id, c.seq_no) for c in out]
    assert keys == sorted(keys)
    assert out == retime(sessions, agents, seed=1, start_ms=START)
    assert out != retime(sessions, agents, seed=2, start_ms=START)
    # only timestamps change
    orig = {c.envelope_id: c for s in sessions.values() for c in s}
    for c in out:
        assert c.model_copy(update={"ts_unix_ms": 0}) == orig[c.envelope_id].model_copy(
            update={"ts_unix_ms": 0}
        )


def test_unknown_agent_is_an_error() -> None:
    sessions = {"s1": _session("ghost", "s1", WALL, [1_000])}
    with pytest.raises(ValueError, match="ghost"):
        retime(sessions, [_agent("a")], seed=0, start_ms=START)


def test_empty_sessions_are_dropped() -> None:
    sessions = {"s1": [], "s2": _session("a", "s2", WALL, [1_000])}
    out = retime(sessions, [_agent("a")], seed=0, start_ms=START)
    assert {c.session_id for c in out} == {"s2"}
