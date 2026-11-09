from __future__ import annotations

import numpy as np
import pytest

from agentfw.synth.sampler import sample_persona, stable_hash
from tests.synth.conftest import World

DAY_MS = 86_400_000
START = 1_767_225_600_000  # 2026-01-01T00:00:00Z


def _sample(world: World, pid: str, *, days: int = 3, seed: int = 0, cap: int | None = None):  # type: ignore[no-untyped-def]
    return sample_persona(
        world.persona(pid),
        world.catalog,
        world.tasks,
        days=days,
        global_seed=seed,
        start_ms=START,
        max_sessions=cap,
    )


def test_stable_hash_is_xxhash64_and_stable() -> None:
    assert stable_hash("agent-a") == stable_hash("agent-a")
    assert stable_hash("agent-a") != stable_hash("agent-b")
    assert 0 <= stable_hash("x") < 2**64
    assert stable_hash("") == 0xEF46DB3751D8E999  # xxh64 of the empty string, seed 0


def test_deterministic_and_seed_sensitive(world: World) -> None:
    a = _sample(world, "outreach-a")
    b = _sample(world, "outreach-a")
    assert [[c.model_dump(mode="json") for c in s] for s in a] == [
        [c.model_dump(mode="json") for c in s] for s in b
    ]
    c = _sample(world, "outreach-a", seed=1)
    assert [s[0].envelope_id for s in a] != [s[0].envelope_id for s in c]


def test_sessions_well_formed(world: World) -> None:
    pid = "support-refunds-a"
    p = world.persona(pid)
    sessions = _sample(world, pid, days=2)
    assert sessions
    starts = [s[0].ts_unix_ms for s in sessions]
    assert starts == sorted(starts)  # time order
    ids = [c.envelope_id for s in sessions for c in s]
    assert len(ids) == len(set(ids))
    assert len({s[0].session_id for s in sessions}) == len(sessions)
    lo, hi = p.working_hours
    for s in sessions:
        assert [c.seq_no for c in s] == list(range(len(s)))
        ts = [c.ts_unix_ms for c in s]
        assert ts == sorted(ts)
        assert len({c.session_id for c in s}) == 1
        assert all(c.agent_id == pid and c.principal_id == f"{pid}-principal" for c in s)
        hour = ((s[0].ts_unix_ms - START) % DAY_MS) / 3_600_000
        assert lo <= hour < hi


def test_no_empty_sessions(world: World) -> None:
    p = world.persona("support-refunds-a")
    noisy = p.model_copy(update={"noise": p.noise.model_copy(update={"abandon": 1.0})})
    sessions = sample_persona(
        noisy, world.catalog, world.tasks, days=2, global_seed=3, start_ms=START
    )
    assert all(len(s) >= 1 for s in sessions)


def test_max_sessions_caps_total(world: World) -> None:
    sessions = _sample(world, "support-refunds-a", days=5, cap=7)
    assert len(sessions) == 7
    full = _sample(world, "support-refunds-a", days=5)
    assert [c.envelope_id for s in sessions for c in s] == [
        c.envelope_id for s in full[:7] for c in s
    ]


def test_session_rate_mean(world: World) -> None:
    pid = "support-refunds-a"
    lam = world.persona(pid).session_rate.lam
    assert lam is not None
    days = 30
    sessions = _sample(world, pid, days=days)
    per_day = np.bincount([(s[0].ts_unix_ms - START) // DAY_MS for s in sessions], minlength=days)
    assert abs(per_day.mean() - lam) / lam < 0.10


def test_repeat_tasks_yield_multi_item_sessions(world: World) -> None:
    sessions = _sample(world, "outreach-a", days=2)
    assert max(len(s) for s in sessions) > 1


def test_id_flow_pools_shared_across_sessions(world: World) -> None:
    # hot ids recur across sessions because pools are built once per persona
    sessions = _sample(world, "support-refunds-a", days=3)
    tickets = [
        c.args["ticket_id"] for s in sessions for c in s if c.tool_id == "zendesk.get_ticket"
    ]
    assert len(tickets) > len(set(tickets))


@pytest.mark.parametrize("pid", ["support-refunds-a", "sales-ops-a", "outreach-a"])
def test_all_personas_sample(world: World, pid: str) -> None:
    assert _sample(world, pid, days=1)
