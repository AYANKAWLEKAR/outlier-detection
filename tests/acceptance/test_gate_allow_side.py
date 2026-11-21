"""Allow-side acceptance of the pre-execution gate (feature spec F4, Task 3.1; slow).

G-a: 200 benign synthetic sessions, each ending in a valid irreversible call, replayed through an
engine with ``gate`` (and ``flow_ledger``) on must produce no ``BLOCK`` on those calls (0 / 200:
one-sided 95% upper bound 3 / 200 = 1.5%, the spec's bar).

The world: every persona with ``id_flow_rate = 1.0`` that uses a tool the catalog annotates
destructive (irreversible whatever the engine has learned), all days benign (no attacks, no
controls), with the dataset's ``schemas.jsonl`` loaded as hints (the annotated catalog, so every
regime has schemas and the prerequisite check runs). A destructive call is *valid* when every
id-like value in its args was returned by an earlier call of the session or seeded from the task
input (``task_ids``) -- ground truth from the raw calls, independent of the tracker and the
ledger -- and *stale-sensitive* when one of those ids was first returned by the call just before
it. Each session's evaluation call is its last stale-sensitive valid destructive call, else its
last valid one; the evaluation set is every stale-sensitive session (latest first), topped up with
the latest other eligible sessions to 200, each truncated after its evaluation call; every other
session runs whole. The replay follows the review protocol (flagged benign calls are approved), so
every call is learned. Floors on the stale-sensitive count pin the suite's power (79 of the 200,
32 of them ``create_refund``; review round 1: the latest-200 selection held 9).

Stale-state regression: ``StaleTracker`` hands ``normalize`` the session context from before the
previous ``observe()`` (it observes each call one call late), the bug class of a gate that reads a
cached snapshot one event old. The same G-a run through it must false-block at least once,
proving the suite can see that bug (measured: 27 / 200, all ``create_refund`` whose charge id
``get_ticket`` returned one call earlier because the lookup was skipped). List-valued ids never
reach the block branch there: ``mass_update``'s ``record_ids[]`` always come from the
``run_soql`` call just before it, so under the stale tracker the key's own flow history learns
them as missing (26 / 491 present) and the gate leaves it to the distribution check; only a key
whose stale misses are rare (``create_refund``: the lookup is skipped ~5% of the time, history
>= 0.9) blocks.
"""

from __future__ import annotations

import functools
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import pytest

from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.core.config import EngineConfig
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, Decision, DetectorResult, Outcome, RawCall
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker, flatten, normalize
from agentfw.core.patterns import is_id_like
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.gate import SCHEMA_REGIMES
from agentfw.synth.generate import GenerateConfig, generate
from agentfw.synth.load import CATALOG_DIR, PERSONAS_DIR, TASKS_DIR, load_catalog, load_personas

pytestmark = pytest.mark.slow

N_SESSIONS = 200
#: Measured floors (seed 0): stale-sensitive last calls in the evaluation set, in total and on
#: ``create_refund`` (the key whose stale misses are rare enough for its history to stay above
#: ``gate_flow_rate``, so a stale context blocks there). They pin the suite's power.
MIN_STALE_SENSITIVE = 79
MIN_STALE_SENSITIVE_REFUND = 32
DAYS = 10
GATE_CFG = EngineConfig(gate=True, flow_ledger=True)

World = tuple[tuple[RawCall, ...], Mapping[str, ToolHint], tuple[str, ...], Counter[str]]


class StaleTracker(SessionTracker):
    """A tracker whose context is one call stale: each call is observed only when the next call of
    its session is, so ``normalize`` sees the session as it was before the previous
    ``observe()`` (ids, ledger, prior tools, ring)."""

    def __init__(self, k: int = 4, max_ids: int = 2000) -> None:
        super().__init__(k, max_ids)
        self._pending: dict[str, ActionEnvelope] = {}

    def observe(self, e: ActionEnvelope) -> None:
        prev = self._pending.get(e.session_id)
        if prev is not None:
            super().observe(prev)
        self._pending[e.session_id] = e


def _arg_ids(r: RawCall) -> set[str]:
    flat, _ = flatten(r.args)
    return {v for vals in flat.values() for v in vals if isinstance(v, str) and is_id_like(v)}


def _pick(calls: list[RawCall], irreversible: frozenset[str]) -> tuple[int, bool] | None:
    """``(index, stale_sensitive)`` of the session's evaluation call, or None.

    A destructive call is *valid* when every id-like arg value was returned by an earlier call of
    the session or is a task input. It is *stale-sensitive* when, in addition, one of those ids was
    first returned by the call just before it, so a context one call stale cannot see it. The
    last stale-sensitive valid call is preferred, else the last valid one."""
    known = set(calls[0].task_ids)  # task inputs + responses of the calls before this one
    before_prev = set(known)  # the same, without the previous call's response
    last: tuple[int, bool] | None = None
    sensitive: int | None = None
    for i, c in enumerate(calls):
        ids = _arg_ids(c)
        if c.tool_id in irreversible and ids <= known:
            last = (i, False)
            if not ids <= before_prev:
                sensitive = i
        before_prev = set(known)
        if c.response_meta is not None:
            known |= set(c.response_meta.ids)
    return (sensitive, True) if sensitive is not None else last


@functools.cache
def g_a_world() -> World:
    """(raw calls, schema hints, the evaluation sessions' last envelope ids in session order, the
    stale-sensitive last calls counted by tool)."""
    catalog = load_catalog()
    destructive = frozenset(
        t for t, s in catalog.items() if s.annotations is not None and s.annotations.destructive
    )
    personas = sorted(
        p.agent_id for p in load_personas() if p.id_flow_rate == 1.0 and destructive & set(p.tools)
    )
    cfg = GenerateConfig(
        seed=0,
        personas_dir=PERSONAS_DIR,
        tasks_dir=TASKS_DIR,
        catalog_dir=CATALOG_DIR,
        persona_ids=personas,
        days=DAYS,
        attack_rate=0.0,
        control_rate=0.0,
        families=[],
    )
    with tempfile.TemporaryDirectory() as d:
        generate(cfg, Path(d))
        raws, labels, _ = load_dir(d)
        hints = load_schema_hints(d)
    assert all(lab.label == "benign" and lab.control_type is None for lab in labels.values())
    sessions: dict[str, list[RawCall]] = defaultdict(list)
    for r in raws:
        sessions[r.session_id].append(r)
    eligible: list[tuple[bool, int, str, int]] = []
    for sid, calls in sessions.items():
        calls.sort(key=lambda r: r.seq_no)
        pick = _pick(calls, destructive)
        if pick is not None:
            eligible.append((pick[1], calls[0].ts_unix_ms, sid, pick[0]))
    # stale-sensitive sessions first, latest (warmest) first; then the latest others
    ranked = sorted(eligible, reverse=True)[:N_SESSIONS]
    assert len(ranked) == N_SESSIONS
    chosen = sorted(ranked, key=lambda t: (t[1], t[2]))
    cut = {sid: idx for _, _, sid, idx in chosen}
    out: list[RawCall] = []
    for sid, calls in sessions.items():
        out.extend(calls[: cut[sid] + 1] if sid in cut else calls)
    last = tuple(sessions[sid][idx].envelope_id for _, _, sid, idx in chosen)
    sensitive = Counter(sessions[sid][idx].tool_id for flag, _, sid, idx in chosen if flag)
    return tuple(out), hints, last, sensitive


def run_g_a(tracker_cls: type[SessionTracker] = SessionTracker) -> list[DetectorResult]:
    """Replay the G-a world through a gate engine; the results of the evaluation sessions' last
    (valid irreversible) calls, in session order."""
    raws, hints, last, _ = g_a_world()
    eng = Engine(GATE_CFG, hints=hints)
    tracker = tracker_cls()
    results: dict[str, DetectorResult] = {}
    for raw in sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no)):
        e = normalize(raw, tracker)
        res = eng.score(e)
        results[e.envelope_id] = res
        flagged = res.decision is not Decision.ALLOW  # all benign: the reviewer approves
        eng.update(e, Outcome.APPROVED if flagged else Outcome.ALLOWED)
        tracker.observe(e)
    return [results[eid] for eid in last]


def count_blocks(results: list[DetectorResult]) -> int:
    return sum(r.decision is Decision.BLOCK for r in results)


def test_g_a_no_false_blocks_on_valid_irreversible_calls() -> None:
    sensitive = g_a_world()[3]
    assert sensitive.total() >= MIN_STALE_SENSITIVE
    assert sensitive["stripe.create_refund"] >= MIN_STALE_SENSITIVE_REFUND
    last_call_results = run_g_a()
    assert len(last_call_results) == N_SESSIONS
    for r in last_call_results:  # the gate really ran, with its prerequisite check
        assert r.trajectory is not None and r.trajectory.irreversible
        assert r.trajectory.gate is not None and r.trajectory.regime in SCHEMA_REGIMES
    assert any(r.trajectory and r.trajectory.regime == "history+schemas" for r in last_call_results)
    blocks = [r for r in last_call_results if r.decision is Decision.BLOCK]
    assert blocks == []


def test_stale_snapshot_fails_g_a() -> None:
    results = run_g_a(tracker_cls=StaleTracker)
    assert count_blocks(results) >= 1
    for r in results:  # every false block is the gate's prerequisite block
        if r.decision is Decision.BLOCK:
            assert r.trajectory is not None and r.trajectory.gate is not None
            assert r.trajectory.gate.action == "block"
            assert Flag.PREREQ_MISSING in r.trajectory.flags
