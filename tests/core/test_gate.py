"""Pre-execution gate for irreversible calls (F4, Task 3.1; ``core.trajectory.gate``).

The scenario: an agent that always looks a charge up (``stripe.retrieve_charge``) before it
refunds it (``stripe.create_refund``), so the refund's ``charge_id`` always has a ledger producer.
The allow-side acceptance (G-a, 200 benign synth sessions) and the stale-state regression live in
``tests/acceptance/test_gate_allow_side.py``.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentfw.bench.ci_regress import CI_DATASET
from agentfw.bench.dataset import load_dir, load_schema_hints
from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.engine import Engine
from agentfw.core.envelope import (
    ActionEnvelope,
    Decision,
    DetectorResult,
    FieldType,
    IdSnapshot,
    Outcome,
    RawCall,
    ResponseMeta,
    TrajectoryInfo,
)
from agentfw.core.hints import ToolAnnotations, ToolHint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_flow, key_sess
from agentfw.core.trajectory.decide import decide
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.gate import GateOutcome, gate
from agentfw.core.trajectory.session import SessionRecord
from agentfw.synth.generate import generate
from tests.factories import make_raw, run_session

T0 = 1_700_000_000_000
STEP = 600_000  # 10 minutes between calls: velocity stays quiet
AGENT = "a1"
LOOKUP = "stripe.retrieve_charge"
REFUND = "stripe.create_refund"
_READ = ToolAnnotations(read_only=True, destructive=False, idempotent=True, open_world=False)
_MONEY_OUT = ToolAnnotations(read_only=False, destructive=True, idempotent=False, open_world=True)
HINTS = {
    "zendesk.get_ticket": ToolHint(annotations=_READ, has_schema=True),
    LOOKUP: ToolHint(annotations=_READ, has_schema=True),
    REFUND: ToolHint(annotations=_MONEY_OUT, has_schema=True),
}
#: The same annotations with no captured schema behind them (regime ``history_only``).
NO_SCHEMA_HINTS = {t: dataclasses.replace(h, has_schema=False) for t, h in HINTS.items()}
GATE = EngineConfig(gate=True, flow_ledger=True)
FABRICATED = "ch_fabricated01"


class Clock:
    def __init__(self) -> None:
        self.ts = T0

    def tick(self) -> int:
        self.ts += STEP
        return self.ts


def charge(i: int) -> str:
    return f"ch_{i:06d}abcd"


def lookup(clock: Clock, sid: str, seq: int, ch: str) -> RawCall:
    return make_raw(
        session_id=sid,
        agent_id=AGENT,
        seq_no=seq,
        ts_unix_ms=clock.tick(),
        tool_id=LOOKUP,
        args={"charge_id": ch},
        response_meta=ResponseMeta(ids=(ch,)),
    )


def refund(
    clock: Clock,
    sid: str,
    seq: int,
    ch: str,
    amount: float = 30.0,
    task_ids: tuple[str, ...] = (),
) -> RawCall:
    return make_raw(
        session_id=sid,
        agent_id=AGENT,
        seq_no=seq,
        ts_unix_ms=clock.tick(),
        tool_id=REFUND,
        args={"charge_id": ch, "amount": amount},
        response_meta=ResponseMeta(ids=(f"re_{seq:02d}{sid}xxxx",)),
        task_ids=task_ids,
    )


def warm(
    cfg: EngineConfig = GATE,
    hints: dict[str, ToolHint] | None = None,
    n: int = 40,
    flowing: bool = True,
) -> tuple[Engine, Clock]:
    """``n`` benign sessions: look a charge up, then refund it (``flowing=False``: the refund's
    charge never appeared earlier in its session, so the key's presence rate is 0)."""
    eng = Engine(cfg, hints=HINTS if hints is None else hints)
    clock = Clock()
    for i in range(n):
        sid = f"w{i:03d}"
        ch = charge(i)
        amount = 20.0 + (i % 7) * 3.5
        calls = [lookup(clock, sid, 0, ch), refund(clock, sid, 1, ch, amount)]
        if not flowing:
            calls = [lookup(clock, sid, 0, charge(1000 + i)), refund(clock, sid, 1, ch, amount)]
        run_session(eng, calls)
    return eng, clock


def score_alone(eng: Engine, raw: RawCall, tracker: SessionTracker | None = None) -> DetectorResult:
    """Score one call (pure); its session context comes from ``tracker`` (empty by default)."""
    return eng.score(normalize(raw, SessionTracker() if tracker is None else tracker))


def gate_info(r: DetectorResult) -> GateOutcome:
    assert r.trajectory is not None and r.trajectory.gate is not None
    return r.trajectory.gate


# -- G-b: missing producer --------------------------------------------------------------------


def test_g_b_missing_producer_blocks() -> None:
    eng, clock = warm()
    r = score_alone(eng, refund(clock, "fab", 0, FABRICATED))
    assert r.trajectory is not None and r.trajectory.irreversible
    assert r.trajectory.regime == "history+schemas"
    assert r.decision is Decision.BLOCK
    assert gate_info(r) == GateOutcome("block", LOOKUP, "charge_id")
    assert {Flag.PREREQ_MISSING, Flag.BLOCK} <= r.trajectory.flags


def test_valid_refund_passes() -> None:
    eng, clock = warm()
    out = run_session(
        eng, [lookup(clock, "ok", 0, charge(500)), refund(clock, "ok", 1, charge(500))]
    )
    assert gate_info(out[1]) == GateOutcome("pass", None, None)
    assert out[1].decision is not Decision.BLOCK
    assert out[0].trajectory is not None and out[0].trajectory.gate is None  # reversible: no gate


def test_task_input_seeded_id_passes() -> None:
    """An id the task itself supplied has a ``task_input`` ledger ref: present, no block."""
    eng, clock = warm()
    r = score_alone(eng, refund(clock, "seed", 0, FABRICATED, task_ids=(FABRICATED,)))
    assert r.trajectory is not None and r.trajectory.regime == "history+schemas"
    assert gate_info(r).action != "block" and gate_info(r).missing_path is None
    assert r.decision is not Decision.BLOCK


def test_cold_key_missing_id_escalates() -> None:
    """T-11: below ``flow_min_obs`` observations a missing id escalates, never blocks."""
    eng, clock = warm(n=12)  # charge_id typed ID (>= type_min_obs) but < flow_min_obs
    r = score_alone(eng, refund(clock, "fab", 0, FABRICATED))
    assert r.trajectory is not None and r.trajectory.regime == "schemas_only"
    assert gate_info(r).action == "escalate"
    assert gate_info(r).missing_path == "charge_id"
    assert r.decision is Decision.REQUIRE_APPROVAL
    assert r.trajectory is not None and Flag.PREREQ_MISSING not in r.trajectory.flags


def test_warm_key_below_gate_flow_rate_does_not_block() -> None:
    """A warm key whose ids historically do not flow (r_flow < gate_flow_rate) never blocks."""
    eng, clock = warm(flowing=False)
    r = score_alone(eng, refund(clock, "fab", 0, FABRICATED))
    assert gate_info(r).action != "block"
    assert r.decision is not Decision.BLOCK


def test_gate_flow_rate_is_the_bar() -> None:
    """``present / total`` exactly at ``gate_flow_rate`` blocks; just under it does not."""
    eng, clock = warm(n=40)
    rec = eng.store.get(key_flow(AGENT, REFUND, "charge_id"))
    assert rec is not None
    rec.present, rec.total = 36, 40  # type: ignore[attr-defined]  # 0.90
    assert gate_info(score_alone(eng, refund(clock, "f1", 0, FABRICATED))).action == "block"
    rec.present = 35  # type: ignore[attr-defined]  # 0.875
    assert gate_info(score_alone(eng, refund(clock, "f2", 0, FABRICATED))).action != "block"


# -- G-c: distribution ------------------------------------------------------------------------


def test_g_c_anomalous_distribution_escalates_not_blocks() -> None:
    eng, clock = warm()
    off, _ = warm(cfg=GATE.replace(gate=False))
    sid = "big"
    ch = charge(777)
    tracker_on, tracker_off = SessionTracker(), SessionTracker()
    for eng_, tr in ((eng, tracker_on), (off, tracker_off)):
        run_session(eng_, [lookup(clock, sid, 0, ch)], tracker=tr)
    spike = refund(clock, sid, 1, ch, amount=9_000_000.0)
    r_off = score_alone(off, spike, tracker_off)
    assert r_off.decision is Decision.BLOCK  # the fast path alone would block it
    r = score_alone(eng, spike, tracker_on)
    assert r.trajectory is not None
    assert r.fused_score >= r.trajectory.threshold_eff
    assert gate_info(r) == GateOutcome("escalate", None, None)
    assert r.decision is Decision.REQUIRE_APPROVAL


# -- G-d: no schemas --------------------------------------------------------------------------


@pytest.mark.parametrize("hints", [NO_SCHEMA_HINTS, {}], ids=["annotations-only", "no-hints"])
def test_g_d_no_schemas_distribution_only(hints: dict[str, ToolHint]) -> None:
    """Regime ``history_only``: the prerequisite check is skipped, never a prereq block. The
    fabricated id still raises the sequence flow score, so the distribution check escalates."""
    eng, clock = warm(hints=hints)
    r = score_alone(eng, refund(clock, "fab", 0, FABRICATED))
    assert r.trajectory is not None and r.trajectory.irreversible
    assert r.trajectory.regime == "history_only"
    assert r.fused_score >= r.trajectory.threshold_eff
    assert gate_info(r) == GateOutcome("escalate", None, None)
    assert r.decision is Decision.REQUIRE_APPROVAL
    assert Flag.PREREQ_MISSING not in r.trajectory.flags


def test_g_d_bare_regime_skips_the_prerequisite_check() -> None:
    eng, clock = warm(hints={}, n=12)  # cold and schema-less: "bare"
    r = score_alone(eng, refund(clock, "fab", 0, FABRICATED))
    assert r.trajectory is not None and r.trajectory.regime == "bare"
    assert r.trajectory.irreversible  # a learned write with b >= irreversible_b (T-35 rule)
    assert gate_info(r).action != "block" and gate_info(r).missing_path is None
    assert r.decision is not Decision.BLOCK


# -- reversible calls -------------------------------------------------------------------------


def test_reversible_never_blocked() -> None:
    """With ``gate`` a reversible call the bands would block is capped at ``REQUIRE_APPROVAL``."""
    listing = "stripe.list_charges"
    hints = {**HINTS, listing: ToolHint(annotations=_READ, has_schema=True)}
    clock = Clock()

    def list_call(sid: str, limit: float) -> RawCall:
        return make_raw(
            session_id=sid,
            agent_id=AGENT,
            seq_no=0,
            ts_unix_ms=clock.tick(),
            tool_id=listing,
            args={"limit": limit},
            response_meta=ResponseMeta(item_count=4),
        )

    history = [list_call(f"l{i:03d}", 10.0 + i % 3) for i in range(60)]
    engines = {on: Engine(GATE.replace(gate=on), hints=hints) for on in (False, True)}
    for eng in engines.values():
        run_session(eng, history)
    spike = list_call("spike", 50_000_000.0)
    r_off, r = score_alone(engines[False], spike), score_alone(engines[True], spike)
    assert r_off.decision is Decision.BLOCK
    assert r.trajectory is not None and not r.trajectory.irreversible
    assert r.trajectory.gate is None  # the gate runs on irreversible calls only
    assert r.decision is Decision.REQUIRE_APPROVAL


def _info(theta: float, irreversible: bool = False) -> TrajectoryInfo:
    return TrajectoryInfo("history+schemas", None, irreversible, theta)


@given(
    fused=st.floats(min_value=0.0, max_value=1.0) | st.just(math.nan),
    theta=st.floats(min_value=0.0, max_value=1.0),
)
def test_reversible_capped_at_require_approval(fused: float, theta: float) -> None:
    cfg = EngineConfig(gate=True)
    base, _ = decide(fused, _info(theta), EngineConfig())
    d, out = decide(fused, _info(theta), cfg)
    assert d is not Decision.BLOCK
    assert d is (Decision.REQUIRE_APPROVAL if base is Decision.BLOCK else base)
    assert out.gate is None and out.flags == frozenset()


@given(
    fused=st.floats(min_value=0.0, max_value=1.0),
    theta=st.floats(min_value=0.0, max_value=1.0),
)
def test_gate_and_provisional_compose(fused: float, theta: float) -> None:
    """T-41: with both switches a reversible band ``BLOCK`` is provisional (``ALLOW``), not
    capped to ``REQUIRE_APPROVAL``; the approval band is left alone by both."""
    both = EngineConfig(gate=True, provisional=True)
    prov_only = EngineConfig(provisional=True)
    assert decide(fused, _info(theta), both) == decide(fused, _info(theta), prov_only)
    d, out = decide(fused, _info(theta), both)
    assert d is not Decision.BLOCK
    if fused >= theta:
        assert d is Decision.ALLOW and Flag.PROVISIONAL in out.flags


def test_decide_needs_the_call_for_an_irreversible_gate() -> None:
    with pytest.raises(ValueError, match="gate"):
        decide(0.1, _info(0.6, irreversible=True), EngineConfig(gate=True))


# -- fresh state ------------------------------------------------------------------------------


class _IdTypes(TypeView):
    """``charge_id`` is an id, everything else numeric (no inference behind it)."""

    def __init__(self) -> None:
        super().__init__(TypeInference(DictStore(), EngineConfig()), AGENT, REFUND)

    def __call__(self, path: str) -> FieldType:
        return FieldType.ID if path == "charge_id" else FieldType.NUMERIC


def _fab_env(sid: str = "fab") -> ActionEnvelope:
    return normalize(refund(Clock(), sid, 0, FABRICATED), SessionTracker())


def test_gate_reads_the_store_at_call_time() -> None:
    """Step 1: no snapshot is kept between calls -- a change to the flow record is seen by the
    very next gate call, and ``gate`` itself never writes."""
    eng, _ = warm()
    seq, e, types = eng._sequence, _fab_env(), _IdTypes()
    before = eng.state_bytes()
    first = gate(e, types, seq, 0.0, 0.6, GATE, regime="history+schemas")
    assert first.action == "block"
    rec = eng.store.get(key_flow(AGENT, REFUND, "charge_id"))
    assert rec is not None
    rec.present = 0  # type: ignore[attr-defined]
    assert gate(e, types, seq, 0.0, 0.6, GATE, regime="history+schemas").action == "pass"
    rec.present = rec.total  # type: ignore[attr-defined]
    assert gate(e, types, seq, 0.0, 0.6, GATE, regime="history+schemas") == first
    assert eng.state_bytes() == before


def test_non_id_like_value_is_never_missing() -> None:
    """T-41 (controller ruling): only an id-like value is an artifact the ledger can track. A
    value inference types ID but that is not id-like (free text) never escalates or blocks, on a
    cold key or a warm one; an id-like value at the same path does."""
    types = _IdTypes()
    text = dataclasses.replace(
        _fab_env(), flat_args={"charge_id": ("refund please",), "amount": (30.0,)}
    )
    for n in (0, 40):  # cold (no history) and warm, flowing
        eng = warm(n=n)[0] if n else Engine(GATE, hints=HINTS)
        seq = eng._sequence
        assert gate(text, types, seq, 0.1, 0.6, GATE, regime="history+schemas") == GateOutcome(
            "pass", None, None
        )
        assert gate(_fab_env(), types, seq, 0.1, 0.6, GATE, regime="history+schemas").action == (
            "block" if n else "escalate"
        )


def test_gate_steps_in_order() -> None:
    """A block beats an escalation; the distribution check runs when nothing is missing."""
    eng, _ = warm()
    seq, types = eng._sequence, _IdTypes()
    e = _fab_env()
    assert gate(e, types, seq, 0.99, 0.6, GATE, regime="schemas_only").action == "block"
    ok = dataclasses.replace(e, session_ids_seen=IdSnapshot(frozenset({FABRICATED}), None))
    assert gate(ok, types, seq, 0.59, 0.6, GATE, regime="history+schemas").action == "pass"
    assert gate(ok, types, seq, 0.6, 0.6, GATE, regime="history+schemas") == GateOutcome(
        "escalate", None, None
    )
    assert gate(e, types, seq, 0.1, 0.6, GATE, regime="bare").action == "pass"


def test_bloom_only_id_is_present() -> None:
    """Review Focus 4: past the ledger's exact capacity an id is known only through the Bloom
    overflow (no ``LedgerRef``, no producer); it is present, so no prerequisite block."""
    eng, clock = warm()
    tracker = SessionTracker(max_ids=2)
    target = charge(901)
    fillers = (f"tk_{1:06d}fill", f"tk_{2:06d}fill")
    first = make_raw(
        session_id="bl",
        agent_id=AGENT,
        seq_no=0,
        ts_unix_ms=clock.tick(),
        tool_id="zendesk.get_ticket",
        args={"ticket_id": fillers[0]},
        response_meta=ResponseMeta(ids=fillers),
    )
    run_session(eng, [first, lookup(clock, "bl", 1, target)], tracker=tracker)
    led = tracker.ledger("bl")
    assert led is not None and led.degraded and led.lookup(target) is None and target in led
    e = normalize(refund(clock, "bl", 2, target), tracker)
    assert e.ledger_refs == () and target in e.session_ids_seen
    r = eng.score(e)
    assert gate_info(r).action == "pass"
    assert r.decision is not Decision.BLOCK
    # the same call without the Bloom membership is blocked
    bare = dataclasses.replace(e, session_ids_seen=IdSnapshot(frozenset(), None))
    assert gate_info(eng.score(bare)).action == "block"


def _bad_lookup(clock: Clock, sid: str, seq: int, ch: str) -> RawCall:
    """A lookup of ``ch`` that errors and returns nothing (the fabricated id has no producer)."""
    return make_raw(
        session_id=sid,
        agent_id=AGENT,
        seq_no=seq,
        ts_unix_ms=clock.tick(),
        tool_id=LOOKUP,
        args={"charge_id": ch},
        response_meta=ResponseMeta(ids=(), error=True),
    )


def test_fabricated_id_laundered_through_an_earlier_calls_args_still_blocks() -> None:
    """T-59 (final review I2, the reviewer's probe): using a fabricated id in an earlier call's
    args gives it a ledger entry with source ``"args"`` -- no producer -- and must not make it
    present. Before T-59 the refund passed the gate and was allowed."""
    eng, clock = warm()
    assert gate_info(score_alone(eng, refund(clock, "fab", 0, FABRICATED))).action == "block"
    tracker = SessionTracker()
    run_session(eng, [_bad_lookup(clock, "lau", 0, FABRICATED)], tracker=tracker)
    e = normalize(refund(clock, "lau", 1, FABRICATED), tracker)
    assert [r.source for r in e.ledger_refs] == ["args"] and FABRICATED in e.session_ids_seen
    r = eng.score(e)
    assert gate_info(r) == GateOutcome("block", LOOKUP, "charge_id")
    assert r.decision is Decision.BLOCK


def test_lookup_that_returns_the_id_grounds_it() -> None:
    """T-59: an id first seen in a lookup's args that the lookup's own response returns gets that
    lookup as its producer (source ``"response"``), so the refund after it passes; the same
    lookup erroring with nothing returned leaves it ungrounded (above)."""
    eng, clock = warm()
    tracker = SessionTracker()
    ch = charge(777)
    run_session(eng, [lookup(clock, "af", 0, ch)], tracker=tracker)  # args, then response: ch
    e = normalize(refund(clock, "af", 1, ch), tracker)
    assert [(r.source, r.producer_seq, r.producer_tool) for r in e.ledger_refs] == [
        ("response", 0, LOOKUP)
    ]
    assert gate_info(eng.score(e)).action == "pass"


# -- switches and purity ----------------------------------------------------------------------


def test_gate_off_reports_no_gate_and_changes_nothing() -> None:
    eng, clock = warm(cfg=GATE.replace(gate=False))
    r = score_alone(eng, refund(clock, "fab", 0, FABRICATED))
    assert r.trajectory is not None and r.trajectory.gate is None
    assert Flag.PREREQ_MISSING not in r.trajectory.flags


def test_gate_flow_rate_validated_only_when_on() -> None:
    with pytest.raises(ValueError, match="gate_flow_rate"):
        Engine(EngineConfig(gate=True, gate_flow_rate=1.5))
    Engine(EngineConfig(gate_flow_rate=1.5))  # gate off: never read, never rejected


def test_update_without_cached_score_records_the_gate_block() -> None:
    """``update`` on an envelope it did not just score decides it afresh (T-34) and records the
    gate's ``PREREQ_MISSING`` and ``BLOCK`` in the session window."""
    eng, clock = warm(cfg=GATE.replace(sessions=True))
    e = normalize(refund(clock, "fab", 0, FABRICATED), SessionTracker())
    eng.update(e, Outcome.BLOCKED)
    rec = eng.store.get(key_sess("fab"))
    assert isinstance(rec, SessionRecord)
    seq_no, names = rec.window[-1]
    assert seq_no == 0 and {"BLOCK", "PREREQ_MISSING"} <= set(names)


class _GateCounted(Engine):
    def __init__(self, *a: object, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.actions: dict[str, int] = {}
        self.missing = 0  # gate outcomes naming a missing path (the prerequisite check fired)

    def score(self, e: ActionEnvelope) -> DetectorResult:
        r = super().score(e)
        assert r.trajectory is not None
        g = r.trajectory.gate
        key = "none" if g is None else g.action
        self.actions[key] = self.actions.get(key, 0) + 1
        self.missing += g is not None and g.missing_path is not None
        return r


def test_strict_replay_of_ci_dataset_with_gate(tmp_path: Path) -> None:
    """``score`` stays pure with ``gate`` (and ``flow_ledger``) on, with and without the
    dataset's schemas (with them the prerequisite check runs)."""
    with_dir, bare = tmp_path / "with", tmp_path / "bare"
    generate(CI_DATASET, with_dir)
    shutil.copytree(with_dir, bare)
    (bare / "schemas.jsonl").unlink()
    for d, schemas in ((with_dir, True), (bare, False)):
        raws, labels, _ = load_dir(d)
        eng = _GateCounted(GATE, hints=load_schema_hints(d))
        rows = replay(raws, labels, eng, SplitSpec(), strict=True, outcomes="review")
        assert len(rows) == len(raws)
        assert eng.actions.get("pass", 0) > 0  # the gate ran on irreversible calls
        if schemas:  # step 2 ran and found missing ids
            assert eng.missing > 0
        else:  # G-d: no schemas, no prerequisite check
            assert eng.missing == 0
