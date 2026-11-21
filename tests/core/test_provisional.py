"""Provisional lifecycle (F6, Task 2.2): grace TTL, validation, absorption, exploration budget,
and the ``decide`` pipeline with every switch off."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import orjson
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentfw.bench.ci_regress import ci_dataset
from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import (
    ActionEnvelope,
    Decision,
    DetectorResult,
    DetectorScore,
    LedgerRef,
    Outcome,
    RawCall,
    ResponseMeta,
    TrajectoryInfo,
)
from agentfw.core.hints import ToolAnnotations, ToolHint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.state import DictStore, key_sess
from agentfw.core.trajectory.decide import decide
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.provisional import ProvEntry, advance
from agentfw.core.trajectory.session import SessionRecord
from tests.core.strategies import envelope_streams, outcomes
from tests.factories import make_env, make_raw, run_session

T0 = 1_700_000_000_000
STEP = 600_000  # 10 minutes between calls: velocity stays quiet
READ = "kb.get_doc"
DELETE = "kb.delete_doc"
DOCS = [f"doc_{i:04d}aaaa" for i in range(5)]  # id-like (8+ chars after the prefix)
NOVEL = "doc_9999zzzz"
PROV = EngineConfig(provisional=True)


# -- scenario helpers ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.ts = T0

    def tick(self) -> int:
        self.ts += STEP
        return self.ts


def call(
    clock: Clock,
    session: str,
    seq: int,
    doc: str,
    *,
    tool: str = READ,
    ids: tuple[str, ...] = (),
    arg: str = "doc_id",
    item_count: int | None = 1,
) -> RawCall:
    return make_raw(
        session_id=session,
        seq_no=seq,
        ts_unix_ms=clock.tick(),
        tool_id=tool,
        args={arg: doc},
        response_meta=ResponseMeta(item_count=item_count, ids=ids),
    )


def warm_engine(
    cfg: EngineConfig = PROV, tool: str = READ, item_count: int | None = 1, **kw: Any
) -> tuple[Engine, SessionTracker, Clock]:
    """An engine that has seen ``tool`` over the five DOCS in 12 sessions of 6 calls, so a never
    seen ``doc_id`` scores novelty 1.0 (BLOCK without provisional)."""
    eng, tr, clock = Engine(cfg, **kw), SessionTracker(), Clock()
    for s in range(12):
        raws = [
            call(clock, f"warm{s}", i, DOCS[(s + i) % 5], tool=tool, item_count=item_count)
            for i in range(6)
        ]
        run_session(eng, raws, tr)
    return eng, tr, clock


def per_call_flags(results: list[DetectorResult]) -> list[frozenset[Flag]]:
    out = []
    for r in results:
        assert r.trajectory is not None
        out.append(r.trajectory.flags)
    return out


def novel_then_benign(clock: Clock, n_benign: int, session: str = "t") -> list[RawCall]:
    return [call(clock, session, 0, NOVEL)] + [
        call(clock, session, i, DOCS[i % 5]) for i in range(1, n_benign + 1)
    ]


def sess(eng: Engine, session: str = "t") -> SessionRecord:
    rec = eng.store.get(key_sess(session))
    assert isinstance(rec, SessionRecord)
    return rec


def snapshot(eng: Engine) -> list[tuple[str, bytes]]:
    opt = orjson.OPT_SORT_KEYS
    return sorted((k, orjson.dumps(r.to_dict(), option=opt)) for k, r in eng.store.items())


# -- F6 lifecycle through the engine ------------------------------------------------------------


def test_hallucinated_read_resolves_unresolved_at_ttl() -> None:  # F6 acceptance, latency 3
    eng, tr, clock = warm_engine()
    res = run_session(eng, novel_then_benign(clock, 3), tr)
    flags = per_call_flags(res)
    assert Flag.PROVISIONAL in flags[0] and Flag.BLOCK not in flags[0]
    assert res[0].decision is Decision.ALLOW
    assert [Flag.UNRESOLVED in f for f in flags[1:]] == [False, False, True]
    [entry] = sess(eng).exploration
    assert entry.state == "unresolved" and entry.seq_no == 0 and entry.ttl == 0
    assert entry.field == "doc_id" and entry.server == "kb"
    assert entry.value_hash is not None and NOVEL not in repr(entry)  # never the raw value
    assert sess(eng).unresolved_mass == pytest.approx(res[0].fused_score)


def test_same_call_blocks_without_the_switch() -> None:
    eng, tr, clock = warm_engine(EngineConfig())
    res = run_session(eng, novel_then_benign(clock, 3), tr)
    assert res[0].decision is Decision.BLOCK
    assert all(Flag.PROVISIONAL not in f and Flag.UNRESOLVED not in f for f in per_call_flags(res))


@pytest.mark.parametrize("refused", [Outcome.BLOCKED, Outcome.DENIED])
def test_refused_provisional_call_opens_no_entry(refused: Outcome) -> None:  # T-36 ruling
    eng, tr, clock = warm_engine()
    raws = novel_then_benign(clock, 5)
    e0 = normalize(raws[0], tr)
    r0 = eng.score(e0)
    assert r0.trajectory is not None and Flag.PROVISIONAL in r0.trajectory.flags
    eng.update(e0, refused)  # refused outside the engine: never executed
    tr.observe(e0)
    assert sess(eng).window == [(0, sorted(f.value for f in r0.trajectory.flags))]
    assert sess(eng).exploration == [] and sess(eng).unresolved_mass == 0.0
    res = run_session(eng, raws[1:], tr)
    assert not any(Flag.UNRESOLVED in f or Flag.OFF_TRACK in f for f in per_call_flags(res))
    assert sess(eng).exploration == [] and sess(eng).unresolved_mass == 0.0


def test_consumed_artifact_validates() -> None:
    eng, tr, clock = warm_engine()
    produced = "doc_prod77bb"
    raws = [
        call(clock, "t", 0, NOVEL, ids=(produced,)),
        call(clock, "t", 1, produced, tool="kb.summarize", arg="ref"),  # consumes call 0's output
        call(clock, "t", 2, DOCS[2]),
        call(clock, "t", 3, DOCS[3]),
        call(clock, "t", 4, DOCS[4]),
    ]
    res = run_session(eng, raws, tr)
    flags = per_call_flags(res)
    assert Flag.PROVISIONAL in flags[0]
    assert not any(Flag.UNRESOLVED in f for f in flags)
    states = {p.seq_no: p.state for p in sess(eng).exploration}
    assert 0 not in states  # validated at call 1, pruned at the next advance
    assert sess(eng).unresolved_mass == 0.0


def test_validation_is_visible_on_the_record_right_after() -> None:
    eng, tr, clock = warm_engine()
    produced = "doc_prod77bb"
    raws = [
        call(clock, "t", 0, NOVEL, ids=(produced,)),
        call(clock, "t", 1, produced, tool="kb.summarize", arg="ref"),
    ]
    run_session(eng, raws, tr)
    assert [(p.seq_no, p.state) for p in sess(eng).exploration] == [(0, "validated")]


def test_vocabulary_shift_absorbs() -> None:
    eng, tr, clock = warm_engine()
    raws = [call(clock, "t", 0, NOVEL), call(clock, "t", 1, NOVEL)] + [
        call(clock, "t", i, DOCS[i]) for i in range(2, 5)
    ]
    res = run_session(eng, raws, tr)
    flags = per_call_flags(res)
    assert Flag.PROVISIONAL in flags[0]
    # learned after call 0 (allowed): the same value now scores benign on the same server
    assert res[1].fused_score < res[1].trajectory.threshold_eff  # type: ignore[union-attr]
    assert Flag.PROVISIONAL not in flags[1]
    assert not any(Flag.UNRESOLVED in f for f in flags)


def test_off_track_once_over_budget() -> None:
    eng, tr, clock = warm_engine()
    res = run_session(eng, novel_then_benign(clock, 11), tr)
    off = [Flag.OFF_TRACK in f for f in per_call_flags(res)]
    assert sum(off) == 1 and off.index(True) == 7  # the 8th call (prov_min_calls)
    rec = sess(eng)
    assert rec.off_track_raised and rec.unresolved_mass / rec.total_mass > 0.25


def test_single_abandoned_branch_raises_off_track_in_long_session_t38() -> None:
    # Pins today's spec-literal budget (total_mass = Σ fused), which conflicts with the F6
    # tolerance band (T-38, pending a user decision): benign fused scores are tiny, so ONE
    # unresolved entry keeps unresolved / total above prov_budget for dozens of calls and
    # raises a full OFF_TRACK (a severe flag) at the session's 8th call. A fix of T-38 must
    # change this test deliberately.
    eng, tr, clock = warm_engine()
    res = run_session(eng, novel_then_benign(clock, 29), tr)  # a 30-call session
    flags = per_call_flags(res)
    assert sum(Flag.UNRESOLVED in f for f in flags) == 1  # one abandoned branch only
    off = [Flag.OFF_TRACK in f for f in flags]
    assert sum(off) == 1 and off.index(True) == 7
    rec = sess(eng)
    assert rec.unresolved_mass / rec.total_mass > 0.25  # still over budget after 30 calls
    assert rec.off_track_raised


def test_one_call_abandoned_branch_costs_one_eighth() -> None:
    eng, tr, clock = warm_engine()
    res = run_session(eng, novel_then_benign(clock, 3), tr)
    density_after_one_unresolved = res[3].trajectory.density  # type: ignore[union-attr]
    assert density_after_one_unresolved == pytest.approx(0.125 / 6)


def test_irreversible_calls_never_provisional() -> None:
    hints = {READ: ToolHint(annotations=ToolAnnotations(destructive=True))}
    eng, tr, clock = warm_engine(hints=hints)
    res = run_session(eng, novel_then_benign(clock, 3), tr)
    assert res[0].trajectory is not None and res[0].trajectory.irreversible
    assert res[0].decision is Decision.BLOCK
    assert all(Flag.PROVISIONAL not in f for f in per_call_flags(res))
    assert sess(eng).exploration == []


def test_irreversible_write_without_blast_radius_mode_is_never_provisional() -> None:
    # br_mode "off": with provisional on, irreversibility still uses b (T-35)
    eng, tr, clock = warm_engine(tool=DELETE, item_count=None)
    assert eng.cfg.br_mode == "off"
    [r] = run_session(eng, [call(clock, "t", 0, NOVEL, tool=DELETE, item_count=None)], tr)
    assert r.trajectory is not None
    assert r.trajectory.irreversible and r.trajectory.blast_radius is None
    assert r.decision is Decision.BLOCK and Flag.PROVISIONAL not in r.trajectory.flags
    plain, tr2, clock2 = warm_engine(EngineConfig(), tool=DELETE, item_count=None)
    [p] = run_session(plain, [call(clock2, "t", 0, NOVEL, tool=DELETE, item_count=None)], tr2)
    assert p.trajectory is not None and not p.trajectory.irreversible  # T-31: annotation only


def test_require_approval_band_is_unchanged() -> None:
    info = TrajectoryInfo("bare", None, False, 0.6)
    assert decide(0.55, info, PROV) == (Decision.REQUIRE_APPROVAL, info)
    assert decide(0.1, info, PROV) == (Decision.ALLOW, info)
    d, out = decide(0.6, info, PROV)
    assert d is Decision.ALLOW and out.flags == frozenset({Flag.PROVISIONAL})
    irr = dataclasses.replace(info, irreversible=True)
    assert decide(0.6, irr, PROV) == (Decision.BLOCK, irr)


# -- switch off -------------------------------------------------------------------------------


@pytest.mark.parametrize("fused", [0.0, 0.39, 0.5, 0.55, 0.6, 0.61, 1.0, float("nan")])
@pytest.mark.parametrize("irr", [False, True])
def test_decide_off_is_the_calibrator_band(fused: float, irr: bool) -> None:
    eng = Engine()
    info = TrajectoryInfo("bare", None, irr, 0.6)
    assert decide(fused, info, eng.cfg) == (eng.calibrator.decide("a", fused, 0.6), info)


class _Checked(Engine):
    """Asserts at every score that ``decide`` agreed with ``Calibrator.decide`` on that state."""

    checked = 0

    def score(self, e: ActionEnvelope) -> DetectorResult:
        r = super().score(e)
        assert r.trajectory is not None
        want = self.calibrator.decide(e.agent_id, r.fused_score, r.trajectory.threshold_eff)
        assert r.decision is want
        self.checked += 1
        return r


@pytest.mark.parametrize("br_mode", ["off", "modulate"])
def test_switch_off_identical_decisions(br_mode: str) -> None:
    raws, labels = ci_dataset()
    eng = _Checked(EngineConfig(br_mode=br_mode))  # type: ignore[arg-type]
    rows = replay(raws, labels, eng, SplitSpec(), outcomes="review")
    assert eng.checked == len(raws) == len(rows)
    assert {r.decision for r in rows} >= {"allow", "block"}
    assert not any(r.flags and "PROVISIONAL" in r.flags.split(",") for r in rows)


# -- session gate, config ---------------------------------------------------------------------


def test_provisional_opens_the_session_gate() -> None:  # ruling R15, T-34
    assert Engine()._book is None
    assert Engine(PROV)._book is not None


@pytest.mark.parametrize(
    "bad", [{"prov_ttl": 0}, {"prov_budget": -0.1}, {"prov_budget": 1.5}, {"prov_min_calls": -1}]
)
def test_prov_settings_validated_only_when_on(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(bad))):
        Engine(PROV.replace(**bad))
    Engine(EngineConfig().replace(**bad))


def test_config_defaults() -> None:
    cfg = EngineConfig()
    assert (cfg.provisional, cfg.prov_ttl, cfg.prov_budget, cfg.prov_min_calls) == (
        False,
        3,
        0.25,
        8,
    )


# -- advance() unit rules ---------------------------------------------------------------------


def entry(seq: int = 0, **kw: Any) -> ProvEntry:
    base: dict[str, Any] = {
        "seq_no": seq,
        "ttl": 3,
        "mass": 0.9,
        "server": "kb",
        "field": "doc_id",
        "value_hash": "h1",
        "state": "provisional",
    }
    base.update(kw)
    return ProvEntry(**base)


def env_at(seq: int, tool: str = READ, refs: tuple[LedgerRef, ...] = ()) -> ActionEnvelope:
    e = make_env(session_id="u", seq_no=seq, ts_unix_ms=T0 + seq, tool_id=tool)
    return dataclasses.replace(e, ledger_refs=refs)


def nov(field: str = "doc_id", vh: str = "h1", score: float = 0.0) -> DetectorScore:
    return DetectorScore("novelty", field, score, {"value_hash": vh})


def ref(source: str, producer_seq: int) -> LedgerRef:
    return LedgerRef("doc_x1y2z3", "ref", producer_seq, READ, source)  # type: ignore[arg-type]


def test_validated_only_by_a_response_ref_of_its_own_call() -> None:  # ruling R7
    rec = SessionRecord(n_calls=1, exploration=[entry(0)])
    assert advance(rec, env_at(1, refs=(ref("response", 0),)), 0.1, 0.6, (), PROV) == set()
    assert rec.exploration[0].state == "validated"


@pytest.mark.parametrize(
    "bad",
    [
        ref("task_input", 0),  # task-input ids carry producer_seq 0: not call 0's artifact
        ref("args", 0),  # the first call that used an ungrounded id is not its producer
        ref("response", 5),  # another call's artifact
    ],
)
def test_other_refs_never_validate(bad: LedgerRef) -> None:  # ruling R7
    rec = SessionRecord(n_calls=1, exploration=[entry(0)])
    advance(rec, env_at(1, refs=(bad,)), 0.1, 0.6, (), PROV)
    assert rec.exploration[0].state == "provisional" and rec.exploration[0].ttl == 2


def test_absorbed_needs_same_server_same_value_and_a_benign_score() -> None:
    def run(tool: str, fused: float, per: tuple[DetectorScore, ...]) -> str:
        rec = SessionRecord(n_calls=1, exploration=[entry(0)])
        advance(rec, env_at(1, tool=tool), fused, 0.6, per, PROV)
        return rec.exploration[0].state

    assert run("kb.search", 0.1, (nov(),)) == "absorbed"  # same server, other tool
    assert run(READ, 0.1, (nov(),)) == "absorbed"
    assert run("crm.get", 0.1, (nov(),)) == "provisional"  # other server
    assert run(READ, 0.6, (nov(),)) == "provisional"  # not benign (fused >= θ_eff)
    assert run(READ, 0.1, (nov(vh="h2"),)) == "provisional"  # another value
    assert run(READ, 0.1, (nov(field="q"),)) == "provisional"  # another field
    assert run(READ, 0.1, ()) == "provisional"
    no_value = SessionRecord(n_calls=1, exploration=[entry(0, field=None, value_hash=None)])
    advance(no_value, env_at(1), 0.1, 0.6, (DetectorScore("novelty", None, 0.0, {}),), PROV)
    assert no_value.exploration[0].state == "provisional"


def test_ttl_counts_down_then_unresolved_once() -> None:
    rec = SessionRecord(n_calls=1, total_mass=0.9, exploration=[entry(0)])
    got = [advance(rec, env_at(i), 0.0, 0.6, (), PROV) for i in range(1, 6)]
    assert got == [set(), set(), {Flag.UNRESOLVED}, set(), set()]
    assert rec.unresolved_mass == pytest.approx(0.9)
    assert rec.exploration == []  # pruned on the call after it closed


def test_off_track_clears_below_budget_and_can_raise_again() -> None:
    cfg = PROV.replace(prov_min_calls=1)
    rec = SessionRecord(n_calls=4, total_mass=1.0, unresolved_mass=0.3)
    assert advance(rec, env_at(4), 0.0, 0.6, (), cfg) == {Flag.OFF_TRACK}
    assert advance(rec, env_at(5), 0.0, 0.6, (), cfg) == set()  # still over: raised once
    assert advance(rec, env_at(6), 0.5, 0.6, (), cfg) == set()  # 0.3 / 1.5 = 0.2: cleared
    assert not rec.off_track_raised
    rec.unresolved_mass = 0.9
    assert advance(rec, env_at(7), 0.0, 0.6, (), cfg) == {Flag.OFF_TRACK}


def test_off_track_waits_for_min_calls() -> None:
    rec = SessionRecord(n_calls=6, total_mass=1.0, unresolved_mass=1.0)
    assert advance(rec, env_at(6), 0.0, 0.6, (), PROV) == set()  # 7th call
    rec.n_calls = 7
    assert advance(rec, env_at(7), 0.0, 0.6, (), PROV) == {Flag.OFF_TRACK}  # 8th


def test_record_round_trips_with_exploration(tmp_path: Path) -> None:
    rec = SessionRecord(
        last_ts=5,
        n_calls=2,
        window=[(0, ["PROVISIONAL"])],
        exploration=[entry(0), entry(1, field=None, value_hash=None, state="absorbed")],
        total_mass=1.5,
        unresolved_mass=0.25,
        off_track_raised=True,
    )
    assert SessionRecord.from_dict(orjson.loads(orjson.dumps(rec.to_dict()))) == rec
    eng, tr, clock = warm_engine()
    run_session(eng, novel_then_benign(clock, 2), tr)
    a = tmp_path / "a.jsonl"
    eng.store.dump(a)
    store = DictStore()
    store.load(a)
    assert store.get(key_sess("t")) == sess(eng)
    b = tmp_path / "b.jsonl"
    store.dump(b)
    assert a.read_bytes() == b.read_bytes()


def test_old_session_records_load_with_empty_exploration() -> None:
    old = {"last_ts": 1, "n_calls": 1, "window": [], "verdict": "ok", "halted": False}
    rec = SessionRecord.from_dict(old)
    assert rec.exploration == [] and rec.total_mass == 0.0 and not rec.off_track_raised


# -- purity: score reports exactly what update stores -----------------------------------------


def test_score_flags_match_the_recorded_window() -> None:
    eng, tr, clock = warm_engine()
    for raw in novel_then_benign(clock, 11):
        e = normalize(raw, tr)
        before = snapshot(eng)
        r1, r2 = eng.score(e), eng.score(e)
        assert snapshot(eng) == before  # score never moves the record
        assert r1.trajectory == r2.trajectory
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
        assert r1.trajectory is not None
        assert sess(eng).window[-1] == (e.seq_no, sorted(f.value for f in r1.trajectory.flags))


def test_update_without_a_cached_score_matches() -> None:
    a, tr_a, clock_a = warm_engine()
    b, tr_b, clock_b = warm_engine()
    for ra, rb in zip(novel_then_benign(clock_a, 9), novel_then_benign(clock_b, 9), strict=True):
        ea, eb = normalize(ra, tr_a), normalize(rb, tr_b)
        a.score(ea)
        a.update(ea, Outcome.ALLOWED)
        b.update(eb, Outcome.ALLOWED)  # never scored: a fresh pure evaluation first
        tr_a.observe(ea)
        tr_b.observe(eb)
    assert sess(a) == sess(b)


_SETTINGS = settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])


@_SETTINGS
@given(envelope_streams(), st.data())
def test_score_is_pure_with_provisional(stream: list[ActionEnvelope], data: st.DataObject) -> None:
    eng = Engine(PROV.replace(cal_theta0=0.2), enabled=KNOWN_DETECTORS)  # provisional often
    for e in stream[:-1]:
        eng.score(e)
        eng.update(e, data.draw(outcomes))
    before, snap = eng.state_bytes(), snapshot(eng)
    a, b = eng.score(stream[-1]), eng.score(stream[-1])
    assert repr(dataclasses.replace(a, latency_us=0.0)) == repr(
        dataclasses.replace(b, latency_us=0.0)
    )
    assert eng.state_bytes() == before and snapshot(eng) == snap


class _WindowChecked(Engine):
    """After every update, the recorded window entry equals the flags ``score`` reported."""

    last: frozenset[Flag] = frozenset()
    n_prov = 0

    def score(self, e: ActionEnvelope) -> DetectorResult:
        r = super().score(e)
        assert r.trajectory is not None
        self.last = r.trajectory.flags
        self.n_prov += Flag.PROVISIONAL in self.last
        return r

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        super().update(e, outcome)
        rec = self.store.get(key_sess(e.session_id))
        assert isinstance(rec, SessionRecord)
        assert rec.window[-1] == (e.seq_no, sorted(f.value for f in self.last))


def test_strict_replay_of_ci_dataset_with_provisional() -> None:
    raws, labels = ci_dataset()
    eng = _WindowChecked(PROV, enabled=KNOWN_DETECTORS)
    rows = replay(raws, labels, eng, SplitSpec(), strict=True, outcomes="review")  # no LeakError
    assert len(rows) == len(raws)
    assert eng.n_prov > 0  # the lifecycle is exercised
    assert not any(r.decision == "block" and "PROVISIONAL" in r.flags.split(",") for r in rows)
