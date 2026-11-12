"""Session verdicts on severe-flag density (Task 3.3, feature spec F7; T-15, T-34, T-43)."""

from __future__ import annotations

import dataclasses
import itertools
import math
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
    Outcome,
    RawCall,
    ResponseMeta,
)
from agentfw.core.fingerprint import resp_fingerprint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.state import DictStore, key_sess
from agentfw.core.trajectory.decide import session_overlay
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.session import SessionBook, SessionRecord, SessionVerdict, ladder
from tests.core.strategies import envelope_streams, outcomes
from tests.factories import make_env, make_raw, run_session

T0 = 1_700_000_000_000
READ = "kb.search"
FP = resp_fingerprint("ticket list alpha beta gamma delta", 3, False)
assert FP is not None
ON = EngineConfig(sessions=True)
ENFORCE = EngineConfig(session_enforce=True)

OK, WARN, HALT = SessionVerdict.OK, SessionVerdict.WARN, SessionVerdict.HALT


def engine(cfg: EngineConfig = ON, **kw: Any) -> Engine:
    return Engine(cfg, enabled=KNOWN_DETECTORS, **kw)


def read_call(
    seq: int, *, session: str = "s1", q: str = "open tickets", fresh: bool = False, ts: int = 0
) -> RawCall:
    """The same read with the same response every time (a ``LOOP`` from the third call on);
    ``fresh`` gives a query, response words and a new id of its own (no flag)."""
    fp = FP
    ids: tuple[str, ...] = ()
    if fresh:
        q = f"topic {seq}"
        fp = resp_fingerprint(" ".join(f"w{seq}x{j}" for j in range(40)), 3, False)
        ids = (f"kb_{seq:08d}",)
    return make_raw(
        session_id=session,
        seq_no=seq,
        ts_unix_ms=ts or T0 + seq * 1_000,
        tool_id=READ,
        args={"q": q, "limit": 10},
        response_meta=ResponseMeta(item_count=3, fp=fp, ids=ids),
    )


def traj_of(r: DetectorResult) -> Any:
    assert r.trajectory is not None
    return r.trajectory


def verdicts(res: list[DetectorResult]) -> list[SessionVerdict | None]:
    return [traj_of(r).session_verdict for r in res]


def stored(eng: Engine, session: str = "s1") -> SessionRecord:
    rec = eng.store.get(key_sess(session))
    assert isinstance(rec, SessionRecord)
    return rec


def snapshot(eng: Engine) -> list[tuple[str, bytes]]:
    opt = orjson.OPT_SORT_KEYS
    return sorted((k, orjson.dumps(r.to_dict(), option=opt)) for k, r in eng.store.items())


# -- the ladder -----------------------------------------------------------------------------------


def test_ladder_thresholds() -> None:
    cfg = EngineConfig()
    assert (cfg.theta_warn, cfg.theta_halt, cfg.session_enforce) == (0.33, 0.6, False)
    got = [ladder(d, False, cfg) for d in (0.0, 1 / 6, 0.3299, 0.33, 2 / 6, 0.5, 0.5999, 0.6, 1.0)]
    assert got == [OK, OK, OK, WARN, WARN, WARN, WARN, HALT, HALT]
    assert ladder(0.0, True, cfg) is HALT  # a halted session reports HALT on every later call
    tight = cfg.replace(theta_warn=0.5, theta_halt=0.5)
    assert [ladder(d, False, tight) for d in (0.49, 0.5)] == [OK, HALT]
    assert [v.value for v in SessionVerdict] == ["ok", "warn_session", "halt_session"]


def test_engine_ladder_on_the_density_including_this_call() -> None:
    res = run_session(engine(), [read_call(i) for i in range(6)])
    assert [traj_of(r).density for r in res] == pytest.approx([0, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6])
    assert verdicts(res) == [OK, OK, OK, WARN, WARN, HALT]


def test_scored_verdict_is_what_update_stores() -> None:
    eng = engine(ON.replace(theta_warn=0.15, theta_halt=0.5))
    tr = SessionTracker()
    raws = [read_call(i) for i in range(5)] + [read_call(i, fresh=True) for i in range(5, 12)]
    halted_seen = False
    for raw in raws:
        e = normalize(raw, tr)
        t = traj_of(eng.score(e))
        eng.update(e, Outcome.ALLOWED)
        tr.observe(e)
        rec = stored(eng)
        assert rec.verdict == t.session_verdict.value
        halted_seen |= t.session_verdict is HALT
        assert rec.halted is halted_seen
        assert rec.fused_mass == t.session_mass


@pytest.mark.parametrize("bad", [{"theta_warn": 0.0}, {"theta_warn": 0.7}, {"theta_halt": 1.5}])
def test_thresholds_validated_only_behind_the_gate(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="theta_warn"):
        SessionBook(DictStore(), EngineConfig().replace(**bad))
    with pytest.raises(ValueError, match="theta_warn"):
        Engine(ON.replace(**bad))
    Engine(EngineConfig().replace(**bad))  # gate shut: never read, never rejected


# -- halt, release, enforcement -------------------------------------------------------------------


def test_halt_is_sticky_until_release() -> None:
    eng = engine()
    tr = SessionTracker()
    res = run_session(eng, [read_call(i) for i in range(6)], tr)
    assert verdicts(res)[-1] is HALT and stored(eng).halted
    calm = run_session(eng, [read_call(i, fresh=True) for i in range(6, 16)], tr)
    assert [traj_of(r).density for r in calm][-1] == 0.0  # the window has emptied ...
    assert verdicts(calm) == [HALT] * 10  # ... and the session stays halted
    n_calls, mass = stored(eng).n_calls, stored(eng).fused_mass
    eng.release_session("s1")
    rec = stored(eng)
    assert (rec.halted, rec.verdict, rec.window) == (False, "ok", [])
    assert (rec.n_calls, rec.fused_mass) == (n_calls, mass)  # only the verdict state is cleared
    after = run_session(eng, [read_call(16, fresh=True)], tr)
    assert verdicts(after) == [OK] and traj_of(after[0]).density == 0.0
    eng.release_session("never-seen")  # unknown session: no-op, nothing written
    assert eng.store.get(key_sess("never-seen")) is None
    shut = Engine(enabled=KNOWN_DETECTORS)
    shut.release_session("s1")  # gate shut: no-op
    assert shut.state_bytes() == 0


def test_enforce_blocks_after_halt_only_when_switched_on() -> None:
    raws = [read_call(i) for i in range(6)] + [read_call(i, fresh=True) for i in range(6, 9)]
    plain = run_session(Engine(enabled=KNOWN_DETECTORS), raws)
    shadow = run_session(engine(), raws)
    enforced_eng = engine(ENFORCE)  # session_enforce alone opens the session gate (T-34)
    tr = SessionTracker()
    enforced = run_session(enforced_eng, raws, tr)
    assert [r.decision for r in shadow] == [r.decision for r in plain]
    assert all(r.decision is not Decision.BLOCK for r in plain)
    assert verdicts(enforced) == verdicts(shadow) == [OK, OK, OK, WARN, WARN] + [HALT] * 4
    # The halting call itself is not blocked by the overlay; every later call is.
    assert [r.decision for r in enforced[:6]] == [r.decision for r in plain[:6]]
    assert [r.decision for r in enforced[6:]] == [Decision.BLOCK] * 3
    for r in enforced[6:]:
        top = r.reasons[0]
        assert top.detector == "session" and top.observed == "halt_session"
        assert top.field is None and top.prereq_missing is None
        assert "s1" not in repr(top)  # no raw ids in reasons
        assert Flag.BLOCK not in traj_of(r).flags  # the window counts the call's own evidence
    assert not any(x.detector == "session" for r in shadow for x in r.reasons)
    # Flags, density and verdicts are those of the shadow run: enforcement changes decisions.
    for a, b in zip(enforced, shadow, strict=True):
        ta, tb = traj_of(a), traj_of(b)
        assert (ta.flags, ta.density, ta.session_verdict) == (
            tb.flags,
            tb.density,
            tb.session_verdict,
        )
    enforced_eng.release_session("s1")
    after = run_session(enforced_eng, [read_call(9, fresh=True)], tr)
    assert after[0].decision is not Decision.BLOCK and verdicts(after) == [OK]


def test_overlay_unit() -> None:
    e = make_env()
    eng = engine(ENFORCE)
    t = traj_of(eng.score(e))
    t = dataclasses.replace(t, flags=frozenset({Flag.PROVISIONAL, Flag.NOVEL_READ}))
    for d in Decision:
        assert session_overlay(d, t, False, ENFORCE) == (d, t)
        assert session_overlay(d, t, True, ON) == (d, t)  # enforcement off: shadow only
        blocked, bt = session_overlay(d, t, True, ENFORCE)
        assert blocked is Decision.BLOCK
        assert bt.flags == {Flag.NOVEL_READ}  # a blocked call is not provisional


def test_enforced_provisional_call_opens_no_entry() -> None:
    cfg = ENFORCE.replace(provisional=True)
    eng = engine(cfg)
    rec = SessionRecord(halted=True, verdict="halt_session")
    eng.store.put(key_sess("s1"), rec)
    e = make_env(session_id="s1", seq_no=0, ts_unix_ms=T0)
    res = eng.score(e)
    assert res.decision is Decision.BLOCK and Flag.PROVISIONAL not in traj_of(res).flags
    eng.update(e, Outcome.ALLOWED)
    assert stored(eng).exploration == []


# -- eviction (Review Focus 5, T-15) --------------------------------------------------------------


def _late(session: str, hours: int) -> RawCall:
    return read_call(0, session=session, fresh=True, ts=T0 + hours * 3_600_000)


def test_halted_session_never_evicted() -> None:
    eng = engine(ENFORCE.replace(sess_sweep_every=1, sess_idle_ms=60_000))
    tr = SessionTracker()
    run_session(eng, [read_call(i, session="h") for i in range(6)], tr)
    run_session(eng, [read_call(0, session="idle", fresh=True)], tr)
    assert stored(eng, "h").halted
    run_session(eng, [_late("other", 1)], tr)  # sweeps: both "h" and "idle" are idle
    assert eng.store.get(key_sess("idle")) is None
    assert stored(eng, "h").halted
    back = run_session(eng, [read_call(7, session="h", fresh=True, ts=T0 + 7_200_000)], tr)
    assert verdicts(back) == [HALT] and back[0].decision is Decision.BLOCK


def test_evicted_idle_session_restarts_ok() -> None:
    eng = engine(ON.replace(sess_sweep_every=1, sess_idle_ms=60_000))
    tr = SessionTracker()
    res = run_session(eng, [read_call(i, session="w") for i in range(5)], tr)
    assert verdicts(res)[-1] is WARN
    run_session(eng, [_late("other", 1)], tr)
    assert eng.store.get(key_sess("w")) is None
    back = run_session(eng, [read_call(9, session="w", fresh=True, ts=T0 + 7_200_000)], tr)
    assert verdicts(back) == [OK] and traj_of(back[0]).density == 0.0
    rec = stored(eng, "w")
    assert rec.n_calls == 1 and [s for s, _ in rec.window] == [9]


# -- the aggregate is reported, never thresholded --------------------------------------------------


def test_aggregate_never_thresholded() -> None:
    # A threshold no score reaches, so no call is blocked (and no BLOCK flag); magnitude alone
    # raises no flag, so the session's mass is all fused score and its window is empty.
    cfg = ON.replace(cal_theta0=2.0, cal_blend_n=10**9)
    eng = Engine(cfg, enabled=frozenset({"magnitude"}))
    tr = SessionTracker()
    step = 600_000
    warm = [
        make_raw(session_id=f"w{i}", ts_unix_ms=T0 + i * step, args={"amount": 37.5 + i % 7})
        for i in range(60)
    ]
    run_session(eng, warm, tr)
    spikes = [
        make_raw(
            session_id="spiky",
            seq_no=i,
            ts_unix_ms=T0 + (60 + i) * step,
            args={"amount": 9e6 * (i + 1) ** 3},
        )
        for i in range(10)
    ]
    res = run_session(eng, spikes, tr)
    masses = [traj_of(r).session_mass for r in res]
    assert masses == pytest.approx(list(itertools.accumulate(r.fused_score for r in res)))
    assert masses[-1] > cfg.theta_halt * cfg.session_window  # far above any density threshold
    assert all(not traj_of(r).flags and traj_of(r).density == 0.0 for r in res)
    assert verdicts(res) == [OK] * 10


def test_book_mass_never_moves_the_verdict() -> None:
    book = SessionBook(DictStore(), EngineConfig())
    for i in range(40):
        book.record(make_env(seq_no=i, ts_unix_ms=T0 + i), set(), fused=1.0)
    rec = book.peek("s1")
    assert rec is not None and rec.fused_mass == 40.0
    assert rec.verdict == "ok" and not rec.halted
    book.record(make_env(seq_no=40, ts_unix_ms=T0 + 40), set(), fused=math.nan)
    assert rec.fused_mass == 40.0  # a NaN score adds nothing


# -- defaults, records, purity --------------------------------------------------------------------


def test_switches_off_report_no_verdict() -> None:
    res = run_session(Engine(enabled=KNOWN_DETECTORS), [read_call(i) for i in range(6)])
    assert all(traj_of(r).session_verdict is None and traj_of(r).session_mass is None for r in res)


def test_record_round_trips_with_verdict_fields(tmp_path: Path) -> None:
    rec = SessionRecord(last_ts=5, n_calls=2, verdict="halt_session", halted=True, fused_mass=2.5)
    assert SessionRecord.from_dict(orjson.loads(orjson.dumps(rec.to_dict()))) == rec
    old = rec.to_dict()
    del old["fused_mass"]  # a record dumped before Task 3.3
    assert SessionRecord.from_dict(old).fused_mass == 0.0
    eng = engine(ENFORCE)
    run_session(eng, [read_call(i) for i in range(7)])
    a = tmp_path / "a.jsonl"
    eng.store.dump(a)
    store = DictStore()
    store.load(a)
    assert store.get(key_sess("s1")) == stored(eng)


_SETTINGS = settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])


@_SETTINGS
@given(envelope_streams(), st.data())
def test_score_is_pure_with_enforcement(stream: list[ActionEnvelope], data: st.DataObject) -> None:
    cfg = ENFORCE.replace(theta_warn=0.1, theta_halt=0.15, sess_sweep_every=7, sess_idle_ms=60_000)
    eng = engine(cfg)
    for e in stream[:-1]:
        eng.score(e)
        eng.update(e, data.draw(outcomes))
    before, snap = eng.state_bytes(), snapshot(eng)
    a, b = eng.score(stream[-1]), eng.score(stream[-1])
    assert repr(dataclasses.replace(a, latency_us=0.0)) == repr(
        dataclasses.replace(b, latency_us=0.0)
    )
    assert eng.state_bytes() == before and snapshot(eng) == snap


def test_strict_replay_with_sessions_and_enforcement() -> None:
    raws, labels = ci_dataset()
    cfg = EngineConfig(sessions=True, session_enforce=True)
    for outcomes_mode in ("allow", "review"):
        rows = replay(raws, labels, engine(cfg), SplitSpec(), strict=True, outcomes=outcomes_mode)
        assert all(r.session_verdict in {"ok", "warn_session", "halt_session"} for r in rows)
