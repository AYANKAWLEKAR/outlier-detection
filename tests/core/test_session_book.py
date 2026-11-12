"""Session records and the per-session flag window (F7 substrate, Task 2.1)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import orjson
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentfw.bench.ci_regress import ci_dataset
from agentfw.bench.replay import ScoreRow, SplitSpec, read_score_table, replay, write_score_table
from agentfw.core.config import EngineConfig
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import ActionEnvelope, DetectorResult, Outcome, RawCall, ResponseMeta
from agentfw.core.fingerprint import resp_fingerprint
from agentfw.core.hints import ToolAnnotations, ToolHint
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.state import DictStore, key_sess
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.session import SessionBook, SessionRecord
from tests.core.strategies import envelope_streams, outcomes
from tests.factories import make_env, make_raw, run_session

T0 = 1_700_000_000_000
READ = "kb.search"
FP = resp_fingerprint("ticket list alpha beta gamma delta", 3, False)
assert FP is not None
POLLING = {READ: ToolHint(annotations=ToolAnnotations(read_only=True, idempotent=True))}
ON = EngineConfig(sessions=True)


def engine(cfg: EngineConfig = ON, **kw: Any) -> Engine:
    return Engine(cfg, enabled=KNOWN_DETECTORS, **kw)


def read_call(
    seq: int, *, session: str = "s1", q: str = "open tickets", fresh: bool = False
) -> RawCall:
    """The same read with the same response every time (a loop once repeated); ``fresh`` gives
    the response words of its own (high information gain)."""
    fp = FP
    if fresh:
        fp = resp_fingerprint(" ".join(f"w{seq}x{j}" for j in range(40)), 3, False)
    return make_raw(
        session_id=session,
        seq_no=seq,
        ts_unix_ms=T0 + seq * 1_000,
        tool_id=READ,
        args={"q": q, "limit": 10},
        response_meta=ResponseMeta(item_count=3, fp=fp),
    )


def env(seq: int, session: str = "s1", ts: int | None = None) -> ActionEnvelope:
    return make_env(
        session_id=session, seq_no=seq, ts_unix_ms=T0 + seq * 1_000 if ts is None else ts
    )


def flags_of(r: DetectorResult) -> frozenset[Flag]:
    assert r.trajectory is not None
    return r.trajectory.flags


def density_of(r: DetectorResult) -> float | None:
    assert r.trajectory is not None
    return r.trajectory.density


def snapshot(eng: Engine) -> list[tuple[str, bytes]]:
    opt = orjson.OPT_SORT_KEYS
    return sorted((k, orjson.dumps(r.to_dict(), option=opt)) for k, r in eng.store.items())


# -- SessionBook ------------------------------------------------------------------------------


def test_key_sess_escapes() -> None:
    assert key_sess("s1") == "sess:s1"
    assert key_sess("a:b%c") == "sess:a%3Ab%25c"


def test_density_counts_highest_weight_per_call() -> None:  # T-09
    book = SessionBook(DictStore(), EngineConfig())
    assert book.density_with(None, set()) == 0.0
    assert book.density_with(None, {Flag.LOOP, Flag.BLOCK}) == pytest.approx(1 / 6)
    assert book.density_with(None, {Flag.LOOP, Flag.NOVEL_READ}) == pytest.approx(1 / 6)
    assert book.density_with(None, {Flag.NOVEL_READ, Flag.COLD_START}) == 0.0
    assert book.density_with(None, {Flag.UNRESOLVED}) == pytest.approx(0.125 / 6)
    book.record(env(0), {Flag.LOOP, Flag.SPIN, Flag.STALL})
    book.record(env(1), set())
    book.record(env(2), {Flag.UNRESOLVED, Flag.PROVISIONAL})
    rec = book.peek("s1")
    assert rec is not None
    assert book.density_with(rec, {Flag.BLOCK}) == pytest.approx((1 + 0 + 0.125 + 1) / 6)


def test_window_is_six_calls() -> None:
    book = SessionBook(DictStore(), EngineConfig())
    for i in range(10):
        book.record(env(i), {Flag.LOOP} if i % 2 else set())
    rec = book.peek("s1")
    assert rec is not None
    assert [s for s, _ in rec.window] == [4, 5, 6, 7, 8, 9]
    assert rec.window[1] == (5, ["LOOP"]) and rec.window[0] == (4, [])
    assert rec.n_calls == 10 and rec.last_ts == T0 + 9_000
    # The density including this call drops the oldest entry: 5, 7, 9 + this one.
    assert book.density_with(rec, {Flag.BLOCK}) == pytest.approx(4 / 6)
    assert book.density_with(rec, set()) == pytest.approx(3 / 6)


def test_window_names_are_sorted() -> None:
    book = SessionBook(DictStore(), EngineConfig())
    book.record(env(0), {Flag.STALL, Flag.BLOCK, Flag.LOOP})
    rec = book.peek("s1")
    assert rec is not None and rec.window == [(0, ["BLOCK", "LOOP", "STALL"])]


def test_session_window_is_configurable() -> None:
    book = SessionBook(DictStore(), EngineConfig(session_window=3))
    for i in range(5):
        book.record(env(i), {Flag.SPIN})
    rec = book.peek("s1")
    assert rec is not None and [s for s, _ in rec.window] == [2, 3, 4]
    assert book.density_with(rec, set()) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="session_window"):
        SessionBook(DictStore(), EngineConfig(session_window=0))


@pytest.mark.parametrize("bad", [{"session_window": 0}, {"sess_sweep_every": 0}])
def test_session_settings_validated_only_behind_the_gate(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=next(iter(bad))):
        SessionBook(DictStore(), EngineConfig().replace(**bad))
    with pytest.raises(ValueError, match=next(iter(bad))):
        Engine(ON.replace(**bad))
    Engine(EngineConfig().replace(**bad))  # gate shut: never read, never rejected


def test_peek_is_pure_and_unknown_session_is_none() -> None:
    store = DictStore()
    book = SessionBook(store, EngineConfig())
    assert book.peek("nope") is None
    assert len(store) == 0


def test_record_round_trips_dump_load(tmp_path: Path) -> None:
    rec = SessionRecord(last_ts=5, n_calls=2, window=[(0, []), (1, ["BLOCK", "LOOP"])])
    assert SessionRecord.from_dict(rec.to_dict()) == rec
    halted = dataclasses.replace(rec, verdict="halt_session", halted=True)
    assert SessionRecord.from_dict(orjson.loads(orjson.dumps(halted.to_dict()))) == halted
    eng = engine()
    tr = SessionTracker()
    run_session(eng, [read_call(i) for i in range(4)], tr)
    run_session(eng, [read_call(i, session="s:2") for i in range(3)], tr)
    a = tmp_path / "a.jsonl"
    eng.store.dump(a)
    store = DictStore()
    store.load(a)
    got = store.get(key_sess("s:2"))
    assert isinstance(got, SessionRecord) and got == eng.store.get(key_sess("s:2"))
    b = tmp_path / "b.jsonl"
    store.dump(b)
    assert a.read_bytes() == b.read_bytes()


def test_maybe_evict_sweeps_every_n_records_and_keeps_halted() -> None:
    store = DictStore()
    book = SessionBook(store, EngineConfig(sess_sweep_every=3, sess_idle_ms=1_000))
    book.record(env(0, "old", ts=T0), set())
    assert book.maybe_evict(T0 + 10_000) == 0  # 1 record since the last sweep
    book.record(env(0, "kept", ts=T0), set())
    halted = book.peek("kept")
    assert halted is not None
    halted.halted = True  # T-15: never idle-evicted
    assert book.maybe_evict(T0 + 10_000) == 0  # 2 records
    book.record(env(0, "fresh", ts=T0 + 9_500), set())
    assert book.maybe_evict(T0 + 10_000) == 1  # 3rd record: sweep
    assert book.peek("old") is None
    assert book.peek("kept") is not None and book.peek("fresh") is not None
    # the counter restarted: nothing is swept until 3 more records
    book.record(env(1, "fresh", ts=T0 + 9_600), set())
    assert book.maybe_evict(T0 + 99_000) == 0


def test_idle_is_strictly_greater_than_sess_idle_ms() -> None:
    book = SessionBook(DictStore(), EngineConfig(sess_sweep_every=1, sess_idle_ms=1_000))
    book.record(env(0, "a", ts=T0), set())
    assert book.maybe_evict(T0 + 1_000) == 0
    book.record(env(0, "b", ts=T0 + 1_001), set())
    assert book.maybe_evict(T0 + 1_001) == 1 and book.peek("a") is None


# -- engine wiring ----------------------------------------------------------------------------


def test_sessions_off_by_default_keeps_state_and_still_reports_flags() -> None:
    eng = Engine(enabled=KNOWN_DETECTORS)
    assert eng.cfg.sessions is False
    res = run_session(eng, [read_call(i) for i in range(4)])
    assert not any(k.startswith("sess:") for k, _ in eng.store.items())
    assert all(density_of(r) is None for r in res)
    assert Flag.LOOP in flags_of(res[2])


def test_sessions_never_move_scores_decisions_or_other_state() -> None:
    raws = [read_call(i) for i in range(8)] + [read_call(i, session="s2") for i in range(5)]
    off, on = Engine(enabled=KNOWN_DETECTORS), engine()
    r_off, r_on = run_session(off, raws), run_session(on, raws)

    def strip(r: DetectorResult) -> DetectorResult:
        assert r.trajectory is not None
        traj = dataclasses.replace(r.trajectory, density=None)
        return dataclasses.replace(r, latency_us=0.0, trajectory=traj)

    assert [repr(strip(r)) for r in r_off] == [repr(strip(r)) for r in r_on]
    assert snapshot(off) == [(k, v) for k, v in snapshot(on) if not k.startswith("sess:")]


def test_recording_and_density_follow_the_one_session_gate() -> None:  # T-34, ruling R15
    raws = [read_call(i) for i in range(4)]
    forced_on = Engine(enabled=KNOWN_DETECTORS)  # sessions off ...
    forced_on._book = SessionBook(forced_on.store, forced_on.cfg)  # ... but the gate is open
    res = run_session(forced_on, raws)
    assert [density_of(r) for r in res] == pytest.approx([0, 0, 1 / 6, 2 / 6])
    rec = forced_on.store.get(key_sess("s1"))
    assert isinstance(rec, SessionRecord) and rec.n_calls == 4
    forced_off = engine()  # sessions on ...
    forced_off._book = None  # ... but the gate is shut
    res = run_session(forced_off, raws)
    assert all(density_of(r) is None for r in res)
    assert not any(k.startswith("sess:") for k, _ in forced_off.store.items())


def test_engine_density_over_a_loop() -> None:
    res = run_session(engine(), [read_call(i) for i in range(6)])
    # loop 1.0 from call 2 on (T-08), spin from call 4: one weight per call, not two
    assert [Flag.LOOP in flags_of(r) for r in res] == [False, False, True, True, True, True]
    assert Flag.SPIN in flags_of(res[4])
    assert [density_of(r) for r in res] == pytest.approx([0, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6])


def test_polling_tool_loop_is_not_severe() -> None:  # F2 risk mitigation
    res = run_session(engine(hints=POLLING), [read_call(i) for i in range(6)])
    loop = [s.score for r in res for s in r.per_detector if s.field == "__loop__"]
    assert loop[2:] == [1.0, 1.0, 1.0, 1.0]  # the vital score still reports the repeat
    assert not any(Flag.LOOP in flags_of(r) for r in res)
    plain = run_session(engine(), [read_call(i) for i in range(6)])
    assert all(Flag.LOOP in flags_of(r) for r in plain[2:])
    assert all(
        (density_of(p) or 0.0) <= (density_of(q) or 0.0) for p, q in zip(res, plain, strict=True)
    )


def test_blocked_calls_enter_the_window() -> None:
    cfg = ON.replace(vel_count_attempts=False)  # spec: a refused call moves nothing else
    for outcome in (Outcome.BLOCKED, Outcome.DENIED, Outcome.APPROVED, Outcome.ALLOWED):
        eng = engine(cfg)
        run_session(eng, [read_call(i) for i in range(3)], outcome=outcome)
        rec = eng.store.get(key_sess("s1"))
        assert isinstance(rec, SessionRecord), outcome
        assert [s for s, _ in rec.window] == [0, 1, 2] and rec.n_calls == 3
    blocked = engine(cfg)
    run_session(blocked, [read_call(i) for i in range(3)], outcome=Outcome.BLOCKED)
    assert [k for k, _ in blocked.store.items()] == [key_sess("s1")]  # no baseline moved


def test_update_without_a_cached_score_records_fresh_flags() -> None:
    a, b = engine(), engine()
    tr_a, tr_b = SessionTracker(), SessionTracker()
    for i in range(5):
        ea, eb = normalize(read_call(i), tr_a), normalize(read_call(i), tr_b)
        a.score(ea)
        a.update(ea, Outcome.ALLOWED)
        b.update(eb, Outcome.ALLOWED)  # never scored: flags computed fresh, before updates
        tr_a.observe(ea)
        tr_b.observe(eb)
    assert snapshot(a) == snapshot(b)


def test_interleaved_sessions_separate_windows() -> None:  # Review Focus 1
    raws: list[RawCall] = []
    for i in range(6):
        raws.append(read_call(i, session="loopy"))
        raws.append(read_call(i, session="calm", q=f"topic {i}", fresh=True))
    eng = engine()
    res = run_session(eng, raws)
    loopy, calm = eng.store.get(key_sess("loopy")), eng.store.get(key_sess("calm"))
    assert isinstance(loopy, SessionRecord) and isinstance(calm, SessionRecord)
    assert loopy.n_calls == calm.n_calls == 6
    assert [s for s, _ in calm.window] == list(range(6))
    assert all(names == [] for _, names in calm.window)
    assert sum("LOOP" in names for _, names in loopy.window) == 4
    assert all(density_of(r) == 0.0 for r in res[1::2])
    assert density_of(res[-2]) == pytest.approx(4 / 6)


def test_engine_evicts_idle_sessions() -> None:
    cfg = ON.replace(sess_sweep_every=2, sess_idle_ms=60_000)
    eng = engine(cfg)
    tr = SessionTracker()
    run_session(eng, [read_call(0, session="early")], tr)
    late = make_raw(session_id="late", seq_no=0, ts_unix_ms=T0 + 3_600_000, tool_id=READ)
    run_session(eng, [late], tr)
    assert eng.store.get(key_sess("early")) is None
    assert eng.store.get(key_sess("late")) is not None


def test_reset_clears_sessions_and_sweep_counter() -> None:
    cfg = ON.replace(sess_sweep_every=2, sess_idle_ms=0)
    eng = engine(cfg)
    run_session(eng, [read_call(0, session="a")])
    eng.reset()
    assert eng.state_bytes() == 0
    tr = SessionTracker()
    run_session(eng, [read_call(0, session="b")], tr)  # 1st record after reset: no sweep
    assert eng.store.get(key_sess("b")) is not None
    run_session(eng, [read_call(5, session="c")], tr)  # 2nd: sweep, "b" is idle
    assert eng.store.get(key_sess("b")) is None and eng.store.get(key_sess("c")) is not None


_SETTINGS = settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])


@_SETTINGS
@given(envelope_streams(), st.data())
def test_score_is_pure_with_sessions(stream: list[ActionEnvelope], data: st.DataObject) -> None:
    eng = engine(ON.replace(sess_sweep_every=7, sess_idle_ms=60_000))
    for e in stream[:-1]:
        eng.score(e)
        eng.update(e, data.draw(outcomes))
    before, snap = eng.state_bytes(), snapshot(eng)
    a, b = eng.score(stream[-1]), eng.score(stream[-1])
    assert repr(dataclasses.replace(a, latency_us=0.0)) == repr(
        dataclasses.replace(b, latency_us=0.0)
    )
    assert eng.state_bytes() == before and snapshot(eng) == snap
    density = density_of(a)
    assert density is not None and 0.0 <= density <= 1.0


# -- replay -----------------------------------------------------------------------------------


def test_strict_replay_of_ci_dataset_with_sessions_and_vital() -> None:
    raws, labels = ci_dataset()
    rows = replay(raws, labels, engine(), SplitSpec(), strict=True, outcomes="review")
    assert len(rows) == len(raws)
    assert all(r.density is not None and 0.0 <= r.density <= 1.0 for r in rows)
    flagged = [r for r in rows if r.flags]
    assert flagged  # flags are exercised
    for r in flagged:
        names = r.flags.split(",")
        assert names == sorted(names) and all(n in Flag.__members__ for n in names)


def test_replay_fills_flags_and_density_and_round_trips(tmp_path: Path) -> None:
    raws = [read_call(i) for i in range(4)]
    rows = replay(raws, {}, engine(), SplitSpec())
    assert [r.flags for r in rows] == ["", "", "LOOP", "LOOP"]
    assert [r.density for r in rows] == pytest.approx([0, 0, 1 / 6, 2 / 6])
    off = replay(raws, {}, Engine(enabled=KNOWN_DETECTORS), SplitSpec())
    assert [r.flags for r in off] == ["", "", "LOOP", "LOOP"]
    assert all(r.density is None for r in off)
    p = tmp_path / "scores.parquet"
    write_score_table(rows, p)
    assert read_score_table(p) == rows
    write_score_table(off, p)
    assert read_score_table(p) == off


def test_score_row_defaults() -> None:
    names = [f.name for f in dataclasses.fields(ScoreRow)]
    assert names[-3:] == ["flags", "density", "session_label"]
