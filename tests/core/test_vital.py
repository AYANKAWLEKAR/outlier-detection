"""Vital-signs detector (F2): loop, spin and stall over the per-session recent-call ring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentfw.bench.registry import make_detector
from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.vital import VitalDetector
from agentfw.core.engine import ALL_DETECTORS, KNOWN_DETECTORS, VITAL, Engine
from agentfw.core.envelope import DetectorResult, RawCall, RecentCall, ResponseMeta
from agentfw.core.fingerprint import arg_tokens, call_fingerprint, resp_fingerprint
from agentfw.core.hints import ToolAnnotations, ToolHint
from agentfw.core.normalize import SessionTracker, normalize
from tests.factories import make_raw, run_session

T0 = 1_700_000_000_000
READ = "kb.search"
WRITE = "crm.update_contact"
FP = resp_fingerprint("ticket list alpha beta gamma delta", 3, False)
assert FP is not None
POLLING = {READ: ToolHint(annotations=ToolAnnotations(read_only=True, idempotent=True))}


def _fp(i: int) -> tuple[int, ...]:
    fp = resp_fingerprint(" ".join(f"w{i}x{j}" for j in range(40)), 3, False)
    assert fp is not None
    return fp


def fresh_reads(n: int) -> list[RawCall]:
    """``n`` different reads, each with a response unlike any other (high information gain)."""
    return [
        call(i, {"q": f"query {i}"}, meta=ResponseMeta(item_count=3, fp=_fp(i))) for i in range(n)
    ]


def call(
    seq: int,
    args: dict[str, Any] | None = None,
    *,
    tool: str = READ,
    session: str = "s1",
    meta: ResponseMeta | None = None,
) -> RawCall:
    return make_raw(
        session_id=session,
        seq_no=seq,
        ts_unix_ms=T0 + seq * 1_000,
        tool_id=tool,
        args={"q": "open tickets", "limit": 10} if args is None else args,
        response_meta=ResponseMeta(item_count=3, fp=FP) if meta is None else meta,
    )


def read_call(seq: int, *, error: bool = False, session: str = "s1") -> RawCall:
    """The same read with the same response every time."""
    return call(seq, session=session, meta=ResponseMeta(item_count=3, error=error, fp=FP))


def vital(r: DetectorResult, field: str) -> float:
    (s,) = [s for s in r.per_detector if s.detector == VITAL and s.field == field]
    return s.score


def scores(results: list[DetectorResult], field: str) -> list[float]:
    return [vital(r, field) for r in results]


def first_index(xs: list[float], v: float) -> int:
    return xs.index(v)


def engine(**kw: Any) -> Engine:
    return Engine(enabled=KNOWN_DETECTORS, **kw)


# -- loop -----------------------------------------------------------------------------------


def test_loop_fires_at_second_duplicate() -> None:  # F2 acceptance, T-08
    raws = [read_call(seq) for seq in range(6)]  # identical read, identical response
    loop = scores(run_session(engine(), raws), "__loop__")
    assert loop[:3] == [0.0, 0.7, 1.0]
    assert loop[3:] == [1.0, 1.0, 1.0]


def test_retry_exemption_after_error() -> None:  # paper scenario F analogue
    raws = [read_call(0, error=True)] + [read_call(i, error=True) for i in range(1, 5)]
    loop = scores(run_session(engine(), raws), "__loop__")
    assert loop[1:4] == [0.0, 0.0, 0.0] and loop[4] == 1.0


def test_retry_exemption_ends_at_a_successful_duplicate() -> None:
    raws = [read_call(0, error=True), read_call(1), read_call(2)]
    loop = scores(run_session(engine(), raws), "__loop__")
    assert loop == [0.0, 0.0, 1.0]  # the retry is exempt; its success is the latest match


def test_near_duplicate_scores_0_7() -> None:
    base = {f"k{i}": f"v{i}" for i in range(11)}
    plus_one = base | {"k11": "v11"}  # 12 tokens, one of which the earlier call lacks: J = 11/12
    loop = scores(run_session(engine(), [call(0, base), call(1, plus_one)]), "__loop__")
    assert loop == [0.0, 0.7]


def test_one_substituted_token_of_12_is_below_the_jaccard_bar() -> None:
    a = {f"k{i}": f"v{i}" for i in range(12)}
    b = a | {"k11": "changed"}  # J = 11/13 = 0.846 < vital_jaccard
    loop = scores(run_session(engine(), [call(0, a), call(1, b)]), "__loop__")
    assert loop == [0.0, 0.0]


def test_near_duplicate_needs_the_same_tool() -> None:
    a = {f"k{i}": f"v{i}" for i in range(11)}
    raws = [call(0, a, tool="kb.other"), call(1, a | {"k11": "v11"})]
    assert scores(run_session(engine(), raws), "__loop__") == [0.0, 0.0]


def _gap_session() -> list[RawCall]:
    """The same read, then 8 different reads, then the first read again (9 calls back)."""
    return [call(0)] + [call(i, {"q": f"topic {i}"}) for i in range(1, 9)] + [call(9)]


def test_polling_tools_use_the_long_window() -> None:
    plain = scores(run_session(engine(), _gap_session()), "__loop__")
    polling = scores(run_session(engine(hints=POLLING), _gap_session()), "__loop__")
    assert plain[9] == 0.0  # outside vital_w_loop = 8
    assert polling[9] == 0.7  # inside vital_w_loop_polling = 32


def test_polling_needs_both_annotations() -> None:
    only_ro = {READ: ToolHint(annotations=ToolAnnotations(read_only=True))}
    assert scores(run_session(engine(hints=only_ro), _gap_session()), "__loop__")[9] == 0.0


# -- spin -----------------------------------------------------------------------------------


def test_spin_within_w_spin() -> None:  # varying args, identical fp
    raws = [call(i, {"q": f"query {i}"}) for i in range(8)]
    spin_scores = scores(run_session(engine(), raws), "__spin__")
    assert first_index(spin_scores, 1.0) <= 4
    assert spin_scores[:4] == [0.0, 0.0, 0.0, 0.0]


def test_spin_off_when_responses_change() -> None:
    raws = fresh_reads(8)
    assert set(scores(run_session(engine(), raws), "__spin__")) == {0.0}


def test_errored_or_fp_less_calls_never_count_as_spin() -> None:  # Review Focus 3
    no_meta = [make_raw(**_kw(j), response_meta=None) for j in range(6)]
    no_fp = [make_raw(**_kw(j), response_meta=ResponseMeta(item_count=3)) for j in range(6)]
    errored = [
        make_raw(**_kw(j), response_meta=ResponseMeta(item_count=3, error=True, fp=FP))
        for j in range(6)
    ]
    mixed = [
        make_raw(**_kw(j), response_meta=None if j % 2 else ResponseMeta(item_count=3, fp=FP))
        for j in range(6)
    ]
    for raws in (no_meta, no_fp, errored, mixed):
        assert set(scores(run_session(engine(), raws), "__spin__")) == {0.0}
    tr = SessionTracker()
    run_session(engine(), no_meta + no_fp, tracker=tr)
    e = normalize(make_raw(**_kw(12)), tr)
    assert all(rc.delta_i is None for rc in e.recent) and e.delta_i is None


def _kw(j: int) -> dict[str, Any]:
    return {"seq_no": j, "ts_unix_ms": T0 + j * 1_000, "tool_id": READ, "args": {"q": f"q {j}"}}


# -- stall ----------------------------------------------------------------------------------


def test_stall_six_reads_no_new_artifacts() -> None:
    raws = fresh_reads(8)
    stall_scores = scores(run_session(engine(), raws), "__stall__")
    assert stall_scores[6] == 1.0 and stall_scores[5] == 0.0


def test_stall_off_when_a_call_produces_an_artifact() -> None:
    raws = fresh_reads(8)
    raws[3] = call(3, {"q": "x"}, meta=ResponseMeta(item_count=3, fp=_fp(3), ids=("kb_00000001",)))
    stall_scores = scores(run_session(engine(), raws), "__stall__")
    assert stall_scores[6:] == [0.0, 0.0]  # high-gain reads, one new artifact in the window


def test_stall_off_when_the_window_holds_a_write() -> None:
    hints = {WRITE: ToolHint(kind="write")}
    raws = fresh_reads(8)
    raws[2] = call(2, {"name": "x"}, tool=WRITE, meta=ResponseMeta(fp=_fp(2)))
    stall_scores = scores(run_session(engine(hints=hints), raws), "__stall__")
    assert stall_scores[6:] == [0.0, 0.0]


def test_stall_half_when_reads_low_gain() -> None:
    # Busy re-reading the same page while one new id trickles in per call: no 1.0 (artifacts
    # appear), but every entry is a read with delta_i < 0.3.
    raws = [
        call(i, {"q": f"query {i}"}, meta=ResponseMeta(item_count=3, fp=FP, ids=(f"kb_{i:08d}",)))
        for i in range(9)
    ]
    stall_scores = scores(run_session(engine(), raws), "__stall__")
    assert stall_scores[6] == 0.0  # entry 0 had delta_i = 1.0 (nothing before it)
    assert stall_scores[7] == 0.5 and stall_scores[8] == 0.5


# -- ring plumbing --------------------------------------------------------------------------


def test_interleaved_sessions_keep_separate_rings() -> None:  # Review Focus 1
    raws: list[RawCall] = []
    for i in range(4):
        raws.append(read_call(2 * i, session="s1"))
        raws.append(call(2 * i + 1, {"q": f"other {i}"}, session="s2"))
    raws.append(read_call(9, session="s2"))  # s1's call, first time in s2
    tr = SessionTracker()
    res = run_session(engine(), raws, tracker=tr)
    loop = scores(res, "__loop__")
    assert loop[0::2][:4] == [0.0, 0.7, 1.0, 1.0]  # s1
    assert loop[1::2] == [0.0, 0.0, 0.0, 0.0]  # s2
    assert loop[8] == 0.0
    e1 = normalize(read_call(20, session="s1"), tr)
    e2 = normalize(read_call(21, session="s2"), tr)
    assert [rc.seq_no for rc in e1.recent] == [0, 2, 4, 6]
    assert [rc.seq_no for rc in e2.recent] == [1, 3, 5, 7, 9]
    assert normalize(read_call(0, session="new"), tr).recent == ()


def test_envelope_ring_fields() -> None:
    tr = SessionTracker()
    raws = [
        call(i, {"q": f"query {i}"}, meta=ResponseMeta(item_count=3, fp=FP, ids=(f"kb_{i:08d}",)))
        for i in range(40)
    ]
    run_session(engine(), raws, tracker=tr)
    probe = call(40, {"q": "probe"})
    e = normalize(probe, tr)
    assert e.call_fp == call_fingerprint(READ, e.flat_args)
    assert len(e.recent) == 32 and [rc.seq_no for rc in e.recent] == list(range(8, 40))
    last = e.recent[-1]
    assert last == RecentCall(
        seq_no=39,
        tool_id=READ,
        call_fp=call_fingerprint(READ, {"q": ("query 39",)}),
        arg_tokens=arg_tokens({"q": ("query 39",)}),
        error=False,
        delta_i=0.0,
        new_artifacts=1,
    )
    assert e.delta_i == 0.0  # the probe's own response matches the ring


def test_new_artifacts_counts_ledger_entries_created() -> None:
    tr = SessionTracker()
    a, b = "ch_aaaaaaaa11", "ch_bbbbbbbb22"
    raws = [
        call(0, {"id": a}, meta=ResponseMeta(item_count=1, ids=(b, "re_cccccccc33"))),
        call(1, {"id": a}, meta=ResponseMeta(item_count=1, ids=(b,))),  # nothing new
        make_raw(seq_no=2, tool_id=READ, args={"x": "y"}, task_ids=("tk_dddddddd44",)),
    ]
    run_session(engine(), raws, tracker=tr)
    e = normalize(call(3), tr)
    assert [rc.new_artifacts for rc in e.recent] == [3, 0, 0]  # task seeds are not produced


def test_normalize_reads_the_ring_without_mutating_it() -> None:
    tr = SessionTracker()
    run_session(engine(), [read_call(0), read_call(1)], tracker=tr)
    a = normalize(read_call(2), tr)
    b = normalize(read_call(2), tr)
    assert a.recent == b.recent and len(a.recent) == 2


def test_tracker_dump_load_round_trips_the_ring(tmp_path: Path) -> None:
    tr = SessionTracker()
    raws = [
        call(i, {"q": f"q {i % 3}"}, session=f"s{i % 2}", meta=m)
        for i, m in enumerate(
            [
                ResponseMeta(item_count=3, fp=FP),
                ResponseMeta(item_count=1, error=True, fp=_fp(1)),
                ResponseMeta(item_count=2),
                ResponseMeta(item_count=3, fp=FP, ids=("kb_00000009",)),
            ]
            * 12
        )
    ]
    run_session(engine(), raws, tracker=tr)
    p = tmp_path / "tr.jsonl"
    tr.dump(p)
    tr2 = SessionTracker.load(p)
    for sid in ("s0", "s1", "s9"):
        probe = call(99, {"q": "q 1"}, session=sid, meta=ResponseMeta(item_count=3, fp=FP))
        a, b = normalize(probe, tr), normalize(probe, tr2)
        assert a.recent == b.recent and a.delta_i == b.delta_i and a.call_fp == b.call_fp
    tr2.dump(tmp_path / "again.jsonl")
    assert (tmp_path / "again.jsonl").read_bytes() == p.read_bytes()


# -- engine wiring --------------------------------------------------------------------------


def _mixed_session() -> list[RawCall]:
    raws = [read_call(i) for i in range(5)]
    raws += [call(5 + i, {"q": f"query {i}"}) for i in range(6)]
    raws += [read_call(11 + i, error=True) for i in range(5)]
    raws += [call(16 + i, {"amount": 10 + i}, tool=WRITE) for i in range(4)]
    return raws


def test_vital_never_moves_the_fused_score() -> None:  # T-07
    raws = _mixed_session()
    plain, withv = run_session(Engine(), raws), run_session(Engine(enabled=KNOWN_DETECTORS), raws)
    assert any(vital(r, "__loop__") == 1.0 for r in withv)
    for v1, vt in zip(plain, withv, strict=True):
        assert v1.fused_score == vt.fused_score and v1.decision == vt.decision
        assert v1.threshold == vt.threshold
        n = len(v1.per_detector)
        assert vt.per_detector[:n] == v1.per_detector
        assert {s.detector for s in vt.per_detector[n:]} == {VITAL}
        assert vt.reasons[: len(v1.reasons)] == v1.reasons


def test_vital_reasons_follow_v1_reasons() -> None:
    res = run_session(engine(), [read_call(i) for i in range(6)])
    r = res[5]
    vital_reasons = [x for x in r.reasons if x.detector == VITAL]
    assert {x.field for x in vital_reasons} == {"__loop__", "__spin__"}
    tail = r.reasons[-len(vital_reasons) :]
    assert list(tail) == vital_reasons


def test_engine_enabled_validation() -> None:
    assert VITAL == "vital" and frozenset({"vital"}) == KNOWN_DETECTORS - ALL_DETECTORS
    assert Engine().enabled == ALL_DETECTORS
    Engine(enabled=frozenset({VITAL}))
    with pytest.raises(ValueError, match="unknown detectors"):
        Engine(enabled=frozenset({"vitals"}))


def test_vital_only_engine_fuses_to_zero() -> None:
    res = run_session(Engine(enabled=frozenset({VITAL})), [read_call(i) for i in range(4)])
    assert [r.fused_score for r in res] == [0.0] * 4
    assert vital(res[3], "__loop__") == 1.0


def test_registry_engine_vital() -> None:
    det = make_detector("engine-vital")
    assert isinstance(det, Engine) and det.enabled == KNOWN_DETECTORS
    with pytest.raises(KeyError):
        make_detector("engine-no-vital")


def test_score_is_pure_with_vital_in_strict_replay() -> None:
    rows = replay(_mixed_session(), {}, make_detector("engine-vital"), SplitSpec(), strict=True)
    assert len(rows) == 20


def test_detector_scores_have_no_raw_ids_and_update_is_noop() -> None:
    det = VitalDetector(EngineConfig(), {}, lambda agent, tool: False)
    tr = SessionTracker()
    run_session(engine(), [call(0, {"id": "ch_aaaaaaaa11"})], tracker=tr)
    e = normalize(call(1, {"id": "ch_aaaaaaaa11"}), tr)
    out = det.score(e, None)  # type: ignore[arg-type]
    assert [s.field for s in out] == ["__loop__", "__spin__", "__stall__"]
    assert all(s.detector == "vital" for s in out)
    assert "ch_aaaaaaaa11" not in repr(out)
    det.update(e, None)  # type: ignore[arg-type]


# -- hints ----------------------------------------------------------------------------------


def test_tool_annotations_default_and_existing_constructors() -> None:
    assert ToolHint().annotations is None
    assert ToolHint("write", {"amount": "money"}).annotations is None
    ann = ToolAnnotations(read_only=True, idempotent=True)
    assert (ann.destructive, ann.open_world) == (None, None)
    assert not hasattr(ann, "__dict__")
    with pytest.raises(AttributeError):
        ann.read_only = False  # type: ignore[misc]
    assert ToolHint(kind="read", annotations=ann).annotations == ann
