from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence
from pathlib import Path

import orjson
import pytest

from agentfw.bench.ci_regress import ci_dataset
from agentfw.bench.replay import SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import (
    PrereqRecord,
    SequenceDetector,
    TransRecord,
    ValidatedRecord,
)
from agentfw.core.engine import KNOWN_DETECTORS, Engine
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType, LedgerRef
from agentfw.core.normalize import SessionTracker
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_prereq, key_trans, key_validated
from tests.core.test_provisional import DELETE, DOCS, call, novel_then_benign, warm_engine
from tests.factories import make_env
from tests.factories import run_session as run_engine_session

AGENT = "a1"
_sid = itertools.count()


class StubTypes(TypeView):
    def __init__(self) -> None:
        super().__init__(TypeInference(DictStore(), EngineConfig()), AGENT, "t")

    def __call__(self, path: str) -> FieldType:
        return FieldType.UNKNOWN


TYPES = StubTypes()


def tid(name: str) -> str:
    """Short tool name -> a valid ``svc.name`` tool_id."""
    return "s." + name.lower()


@pytest.fixture
def seq() -> SequenceDetector:
    return SequenceDetector(DictStore(), EngineConfig())


def run_session(
    det: SequenceDetector, tools: Sequence[str], sid: str | None = None, learn: bool = True
) -> None:
    """Feed a session: each call is built with real prior_tools, optionally learned."""
    tr = SessionTracker()
    sid = sid or f"s{next(_sid)}"
    for i, t in enumerate(tools):
        e = make_env(tracker=tr, session_id=sid, agent_id=AGENT, tool_id=tid(t), seq_no=i, args={})
        if learn:
            det.update(e, TYPES)
        tr.observe(e)


def probe(det: SequenceDetector, prior: Sequence[str], tool: str) -> ActionEnvelope:
    tr = SessionTracker()
    sid = f"p{next(_sid)}"
    for i, t in enumerate(prior):
        tr.observe(
            make_env(tracker=tr, session_id=sid, agent_id=AGENT, tool_id=tid(t), seq_no=i, args={})
        )
    return make_env(
        tracker=tr, session_id=sid, agent_id=AGENT, tool_id=tid(tool), seq_no=len(prior), args={}
    )


def one(det: SequenceDetector, e: ActionEnvelope) -> DetectorScore:
    out = det.score(e, TYPES)
    assert len(out) == 1
    assert out[0].detector == "sequence" and out[0].field is None
    return out[0]


def obs_y_with_x(det: SequenceDetector, x: str = "X", y: str = "Y") -> None:
    run_session(det, [x, y])


def prereqs(det: SequenceDetector, tool: str) -> frozenset[str]:
    return det.prerequisites(AGENT, tid(tool))


def test_skip_in_deterministic_agent_scores_high(seq: SequenceDetector) -> None:
    for _ in range(100):
        run_session(seq, ["A", "B", "C"])
    e = probe(seq, ["A"], "C")
    s_trans, ev = seq._s_trans(e)  # the transition part alone (B is also a mined prereq of C)
    assert s_trans > 0.8
    assert ev["k"] == 2
    assert one(seq, e).score > 0.8


def test_expected_transition_scores_low(seq: SequenceDetector) -> None:
    for _ in range(100):
        run_session(seq, ["A", "B", "C"])
    assert one(seq, probe(seq, ["A"], "B")).score < 0.05


def test_surprise_discounted_in_high_entropy_agent(seq: SequenceDetector) -> None:
    tools = [f"T{i}" for i in range(8)]
    for a, b in itertools.product(tools, tools):
        for c in tools:
            for _ in range(3):
                run_session(seq, [a, b, c])
    s = one(seq, probe(seq, ["T0", "T1"], "T2"))
    assert s.score < 0.2
    assert s.evidence["k"] == 2
    assert float(str(s.evidence["entropy_ratio"])) > 0.9


def test_backoff_to_k1_when_context_sparse(seq: SequenceDetector) -> None:
    for i in range(25):
        run_session(seq, [f"X{i}", "A", "B"])
    s = one(seq, probe(seq, ["Xnew", "A"], "B"))
    assert s.evidence["k"] == 1


def test_backoff_to_unigram_when_cold(seq: SequenceDetector) -> None:
    for i in range(5):
        run_session(seq, [f"X{i}", "A", "B"])
    assert one(seq, probe(seq, ["Xnew", "A"], "B")).evidence["k"] == 0


def test_degenerate_vocabulary_scores_zero(seq: SequenceDetector) -> None:
    assert one(seq, probe(seq, [], "A")).score == 0.0
    for _ in range(30):
        run_session(seq, ["A"])
    s = one(seq, probe(seq, [], "A"))
    assert s.score == 0.0  # V = 2
    assert math.isfinite(s.score)


def test_prereq_appears_exactly_at_50th_obs(seq: SequenceDetector) -> None:
    for _ in range(49):
        obs_y_with_x(seq)
    assert prereqs(seq, "Y") == frozenset()
    obs_y_with_x(seq)
    assert prereqs(seq, "Y") == frozenset({tid("X")})


# Boundary of the default prereq_rate (0.90 since docs/decisions/003-prereq-threshold.md).
_RATE_PCT = round(100 * EngineConfig().prereq_rate)


def test_prereq_not_mined_just_below_prereq_rate(seq: SequenceDetector) -> None:
    for i in range(100):
        if i < _RATE_PCT - 1:
            obs_y_with_x(seq)
        else:
            run_session(seq, ["Z", "Y"])
    assert prereqs(seq, "Y") == frozenset()


def test_prereq_mined_exactly_at_prereq_rate(seq: SequenceDetector) -> None:
    assert _RATE_PCT == 90
    for i in range(100):
        obs_y_with_x(seq) if i < _RATE_PCT else run_session(seq, ["Z", "Y"])
    assert prereqs(seq, "Y") == frozenset({tid("X")})


def test_prereq_counts_any_window_position_once(seq: SequenceDetector) -> None:
    for _ in range(50):
        run_session(seq, ["X", "X", "M", "N", "Y"])  # window of 4: X, X, M, N -> X counted once
    rec = seq._store.get(key_prereq(AGENT, tid("Y")))
    assert isinstance(rec, PrereqRecord)
    assert rec.n_y == 50 and rec.cnt[tid("X")] == 50


def test_missing_prereq_scores_one_with_evidence() -> None:
    # Spec §6.6(b) binary s_prereq, pinned (the default is graded by user decision, decision 008).
    seq = SequenceDetector(DictStore(), EngineConfig().replace(seq_prereq_graded=False))
    for _ in range(60):
        obs_y_with_x(seq)
    s = one(seq, probe(seq, ["Q"], "Y"))
    assert s.score == 1.0
    assert s.evidence["fired"] == "prereq"
    assert s.evidence["prereq_missing"] == tid("X")
    ok = one(seq, probe(seq, ["X"], "Y"))
    assert ok.evidence["prereq_missing"] is None
    assert ok.score < 1.0


def test_graded_prereq_scores_the_mined_rate_of_the_missing_tool() -> None:
    """Task 1.15 switch, default by user decision (decision 008): s_prereq = max mined rate over
    the missing prerequisites; ``seq_prereq_graded=False`` is the spec's binary score."""
    graded = SequenceDetector(DictStore(), EngineConfig())
    binary = SequenceDetector(DictStore(), EngineConfig().replace(seq_prereq_graded=False))
    for det in (graded, binary):
        for i in range(100):  # X: 95 %, W: 92 % of Y's windows
            tools = ["X", "W", "Y"] if i < 92 else ["X", "Z", "Y"] if i < 95 else ["Z", "Y"]
            run_session(det, tools)
    assert prereqs(graded, "Y") == frozenset({tid("X"), tid("W")})
    s = one(graded, probe(graded, ["Z"], "Y"))  # both missing; Z -> Y is a known transition
    assert s.evidence["fired"] == "prereq" and s.evidence["prereq_missing"] == tid("W")
    assert s.score == pytest.approx(0.95)
    assert one(graded, probe(graded, ["X", "Z"], "Y")).score == pytest.approx(0.92)  # W only
    assert one(binary, probe(binary, ["Z"], "Y")).score == 1.0
    assert one(graded, probe(graded, ["X", "W"], "Y")).evidence["prereq_missing"] is None


def test_score_is_pure(seq: SequenceDetector) -> None:
    for _ in range(60):
        obs_y_with_x(seq)
    store = seq._store
    before = store.nbytes()
    dump = {k: orjson.dumps(r.to_dict()) for k, r in store.items()}
    e = probe(seq, ["Q"], "Y")
    a = one(seq, e)
    b = one(seq, e)
    assert a == b
    assert store.nbytes() == before
    assert {k: orjson.dumps(r.to_dict()) for k, r in store.items()} == dump


def test_score_uses_fresh_prereqs_when_cache_is_stale(seq: SequenceDetector) -> None:
    for _ in range(60):
        obs_y_with_x(seq)
    rec = seq._store.get(key_prereq(AGENT, tid("Y")))
    assert isinstance(rec, PrereqRecord)
    rec.cached = (0, frozenset())  # stale
    assert one(seq, probe(seq, ["Q"], "Y")).evidence["prereq_missing"] == tid("X")
    assert rec.cached == (0, frozenset())  # score did not write


def test_records_roundtrip_and_deterministic(seq: SequenceDetector, tmp_path: object) -> None:
    for _ in range(55):
        obs_y_with_x(seq)
    t = seq._store.get(key_trans(AGENT))
    p = seq._store.get(key_prereq(AGENT, tid("Y")))
    assert isinstance(t, TransRecord) and isinstance(p, PrereqRecord)
    assert TransRecord.from_dict(t.to_dict()).to_dict() == t.to_dict()
    assert PrereqRecord.from_dict(p.to_dict()).to_dict() == p.to_dict()
    assert orjson.dumps(p.to_dict()) == orjson.dumps(PrereqRecord.from_dict(p.to_dict()).to_dict())
    assert t.tools == {tid("X"), tid("Y")}


# -- age discount (F6, Task 2.3) ----------------------------------------------------------------

AGE = EngineConfig(seq_age_discount=True)


def pair() -> tuple[SequenceDetector, SequenceDetector]:
    """(plain, discounted) detectors, trained identically by ``feed``."""
    return SequenceDetector(DictStore(), EngineConfig()), SequenceDetector(DictStore(), AGE)


def feed(dets: tuple[SequenceDetector, ...], tools: Sequence[str]) -> None:
    for d in dets:
        run_session(d, tools)


def trained() -> tuple[SequenceDetector, SequenceDetector]:
    plain, disc = pair()
    for _ in range(100):
        feed((plain, disc), ["A", "B", "C"])
    return plain, disc


def test_first_use_of_a_new_tool_has_zero_transition_weight() -> None:
    plain, disc = trained()
    e = probe(disc, ["A"], "N")
    s_plain, _ = plain._s_trans(probe(plain, ["A"], "N"))
    s_disc, ev = disc._s_trans(e)
    assert s_plain > 0.5
    assert s_disc == 0.0 and ev["age_weight"] == 0.0


def test_ramp_over_three_calls() -> None:
    plain, disc = trained()
    for k, omega in enumerate([0.0, 1 / 3, 2 / 3, 1.0]):
        sp, _ = plain._s_trans(probe(plain, ["A"], "N"))
        sd, ev = disc._s_trans(probe(disc, ["A"], "N"))
        assert ev["age_weight"] == pytest.approx(omega), k
        assert sd == pytest.approx(omega * sp)
        feed((plain, disc), ["A", "N"])
    assert disc._s_trans(probe(disc, ["A"], "N"))[1]["age_weight"] == 1.0  # stays 1


def irreversible_consumer(det: SequenceDetector, producer: str | None, **kw: bool) -> None:
    e = probe(det, ["A"], "W")
    tool = None if producer is None else tid(producer)
    ref = LedgerRef("doc_xxxxxxxx", "doc_id", 0, tool, "args")
    det.update(dataclasses.replace(e, ledger_refs=(ref,)), TYPES, **kw)


def test_consumed_by_irreversible_lifts_discount_early() -> None:
    _, disc = trained()
    irreversible_consumer(disc, "N", irreversible=True)
    s, ev = disc._s_trans(probe(disc, ["A"], "N"))
    assert ev["age_weight"] == 1.0 and s > 0.5
    rec = disc._store.get(key_validated(AGENT))
    assert isinstance(rec, ValidatedRecord) and rec.tools == {tid("N")}


@pytest.mark.parametrize(
    ("producer", "irreversible"), [("N", False), (None, True)], ids=["reversible", "task-input"]
)
def test_validation_needs_an_irreversible_call_and_a_producer_tool(
    producer: str | None, irreversible: bool
) -> None:
    _, disc = trained()
    irreversible_consumer(disc, producer, irreversible=irreversible)
    assert disc._s_trans(probe(disc, ["A"], "N"))[1]["age_weight"] == 0.0
    assert disc._store.get(key_validated(AGENT)) is None


def test_validated_record_round_trips(tmp_path: Path) -> None:
    _, disc = trained()
    irreversible_consumer(disc, "N", irreversible=True)
    assert key_validated("a:b") == "s:a%3Ab:validated"
    f = tmp_path / "state.jsonl"
    disc._store.dump(f)
    store = DictStore()
    store.load(f)
    rec = store.get(key_validated(AGENT))
    assert isinstance(rec, ValidatedRecord) and rec.tools == {tid("N")}


def test_discount_never_touches_prereq_or_flow() -> None:
    # age_h above Y's 60 observations, so the age weight is < 1 and the test is not vacuous
    plain = SequenceDetector(DictStore(), EngineConfig())
    disc = SequenceDetector(DictStore(), dataclasses.replace(AGE, age_h=1000))
    for _ in range(60):
        feed((plain, disc), ["X", "Y"])
    _, ev = disc._s_trans(probe(disc, [], "Y"))
    assert float(str(ev["age_weight"])) < 1.0

    def parts(d: SequenceDetector) -> dict[str, tuple[float, dict[str, object]]]:
        return {n: (s, ev) for n, s, ev in d._parts(probe(d, [], "Y"), TYPES)}  # X is missing

    p_plain, p_disc = parts(plain), parts(disc)
    assert p_plain["prereq"][0] > 0.0
    for part in ("prereq", "flow"):
        assert p_disc[part] == p_plain[part]
    one_disc = one(disc, probe(disc, [], "Y"))
    assert one_disc.score > 0.0 and one_disc.evidence["fired"] == "prereq"
    assert one_disc.evidence["prereq_missing"] == tid("X")


def test_off_is_byte_identical_and_emits_nothing(tmp_path: Path) -> None:
    off = SequenceDetector(DictStore(), EngineConfig())
    for _ in range(3):
        run_session(off, ["A", "B"])
    ev = one(off, probe(off, ["A"], "N")).evidence
    assert "age_weight" not in ev
    irreversible_consumer(off, "N", irreversible=True)
    assert off._store.get(key_validated(AGENT)) is None


def test_magnitude_and_novelty_unaffected() -> None:
    def results(cfg: EngineConfig) -> list[list[tuple[str, str | None, float]]]:
        eng, tr, clock = warm_engine(cfg)
        raws = novel_then_benign(clock, 3)
        return [
            [(s.detector, s.field, s.score) for s in r.per_detector if s.detector != "sequence"]
            for r in run_engine_session(eng, raws, tr)
        ]

    on, off = results(EngineConfig(seq_age_discount=True)), results(EngineConfig())
    assert on == off
    assert max(s for d, _, s in on[0] if d == "novelty") == 1.0


def test_engine_passes_irreversibility_under_br_mode_off() -> None:
    eng, tr, clock = warm_engine(AGE, tool=DELETE, item_count=None)
    assert eng.cfg.br_mode == "off"
    produced = "doc_prod77bb"
    raws = [
        call(clock, "t", 0, DOCS[0], tool="kb.get_doc", ids=(produced,)),
        call(clock, "t", 1, produced, tool=DELETE, item_count=None),
    ]
    run_engine_session(eng, raws, tr)
    rec = eng.store.get(key_validated("a1"))
    # (the warm-up repeats of delete_doc also join)
    assert isinstance(rec, ValidatedRecord) and "kb.get_doc" in rec.tools


def test_strict_replay_of_ci_dataset_with_age_discount() -> None:
    raws, labels = ci_dataset()
    eng = Engine(AGE, enabled=KNOWN_DETECTORS)
    rows = replay(raws, labels, eng, SplitSpec(), strict=True, outcomes="review")  # no LeakError
    assert len(rows) == len(raws)
