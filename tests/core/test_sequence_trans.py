from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import orjson
import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import PrereqRecord, SequenceDetector, TransRecord
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType
from agentfw.core.normalize import SessionTracker
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_prereq, key_trans
from tests.factories import make_env

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


def test_missing_prereq_scores_one_with_evidence(seq: SequenceDetector) -> None:
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
    """Task 1.15 opt-in (decision 008): s_prereq = max mined rate over the missing prerequisites."""
    graded = SequenceDetector(DictStore(), EngineConfig().replace(seq_prereq_graded=True))
    binary = SequenceDetector(DictStore(), EngineConfig())
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
