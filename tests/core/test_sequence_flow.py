from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import orjson
import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import FlowRecord, SequenceDetector, ToolKindRecord
from agentfw.core.envelope import (
    ActionEnvelope,
    DetectorScore,
    FieldType,
    PrevResponse,
    ResponseMeta,
)
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_flow, key_kind
from tests.factories import make_env

AGENT = "a1"
READ = "s.retrieve_charge"
WRITE = "s.create_refund"
_sid = itertools.count()


class IdTypes(TypeView):
    """Types every listed path as ID, everything else UNKNOWN."""

    def __init__(self, *paths: str) -> None:
        super().__init__(TypeInference(DictStore(), EngineConfig()), AGENT, "t")
        self._paths = set(paths or ("charge_id",))

    def __call__(self, path: str) -> FieldType:
        return FieldType.ID if path in self._paths else FieldType.UNKNOWN


TYPES = IdTypes()


@pytest.fixture
def seq() -> SequenceDetector:
    return SequenceDetector(DictStore(), EngineConfig())


def call(
    tr: SessionTracker,
    sid: str,
    i: int,
    tool: str,
    args: Mapping[str, Any],
    item_count: int | None = None,
) -> ActionEnvelope:
    rm = None if item_count is None else ResponseMeta(item_count=item_count)
    return make_env(
        tracker=tr,
        session_id=sid,
        agent_id=AGENT,
        tool_id=tool,
        seq_no=i,
        args=dict(args),
        response_meta=rm,
    )


def run(
    det: SequenceDetector,
    steps: Sequence[tuple[str, Mapping[str, Any], int | None]],
    types: TypeView = TYPES,
) -> None:
    """Feed one session of (tool, args, item_count); every call is learned (all allowed)."""
    tr = SessionTracker()
    sid = f"s{next(_sid)}"
    for i, (tool, args, n) in enumerate(steps):
        e = call(tr, sid, i, tool, args, n)
        det.update(e, types)
        tr.observe(e)


def train(det: SequenceDetector, flowing: int, external: int = 0) -> None:
    """`flowing` sessions look the id up first; `external` sessions refund an unseen id."""
    for i in range(flowing):
        cid = f"ch_3N{i:06d}"
        run(det, [(READ, {"charge_id": cid}, 1), (WRITE, {"charge_id": cid}, None)])
    for i in range(external):
        run(det, [(WRITE, {"charge_id": f"ch_9X{i:06d}"}, None)])


def probe(steps: Sequence[tuple[str, Mapping[str, Any], int | None]]) -> ActionEnvelope:
    """Build the last step as a real envelope after observing the earlier ones."""
    tr = SessionTracker()
    sid = f"p{next(_sid)}"
    for i, (tool, args, n) in enumerate(steps[:-1]):
        tr.observe(call(tr, sid, i, tool, args, n))
    tool, args, n = steps[-1]
    return call(tr, sid, len(steps) - 1, tool, args, n)


def one(det: SequenceDetector, e: ActionEnvelope, types: TypeView = TYPES) -> DetectorScore:
    out = det.score(e, types)
    assert len(out) == 1
    return out[0]


def test_flow_score_equals_historical_rate(seq: SequenceDetector) -> None:
    train(seq, flowing=19, external=1)  # 19/20 of refund ids were already in the session
    e = probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, None)])
    s, ev = seq._s_flow(e, TYPES)
    assert s == pytest.approx(0.95)
    assert ev["missing_id_path"] == "charge_id"
    assert ev["r_flow"] == pytest.approx(0.95)
    assert one(seq, e).score >= 0.95
    assert "ch_3FAKE0001" not in orjson.dumps(one(seq, e).evidence).decode()


def test_external_id_writer_scores_zero(seq: SequenceDetector) -> None:
    train(seq, flowing=0, external=25)
    e = probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, None)])
    s, ev = seq._s_flow(e, TYPES)
    assert s == 0.0
    assert ev["r_flow"] == 0.0


def test_present_id_scores_zero(seq: SequenceDetector) -> None:
    train(seq, flowing=25)
    e = probe(
        [(READ, {"charge_id": "ch_3SEEN0003"}, 1), (WRITE, {"charge_id": "ch_3SEEN0003"}, None)]
    )
    s, ev = seq._s_flow(e, TYPES)
    assert s == 0.0
    assert ev["missing_id_path"] is None


def test_cold_flow_below_min_obs_scores_zero(seq: SequenceDetector) -> None:
    train(seq, flowing=19)  # total 19 < flow_min_obs
    s, _ = seq._s_flow(probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, None)]), TYPES)
    assert s == 0.0
    train(seq, flowing=1)  # total 20
    s, _ = seq._s_flow(probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, None)]), TYPES)
    assert s == 1.0


def test_reads_never_flow_scored(seq: SequenceDetector) -> None:
    for i in range(30):
        run(
            seq,
            [
                (READ, {"charge_id": f"ch_3M{i:06d}"}, 1),
                (WRITE, {"charge_id": f"ch_3M{i:06d}"}, None),
            ],
        )
    assert not seq.is_write(AGENT, READ)  # has item_count history
    s, _ = seq._s_flow(probe([(READ, {"charge_id": "ch_3FAKE0001"}, None)]), TYPES)
    assert s == 0.0


def test_write_kind_inference(seq: SequenceDetector) -> None:
    assert not seq.is_write(AGENT, WRITE)  # unseen
    run(seq, [(WRITE, {"charge_id": "ch_3ONEONE01"}, None)])
    assert seq.is_write(AGENT, WRITE)
    run(seq, [(WRITE, {"charge_id": "ch_3TWOTWO02"}, 3)])  # later shows an item_count -> read
    assert not seq.is_write(AGENT, WRITE)
    run(seq, [("s.noargs", {}, None)])
    assert not seq.is_write(AGENT, "s.noargs")  # no id arg -> not a write


def test_kind_is_causal_own_item_count_not_used_in_score(seq: SequenceDetector) -> None:
    train(seq, flowing=25)
    # the probe itself carries item_count, but score() only sees past updates
    e = probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, 5)])
    s, _ = seq._s_flow(e, TYPES)
    assert s == 1.0


def test_hint_overrides_inference() -> None:
    hints = {WRITE: ToolHint(kind="read"), READ: ToolHint(kind="write")}
    det = SequenceDetector(DictStore(), EngineConfig(), hints)
    for i in range(25):
        run(
            det,
            [
                (READ, {"charge_id": f"ch_3M{i:06d}"}, 1),
                (WRITE, {"charge_id": f"ch_3M{i:06d}"}, None),
            ],
        )
    assert not det.is_write(AGENT, WRITE)
    assert det.is_write(AGENT, READ)


def test_dummy_lookup_of_different_id_still_fires(seq: SequenceDetector) -> None:
    train(seq, flowing=30)
    e = probe(
        [
            (READ, {"charge_id": "ch_3OTHER0002"}, 1),  # decoy lookup of a different id
            (WRITE, {"charge_id": "ch_3FAKE0001"}, None),
        ]
    )
    assert "ch_3OTHER0002" in e.session_ids_seen and "ch_3FAKE0001" not in e.session_ids_seen
    s = one(seq, e)
    assert s.score >= 0.9
    assert s.evidence["fired"] == "flow"
    assert s.evidence["missing_id_path"] == "charge_id"


def test_multivalued_ids_take_max_over_missing(seq: SequenceDetector) -> None:
    lt = IdTypes("charge_id[]")
    for i in range(19):  # history on the list path: 19/20 present
        a = f"ch_3L{i:06d}"
        run(seq, [(READ, {"charge_id": a}, 1), (WRITE, {"charge_id": [a]}, None)], lt)
    run(seq, [(WRITE, {"charge_id": ["ch_9LEXTERN1"]}, None)], lt)
    e = probe(
        [
            (READ, {"charge_id": "ch_3AAAAAAA"}, 1),
            (WRITE, {"charge_id": ["ch_3AAAAAAA", "ch_3BBBBBBB"]}, None),
        ]
    )
    s, _ = seq._s_flow(e, lt)
    assert s == pytest.approx(0.95)  # ch_b is missing even though ch_a is present


def test_only_id_typed_paths_are_considered(seq: SequenceDetector) -> None:
    train(seq, flowing=25)
    e = probe([(WRITE, {"note_id": "ch_3FAKE0001", "charge_id": "ch_3XXXXXXX"}, None)])
    other = IdTypes("note_id")  # charge_id UNKNOWN here
    s, ev = seq._s_flow(e, other)
    assert s == 0.0  # no flow history for note_id yet
    assert ev["missing_id_path"] == "note_id"


def test_fired_is_none_when_all_parts_zero(seq: SequenceDetector) -> None:
    out = one(seq, probe([(WRITE, {"charge_id": "ch_3ONEONE01"}, None)]))
    assert out.score == 0.0
    assert out.evidence["fired"] is None


def test_flow_part_is_registered_in_parts(seq: SequenceDetector) -> None:
    train(seq, flowing=30)
    e = probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, None)])
    parts = seq._parts(e, TYPES)
    assert [p[0] for p in parts] == ["trans", "prereq", "flow"]
    assert one(seq, e).score == max(p[1] for p in parts)


def test_score_is_pure(seq: SequenceDetector) -> None:
    train(seq, flowing=25)
    store = seq._store
    dump = {k: orjson.dumps(r.to_dict()) for k, r in store.items()}
    n = store.nbytes()
    e = probe([(WRITE, {"charge_id": "ch_3FAKE0001"}, None)])
    assert one(seq, e) == one(seq, e)
    assert store.nbytes() == n
    assert {k: orjson.dumps(r.to_dict()) for k, r in store.items()} == dump


def test_update_counts_present_and_total(seq: SequenceDetector) -> None:
    train(seq, flowing=3, external=2)
    rec = seq._store.get(key_flow(AGENT, WRITE, "charge_id"))
    assert isinstance(rec, FlowRecord)
    assert (rec.present, rec.total) == (3, 5)
    kind = seq._store.get(key_kind(AGENT, WRITE))
    assert isinstance(kind, ToolKindRecord)
    assert kind.has_id_arg and not kind.has_item_count


def test_records_roundtrip_and_json_finite(seq: SequenceDetector) -> None:
    train(seq, flowing=5, external=1)
    for key in (key_flow(AGENT, WRITE, "charge_id"), key_kind(AGENT, WRITE)):
        rec = seq._store.get(key)
        assert rec is not None
        d = rec.to_dict()
        assert type(rec).from_dict(d).to_dict() == d
        assert orjson.dumps(d, option=orjson.OPT_SORT_KEYS) == orjson.dumps(
            type(rec).from_dict(d).to_dict(), option=orjson.OPT_SORT_KEYS
        )
    assert math.isfinite(one(seq, probe([(WRITE, {"charge_id": "ch_3ZZZZZZZ"}, None)])).score)


def test_item_count_via_prev_response_marks_read(seq: SequenceDetector) -> None:
    """Pre-response envelopes (no own response_meta): the count arrives on the next call."""
    first = call(SessionTracker(), "sq2", 0, READ, {"charge_id": "ch_3SQTWO001"})
    assert first.response_meta is None
    seq.update(first, TYPES)
    assert seq.is_write(AGENT, READ)  # nothing known yet
    nxt = call(SessionTracker(), "sq2", 1, WRITE, {"charge_id": "ch_3SQTWO001"})
    nxt = dataclasses.replace(nxt, prev_response=PrevResponse(READ, 3))
    seq.update(nxt, TYPES)
    assert not seq.is_write(AGENT, READ)
    assert seq.is_write(AGENT, WRITE)
    s, _ = seq._s_flow(probe([(READ, {"charge_id": "ch_3FAKE0001"}, None)]), TYPES)
    assert s == 0.0
