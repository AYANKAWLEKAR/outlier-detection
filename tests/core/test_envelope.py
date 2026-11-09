import dataclasses

import pytest
from pydantic import ValidationError

from agentfw.core.envelope import (
    ActionEnvelope,
    DetectorResult,
    IdSnapshot,
    Label,
    RawCall,
    ResponseMeta,
)
from tests.factories import make_raw


def test_rawcall_json_roundtrip() -> None:
    raw = make_raw(
        args={"amount": 37.5, "to": ["x@acme.example"]},
        response_meta=ResponseMeta(item_count=3, ids=("ch_ABCDEFGH12",)),
    )
    assert RawCall.model_validate_json(raw.model_dump_json()) == raw
    assert raw.schema_version == 1


def test_rawcall_rejects_unnamespaced_tool() -> None:
    with pytest.raises(ValidationError):
        make_raw(tool_id="create_refund")


def test_rawcall_rejects_negative_seq() -> None:
    with pytest.raises(ValidationError):
        make_raw(seq_no=-1)


def test_make_raw_ids_are_unique_valid_ulids() -> None:
    a, b = make_raw(), make_raw()
    assert len(a.envelope_id) == 26 and a.envelope_id != b.envelope_id
    assert make_raw(envelope_id="custom").envelope_id == "custom"


def test_response_meta_ids_bounded() -> None:
    with pytest.raises(ValidationError):
        ResponseMeta(ids=tuple(f"ch_{i:010d}" for i in range(257)))
    assert len(ResponseMeta(ids=tuple(f"ch_{i:010d}" for i in range(256))).ids) == 256


def test_label_integrity_rules() -> None:
    Label(envelope_id="e", label="benign", control_type="benign_unusual")
    with pytest.raises(ValidationError):
        Label(envelope_id="e", label="attack")  # no family / attack_id
    with pytest.raises(ValidationError):
        Label(
            envelope_id="e",
            label="attack",
            attack_family="prereq_skip",
            attack_id="A1",
            control_type="benign_unusual",
        )
    ok = Label(envelope_id="e", label="attack", attack_family="prereq_skip", attack_id="A1")
    assert ok.injection_level is None and ok.strength is None and ok.step is None


def test_id_snapshot_contains() -> None:
    snap = IdSnapshot(exact=frozenset({"ch_1"}), bloom=None)
    assert "ch_1" in snap and "ch_2" not in snap


def test_envelope_is_frozen() -> None:
    raw = make_raw()
    env = ActionEnvelope(
        **raw.model_dump(),
        flat_args={"amount": (37.5,)},
        derived={},
        prior_tools=(),
        session_ids_seen=IdSnapshot(frozenset(), None),
        prev_response=None,
    )
    assert env.blast_radius is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.seq_no = 5  # type: ignore[misc]


def test_detector_result_defaults_not_shared() -> None:
    a = DetectorResult(fused_score=0.1, per_detector=(), reasons=(), latency_us=1.0)
    b = DetectorResult(fused_score=0.2, per_detector=(), reasons=(), latency_us=1.0)
    assert a.field_types == {} and a.field_types is not b.field_types
    assert a.threshold is None and a.decision is None
