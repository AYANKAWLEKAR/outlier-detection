from __future__ import annotations

import logging
from pathlib import Path

import orjson
import pytest

from agentfw.core.envelope import RawCall, ResponseMeta
from agentfw.core.ledger import ArtifactLedger, LedgerRef
from agentfw.core.normalize import SessionTracker
from tests.factories import make_env, make_raw

CH = "ch_aaaaaaaa11"
RE = "re_bbbbbbbb22"


def test_response_id_becomes_producer_and_later_arg_is_a_ref() -> None:
    t = SessionTracker()
    read = make_env(
        make_raw(
            seq_no=0,
            tool_id="stripe.retrieve_charge",
            args={"charge_id": CH},
            response_meta=ResponseMeta(ids=(RE,)),
        ),
        t,
    )
    t.observe(read)
    w = make_env(make_raw(seq_no=1, tool_id="stripe.create_refund", args={"refund_id": RE}), t)
    assert w.ledger_refs == (LedgerRef(RE, "refund_id", 0, "stripe.retrieve_charge", "response"),)


def test_arg_id_first_seen_in_earlier_call_is_an_args_producer() -> None:
    t = SessionTracker()
    t.observe(make_env(make_raw(seq_no=0, tool_id="a.read", args={"charge_id": CH}), t))
    e = make_env(make_raw(seq_no=1, args={"c": CH}), t)
    assert e.ledger_refs == (LedgerRef(CH, "c", 0, "a.read", "args"),)
    t.observe(e)
    led = t.ledger("s1")
    assert led is not None
    entry = led.lookup(CH)
    assert entry is not None and entry.consumers == [1]


def test_args_first_entry_gets_a_producer_when_a_response_returns_it() -> None:  # T-59
    t = SessionTracker()
    t.observe(make_env(make_raw(seq_no=0, tool_id="a.read", args={"charge_id": CH}), t))
    t.observe(make_env(make_raw(seq_no=1, args={"c": CH}), t))  # consumer of the args entry
    led = t.ledger("s1")
    assert led is not None
    back = make_raw(seq_no=2, tool_id="a.lookup", response_meta=ResponseMeta(ids=(CH,)))
    assert led.record(make_env(back, t)) == 0  # not a new artifact
    entry = led.lookup(CH)
    assert entry is not None
    assert (entry.producer_seq, entry.producer_tool, entry.source) == (2, "a.lookup", "response")
    assert entry.consumers == [1]
    later = make_raw(seq_no=3, tool_id="a.other", response_meta=ResponseMeta(ids=(CH,)))
    led.record(make_env(later, t))  # a response producer is never replaced
    assert (entry.producer_seq, entry.source) == (2, "response")


def test_echoing_lookup_is_the_producer_and_an_errored_one_is_not() -> None:  # T-59
    t = SessionTracker()
    echo = make_raw(
        seq_no=0, tool_id="a.lookup", args={"id": CH}, response_meta=ResponseMeta(ids=(CH,))
    )
    fail = make_raw(
        seq_no=1, tool_id="a.lookup", args={"id": RE}, response_meta=ResponseMeta(error=True)
    )
    t.observe(make_env(echo, t))
    t.observe(make_env(fail, t))
    e = make_env(make_raw(seq_no=2, args={"x": CH, "y": RE}), t)
    assert [(r.artifact_id, r.source, r.producer_seq) for r in e.ledger_refs] == [
        (CH, "response", 0),
        (RE, "args", 1),
    ]


def test_task_ids_seed_before_the_first_call() -> None:
    t = SessionTracker()
    e = make_env(make_raw(seq_no=0, args={"charge_id": CH}, task_ids=(CH,)), t)
    assert e.ledger_refs[0].source == "task_input" and e.ledger_refs[0].producer_seq == 0
    assert e.ledger_refs[0].producer_tool is None and e.task_ids == (CH,)


def test_task_input_source_survives_later_response() -> None:
    t = SessionTracker()
    t.observe(make_env(make_raw(seq_no=0, task_ids=(CH,)), t))
    t.observe(make_env(make_raw(seq_no=1, response_meta=ResponseMeta(ids=(CH,))), t))
    led = t.ledger("s1")
    assert led is not None
    entry = led.lookup(CH)
    assert entry is not None and entry.source == "task_input" and entry.producer_tool is None
    assert make_env(make_raw(seq_no=2, args={"x": CH}), t).ledger_refs[0].source == "task_input"


def test_own_args_are_not_their_own_producer() -> None:
    e = make_env(make_raw(seq_no=0, args={"a": CH, "b": CH}), SessionTracker())
    assert e.ledger_refs == ()


def test_refs_in_sorted_path_order_one_per_path() -> None:
    t = SessionTracker()
    t.observe(make_env(make_raw(seq_no=0, response_meta=ResponseMeta(ids=(CH, RE))), t))
    e = make_env(make_raw(seq_no=1, args={"z": CH, "a": [RE, RE], "m": "plain"}), t)
    assert [(r.path, r.artifact_id) for r in e.ledger_refs] == [("a[]", RE), ("z", CH)]


def test_interleaved_sessions_do_not_share_ledgers() -> None:
    t = SessionTracker()
    t.observe(
        make_env(make_raw(session_id="s1", seq_no=0, response_meta=ResponseMeta(ids=(RE,))), t)
    )
    t.observe(make_env(make_raw(session_id="s2", seq_no=0, args={"q": "x"}), t))
    assert make_env(make_raw(session_id="s2", seq_no=1, args={"r": RE}), t).ledger_refs == ()
    assert len(make_env(make_raw(session_id="s1", seq_no=1, args={"r": RE}), t).ledger_refs) == 1
    assert t.ledger("nope") is None


def test_overflow_degrades_to_bloom_membership(caplog: pytest.LogCaptureFixture) -> None:
    led = ArtifactLedger(max_entries=4)
    ids = [f"ch_{i:010d}" for i in range(6)]
    with caplog.at_level(logging.WARNING, logger="agentfw.core.ledger"):
        for i, v in enumerate(ids):
            led.record(make_env(make_raw(seq_no=i, args={"id": v})))
    assert led.degraded is True
    assert all(v in led for v in ids)
    assert (
        led.lookup(ids[0]) is not None and led.lookup(ids[4]) is None and led.lookup(ids[5]) is None
    )
    assert len(led) == 4
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


def test_tracker_dump_load_round_trip(tmp_path: Path) -> None:
    t = SessionTracker(max_ids=12)
    calls = []
    for i in range(30):
        sid = "s1" if i % 3 else "s2"
        raw = make_raw(
            session_id=sid,
            seq_no=i,
            tool_id=f"x.t{i % 4}",
            args={"id": f"ch_{i % 17:010d}", "other": f"ch_{(i * 7) % 29:010d}"},
            response_meta=ResponseMeta(item_count=i, ids=(f"re_{i:010d}",)),
            task_ids=(CH,) if i < 3 else (),
        )
        calls.append(raw)
        t.observe(make_env(raw, t))
    p = tmp_path / "tracker.jsonl"
    t.dump(p)
    lines = p.read_bytes().splitlines()
    assert [orjson.loads(x)["session_id"] for x in lines] == ["s1", "s2"]
    t2 = SessionTracker.load(p)
    probe = [
        make_raw(session_id=s, seq_no=99, args={"a": f"ch_{i:010d}", "b": f"re_{i:010d}", "c": CH})
        for s in ("s1", "s2")
        for i in range(30)
    ]
    for raw in probe:
        a, b = make_env(raw, t), make_env(raw, t2)
        assert a.ledger_refs == b.ledger_refs
        assert a.prior_tools == b.prior_tools and a.prev_response == b.prev_response
        assert a.session_ids_seen.exact == b.session_ids_seen.exact
    t2.dump(tmp_path / "again.jsonl")
    assert (tmp_path / "again.jsonl").read_bytes() == p.read_bytes()
    empty = tmp_path / "empty.jsonl"
    SessionTracker().dump(empty)
    assert SessionTracker.load(empty).ledger("s1") is None


def test_rawcall_task_ids_default_is_omitted_and_round_trips() -> None:
    raw = make_raw()
    assert "task_ids" not in raw.model_dump(mode="json")
    assert "task_ids" not in orjson.loads(raw.model_dump_json())
    seeded = make_raw(task_ids=(CH,))
    assert RawCall.model_validate_json(seeded.model_dump_json()) == seeded
    assert RawCall.model_validate(seeded.model_dump(mode="json")).task_ids == (CH,)
    assert RawCall.model_validate_json(raw.model_dump_json()).task_ids == ()
    with pytest.raises(ValueError):
        make_raw(task_ids=tuple(f"ch_{i:010d}" for i in range(65)))


def test_record_returns_the_number_of_new_artifacts() -> None:
    led = ArtifactLedger(max_entries=2)
    led.seed_task(["tk_aaaaaaaa00"])  # seeds are not any call's artifacts
    first = make_env(
        make_raw(
            seq_no=0, args={"a": CH, "t": "tk_aaaaaaaa00"}, response_meta=ResponseMeta(ids=(RE,))
        )
    )
    assert led.record(first) == 2  # CH (args) fills the exact table; RE overflows to the Bloom
    assert led.record(make_env(make_raw(seq_no=1, args={"a": CH, "b": RE}))) == 0
    third = make_env(make_raw(seq_no=2, response_meta=ResponseMeta(ids=("re_cccccccc33", RE))))
    assert led.record(third) == 1
