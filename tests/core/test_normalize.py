from __future__ import annotations

from agentfw.core.envelope import PrevResponse, ResponseMeta
from agentfw.core.normalize import SessionTracker, flatten, normalize
from tests.factories import make_env, make_raw


def test_flatten_nested_lists_and_lengths() -> None:
    flat, d = flatten({"a": {"b": 1}, "to": ["X@Acme.example", "y@acme.example"], "empty": []})
    assert flat["a.b"] == (1,)
    assert flat["to[]"] == ("X@Acme.example", "y@acme.example")
    assert d["to.__len__"] == (2,) and d["empty.__len__"] == (0,)
    assert d["to[].__domain__"] == ("acme.example", "acme.example")


def test_long_strings_become_length_only() -> None:
    flat, d = flatten({"body": "x" * 120})
    assert "body" not in flat and d["body.__strlen__"] == (120,)


def test_id_prefix_and_url_host() -> None:
    _, d = flatten(
        {
            "c": "ch_3Nabc12345XYZ",
            "sf": "003Dn00000ABCDEFGH",
            "u": "https://Files.Example.com/x?y=1",
        }
    )
    assert d["c.__prefix__"] == ("ch_",) and d["sf.__prefix__"] == ("003",)
    assert d["u.__host__"] == ("files.example.com",)


def test_domain_case_and_idn_normalized() -> None:
    _, d = flatten({"to": "Bob@ACME.example", "cc": "a@BÜCHER.example"})
    assert d["to.__domain__"] == ("acme.example",)
    assert d["cc.__domain__"] == ("xn--bcher-kva.example",)


def test_messy_args_do_not_crash() -> None:
    deep: dict[str, object] = {}
    cur = deep
    for _ in range(50):
        nxt: dict[str, object] = {}
        cur["n"] = nxt
        cur = nxt
    flat, d = flatten(
        {
            "none": None,
            "flag": True,
            "items": [{"id": "ch_AAAAAAAAAA"}, {"id": None}],
            "deep": deep,
            5: "intkey",  # type: ignore[dict-item]
        }
    )
    assert flat["items[].id"] == ("ch_AAAAAAAAAA", None)
    assert any(k.endswith("__truncated__") for k in d)
    assert flat["5"] == ("intkey",)
    assert flat["none"] == (None,) and flat["flag"] == (True,)


def test_tracker_is_causal_and_bounded() -> None:
    tr = SessionTracker(k=4, max_ids=2000)
    e0 = normalize(
        make_raw(
            tool_id="zendesk.get_ticket",
            args={"ticket_id": "T-00000001"},
            response_meta=ResponseMeta(item_count=1, ids=("ch_AAAAAAAAAA",)),
        ),
        tr,
    )
    assert e0.prior_tools == () and e0.prev_response is None
    tr.observe(e0)
    e1 = normalize(
        make_raw(seq_no=1, tool_id="stripe.retrieve_charge", args={"charge_id": "ch_AAAAAAAAAA"}),
        tr,
    )
    assert e1.prior_tools == ("zendesk.get_ticket",)
    assert "ch_AAAAAAAAAA" in e1.session_ids_seen
    assert "T-00000001" in e1.session_ids_seen
    assert e1.prev_response == PrevResponse("zendesk.get_ticket", 1)
    for i in range(2, 8):
        tr.observe(normalize(make_raw(seq_no=i, tool_id=f"x.t{i}"), tr))
    assert normalize(make_raw(seq_no=8), tr).prior_tools == ("x.t4", "x.t5", "x.t6", "x.t7")


def test_ids_overflow_to_bloom_and_sessions_isolated() -> None:
    tr = SessionTracker(max_ids=10)
    for i in range(50):
        tr.observe(make_env(make_raw(seq_no=i, args={"id": f"ch_{i:010d}"}), tr))
    snap = normalize(make_raw(seq_no=50), tr).session_ids_seen
    assert all(f"ch_{i:010d}" in snap for i in range(50)) and len(snap.exact) == 10
    assert sum(f"zz_{i:010d}" in snap for i in range(10_000)) < 100
    assert "ch_0000000001" not in normalize(make_raw(session_id="s2"), tr).session_ids_seen


def test_normalize_does_not_observe_and_snapshot_is_immutable() -> None:
    tr = SessionTracker()
    before = normalize(make_raw(args={"id": "ch_AAAAAAAAAA"}), tr)
    assert normalize(make_raw(seq_no=1), tr).prior_tools == ()
    tr.observe(before)
    after = normalize(make_raw(seq_no=2), tr)
    assert "ch_AAAAAAAAAA" in after.session_ids_seen
    assert "ch_AAAAAAAAAA" not in before.session_ids_seen
