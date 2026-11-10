"""JSONL envelope sink: three append-only files, sorted keys, loadable by the bench readers."""

from __future__ import annotations

from pathlib import Path

import orjson

from agentfw.bench.dataset import load_labels, load_traces
from agentfw.core.envelope import (
    BaselineSummary,
    Decision,
    DetectorResult,
    DetectorScore,
    FieldType,
    Label,
    RawCall,
    Reason,
    ResponseMeta,
)
from agentfw.sandbox.proxy_py import EnvelopeSink, JsonlSink


def _raw(i: int) -> RawCall:
    return RawCall(
        envelope_id=f"01HZZZZZZZZZZZZZZZZZZZZZ{i:02d}",
        session_id="sess_1",
        agent_id="agent_1",
        principal_id="user_1",
        seq_no=i,
        ts_unix_ms=1_700_000_000_000 + i,
        tool_id="stripe.create_refund",
        args={"zeta": 1, "alpha": {"b": 2, "a": [1.5, None]}},
        response_meta=ResponseMeta(item_count=1, bytes=10, ids=("ch_0000000001",)),
    )


def test_jsonl_sink_appends_sorted_and_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out"
    sink: EnvelopeSink = JsonlSink(out)
    sink.write_call(_raw(0))
    sink.write_call(_raw(1))
    lab = Label(
        envelope_id=_raw(1).envelope_id,
        label="attack",
        attack_family="magnitude_spike",
        injection_level="action",
        attack_id="atk_1",
    )
    sink.write_labels([lab])
    sink.write_labels([])

    lines = (out / "calls.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    assert lines[0] == orjson.dumps(orjson.loads(lines[0]), option=orjson.OPT_SORT_KEYS)
    assert load_traces(out / "calls.jsonl") == [_raw(0), _raw(1)]
    assert load_labels(out / "labels.jsonl") == {lab.envelope_id: lab}

    # a second sink on the same directory appends instead of truncating
    JsonlSink(out).write_call(_raw(2))
    assert [r.seq_no for r in load_traces(out / "calls.jsonl")] == [0, 1, 2]


def test_jsonl_sink_verdict_serializes_detector_result(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path)
    res = DetectorResult(
        fused_score=0.75,
        per_detector=(DetectorScore("magnitude", "amount", 0.9, {"pct": 0.99, "ref": (1.0, 2.0)}),),
        reasons=(
            Reason(
                detector="magnitude",
                field="amount",
                observed=float("inf"),
                baseline_summary=BaselineSummary(median=1.0, iqr=0.5, n=40),
                percentile=0.99,
                blast_radius=None,
                prereq_missing=None,
            ),
        ),
        latency_us=12.5,
        threshold=0.6,
        decision=Decision.BLOCK,
        field_types={"amount": FieldType.NUMERIC},
    )
    sink.write_verdict("env_1", res)
    line = (tmp_path / "verdicts.jsonl").read_bytes().splitlines()[0]
    assert line == orjson.dumps(orjson.loads(line), option=orjson.OPT_SORT_KEYS)
    v = orjson.loads(line)
    assert v["envelope_id"] == "env_1" and v["mode"] == "shadow"
    r = v["result"]
    assert r["fused_score"] == 0.75 and r["decision"] == "block" and r["threshold"] == 0.6
    assert r["per_detector"][0]["evidence"] == {"pct": 0.99, "ref": [1.0, 2.0]}
    assert r["reasons"][0]["baseline_summary"] == {"median": 1.0, "iqr": 0.5, "n": 40}
    assert r["reasons"][0]["observed"] is None  # non-finite floats become null
    assert r["field_types"] == {"amount": "numeric"}
