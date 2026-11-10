"""Threshold-saturation diagnostic (Task 1.14 fix round 1)."""

from __future__ import annotations

import orjson
import pytest

from agentfw.bench.diagnostics import threshold_saturation
from agentfw.bench.replay import ScoreRow

Top = tuple[str, str | None, dict[str, object]]


def _row(i: int, agent: str, fused: float, thr: float | None, top: Top, **kw: object) -> ScoreRow:
    det, field, ev = top
    per = orjson.dumps(
        [
            {"detector": det, "field": field, "score": fused, "evidence": ev},
            {"detector": "novelty", "field": "x", "score": 0.0, "evidence": {}},
        ]
    ).decode()
    base: dict[str, object] = {
        "envelope_id": f"e{i}", "session_id": f"s{i}", "agent_id": agent, "tool_id": "t",
        "ts_unix_ms": i, "split": "train", "fused_score": fused, "per_detector_json": per,
        "label": "benign", "attack_family": None, "attack_id": None, "injection_level": None,
        "strength": None, "step": None, "control_type": None, "is_warm": True, "n_key_obs": 100,
        "latency_us": 1.0, "threshold": thr,
    }  # fmt: skip
    base.update(kw)
    return ScoreRow(**base)  # type: ignore[arg-type]


VEL: Top = ("velocity", None, {"signal": "count"})
SEQ: Top = ("sequence", None, {"fired": "trans"})
MAG: Top = ("magnitude", "amount", {})


def test_threshold_saturation_per_agent_and_top1_attribution() -> None:
    rows = [
        _row(0, "A", 0.2, 0.6, MAG),
        _row(1, "A", 1.0, 0.8, VEL),
        _row(2, "A", 1.0, 0.9, SEQ),
        _row(3, "A", 0.3, 1.0, MAG),  # last A row: final live threshold 1.0
        _row(4, "A", 1.0, 1.0, MAG, label="attack"),  # attack: counted apart from benign
        _row(9, "A", 0.995, 1.0, MAG, label="attack"),
        _row(5, "B", 0.1, 0.7, MAG),
        _row(6, "B", 1.0, 0.95, MAG),
        _row(7, "B", float("nan"), None, MAG),  # unscored: ignored
    ]
    rep = threshold_saturation(rows)
    a, b = rep.agents["A"], rep.agents["B"]
    assert a.final_threshold == pytest.approx(1.0) and b.final_threshold == pytest.approx(0.95)
    assert (a.n_benign, a.n_one) == (4, 2) and a.share_one == pytest.approx(0.5)
    assert (b.n_benign, b.n_one) == (2, 1)
    # The replay allows every row, so the calibrator also learns attack scores: report them.
    assert (a.n_attack, a.n_attack_one, a.n_attack_ge99) == (2, 1, 2)
    assert (a.n_benign_ge99, b.n_attack) == (2, 0)
    assert rep.n_saturated == 1  # final threshold >= 0.999: A only
    assert rep.top1 == {"velocity:count": 1, "sequence:trans": 1, "magnitude:amount": 1}


def test_threshold_saturation_empty() -> None:
    rep = threshold_saturation([])
    assert rep.agents == {} and rep.n_saturated == 0 and rep.top1 == {}
