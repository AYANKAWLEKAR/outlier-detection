from __future__ import annotations

import pytest

from agentfw.bench.baselines.random_ import RandomDetector
from agentfw.bench.baselines.static_threshold import StaticThreshold
from agentfw.bench.protocol import BatchFit, Detector
from agentfw.bench.registry import DETECTORS, make_detector
from agentfw.core.envelope import ActionEnvelope, Outcome
from tests.factories import make_env


def _train() -> list[ActionEnvelope]:
    return [
        make_env(args={"amount": float(a), "reason": r})
        for a, r in zip(range(10, 101), ["duplicate", "fraudulent"] * 51, strict=False)
    ]


def test_random_deterministic_and_unit_interval() -> None:
    det = RandomDetector(seed=3)
    e = make_env(args={"amount": 1.0})
    assert det.score(e).fused_score == det.score(e).fused_score
    assert RandomDetector(seed=3).score(e).fused_score == det.score(e).fused_score
    assert RandomDetector(seed=4).score(e).fused_score != det.score(e).fused_score
    scores = [det.score(make_env(args={"amount": 1.0})).fused_score for _ in range(10_000)]
    assert all(0.0 <= s < 1.0 for s in scores)
    assert abs(sum(scores) / len(scores) - 0.5) < 0.02
    assert det.state_bytes() == 0


def test_static_threshold_numeric_bound() -> None:
    det = StaticThreshold()
    assert isinstance(det, BatchFit)
    typed: Detector = det  # structural conformance, checked by mypy
    assert typed is det
    det.fit(_train())
    big = det.score(make_env(args={"amount": 1000.0, "reason": "duplicate"}))
    assert big.fused_score == 1.0
    assert big.per_detector and big.per_detector[0].detector == "static_threshold"
    assert big.per_detector[0].field == "amount"
    assert big.threshold is None
    assert det.score(make_env(args={"amount": 50.0, "reason": "duplicate"})).fused_score == 0.0


def test_static_threshold_unseen_category_and_unseen_path() -> None:
    det = StaticThreshold()
    det.fit(_train())
    assert det.score(make_env(args={"amount": 50.0, "reason": "chargeback"})).fused_score == 1.0
    # never seen in train: no bound, no violation
    assert det.score(make_env(args={"amount": 5.0, "note_x": 99999})).fused_score == 0.0
    assert det.score(make_env(tool_id="other.tool", args={"amount": 1e9})).fused_score == 0.0


def test_static_threshold_update_noop_and_reset() -> None:
    det = StaticThreshold()
    det.fit(_train())
    before = det.state_bytes()
    assert before > 0
    det.update(make_env(args={"amount": 5000.0, "reason": "zzz"}), Outcome.ALLOWED)
    assert det.state_bytes() == before
    assert det.score(make_env(args={"amount": 5000.0})).fused_score == 1.0
    det.reset()
    assert det.state_bytes() < before
    assert det.score(make_env(args={"amount": 5000.0})).fused_score == 0.0


def test_registry() -> None:
    assert set(DETECTORS) >= {"random", "static_threshold"}
    assert isinstance(make_detector("random", seed=1), RandomDetector)
    assert isinstance(make_detector("static_threshold", quantile=0.9), StaticThreshold)
    with pytest.raises(KeyError, match="random"):
        make_detector("nope")
