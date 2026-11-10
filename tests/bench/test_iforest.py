from __future__ import annotations

import numpy as np

from agentfw.bench.baselines.isolation_forest import IsolationForestBaseline
from agentfw.bench.protocol import BatchFit, Detector
from agentfw.bench.registry import make_detector
from agentfw.core.envelope import ActionEnvelope, Outcome
from tests.factories import make_env


def _train(n: int = 200, tool: str = "stripe.create_refund") -> list[ActionEnvelope]:
    rng = np.random.default_rng(0)
    return [
        make_env(
            tool_id=tool,
            args={"amount": float(rng.normal(100, 5)), "reason": ["dup", "fraud"][i % 2]},
        )
        for i in range(n)
    ]


def test_injected_outlier_ranks_above_all_inliers() -> None:
    train = _train()
    outlier = make_env(args={"amount": 5000.0, "reason": "dup"})  # 50x the typical amount
    det = IsolationForestBaseline(seed=0)
    assert isinstance(det, BatchFit)
    typed: Detector = det
    assert typed is det
    det.fit([*train[:-1], outlier])  # injected into the toy data
    top = det.score(outlier).fused_score
    inliers = [det.score(e).fused_score for e in train[:-1]]
    assert top == 1.0
    assert top > max(inliers)
    assert min(inliers) >= 0.0 and max(inliers) <= 1.0


def test_held_out_extreme_outlier_scores_above_typical_inliers() -> None:
    # IF cannot rank extrapolation above the train tail (a known weakness), but it must still
    # score a far-out point above the typical in-distribution one.
    det = IsolationForestBaseline(seed=0)
    det.fit(_train())
    far = det.score(make_env(args={"amount": 5000.0, "reason": "dup"})).fused_score
    centre = [
        det.score(make_env(args={"amount": float(a), "reason": "fraud"})).fused_score
        for a in np.linspace(97, 103, 30)
    ]
    assert far > max(centre)


def test_deterministic_per_seed() -> None:
    probe = make_env(args={"amount": 140.0, "reason": "dup"})

    def s(seed: int) -> float:
        det = IsolationForestBaseline(seed=seed)
        det.fit(_train())
        return det.score(probe).fused_score

    assert s(3) == s(3)
    assert s(3) != s(4)


def test_untrained_tool_scores_zero() -> None:
    det = IsolationForestBaseline(min_train=50)
    det.fit(_train(10))
    res = det.score(make_env(args={"amount": 9e9}))
    assert res.fused_score == 0.0
    assert res.per_detector[0].evidence == {"untrained": True}
    det.fit(_train())
    unseen = det.score(make_env(tool_id="other.tool", args={"amount": 9e9}))
    assert unseen.fused_score == 0.0 and unseen.per_detector[0].evidence == {"untrained": True}


def test_score_and_update_do_not_change_state_and_reset() -> None:
    det = IsolationForestBaseline()
    det.fit(_train())
    before = det.state_bytes()
    assert before > 0
    e = make_env(args={"amount": 5000.0, "reason": "zzz"})
    r1 = det.score(e).fused_score
    det.update(e, Outcome.ALLOWED)
    assert det.state_bytes() == before
    assert det.score(e).fused_score == r1
    det.reset()
    assert det.score(e).fused_score == 0.0


def test_high_cardinality_categorical_ignored_and_missing_filled() -> None:
    train = [make_env(args={"amount": 10.0 + i % 7, "tag": f"t{i}"}) for i in range(100)]
    det = IsolationForestBaseline(max_onehot=20)
    det.fit(train)
    res = det.score(make_env(args={"tag": "brand-new"}))  # amount missing -> median
    assert 0.0 <= res.fused_score <= 1.0


def test_registry_entry() -> None:
    assert isinstance(make_detector("isolation_forest", seed=1), IsolationForestBaseline)
