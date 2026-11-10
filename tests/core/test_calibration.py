from __future__ import annotations

import math

import pytest

from agentfw.core.calibration import Calibrator, CalRecord
from agentfw.core.config import EngineConfig
from agentfw.core.envelope import Decision, Outcome
from agentfw.core.state import DictStore, key_cal


@pytest.fixture
def cal() -> Calibrator:
    return Calibrator(DictStore(), EngineConfig())


def test_threshold_blend(cal: Calibrator) -> None:
    assert cal.threshold("a") == 0.6
    for _ in range(250):
        cal.observe("a", 0.1, Outcome.ALLOWED)
    assert cal.threshold("a") == pytest.approx(0.5 * 0.1 + 0.5 * 0.6, abs=1e-6)


def test_threshold_full_weight_after_blend_n(cal: Calibrator) -> None:
    for _ in range(600):
        cal.observe("a", 0.1, Outcome.ALLOWED)
    assert cal.threshold("a") == pytest.approx(0.1, abs=1e-6)


def test_threshold_is_pure(cal: Calibrator) -> None:
    store = DictStore()
    c = Calibrator(store, EngineConfig())
    c.threshold("a")
    assert len(store) == 0
    c.observe("a", 0.1, Outcome.ALLOWED)
    before = store.nbytes()
    for _ in range(3):
        c.threshold("a")
        c.decide("a", 0.5)
    assert store.nbytes() == before


def test_decision_bands(cal: Calibrator) -> None:
    assert cal.decide("a", 0.6) is Decision.BLOCK
    assert cal.decide("a", 0.95) is Decision.BLOCK
    assert cal.decide("a", 0.59) is Decision.REQUIRE_APPROVAL
    assert cal.decide("a", 0.5) is Decision.REQUIRE_APPROVAL
    assert cal.decide("a", 0.49) is Decision.ALLOW
    assert cal.decide("a", 0.0) is Decision.ALLOW


def test_blocked_and_denied_not_learned_approved_is(cal: Calibrator) -> None:
    cal.observe("a", 0.9, Outcome.BLOCKED)
    cal.observe("a", 0.9, Outcome.DENIED)
    assert cal.threshold("a") == 0.6
    rec = cal.store.get(key_cal("a"))
    assert rec is None or (isinstance(rec, CalRecord) and rec.n_scored == 0)
    cal.observe("a", 0.2, Outcome.APPROVED)
    rec = cal.store.get(key_cal("a"))
    assert isinstance(rec, CalRecord) and rec.n_scored == 1


def test_non_finite_fused_skipped(cal: Calibrator) -> None:
    cal.observe("a", math.nan, Outcome.ALLOWED)
    cal.observe("a", math.inf, Outcome.ALLOWED)
    assert cal.threshold("a") == 0.6
    rec = cal.store.get(key_cal("a"))
    assert rec is None or (isinstance(rec, CalRecord) and rec.n_scored == 0)


def test_agents_independent(cal: Calibrator) -> None:
    for _ in range(500):
        cal.observe("a", 0.1, Outcome.ALLOWED)
    assert cal.threshold("b") == 0.6
    assert cal.threshold("a") == pytest.approx(0.1, abs=1e-6)


def test_cal_record_roundtrip() -> None:
    rec = CalRecord.new(EngineConfig())
    for v in (0.1, 0.2, 0.3, 0.15, 0.25, 0.05):
        rec.q.add(v)
        rec.n_scored += 1
    back = CalRecord.from_dict(rec.to_dict())
    assert back.n_scored == rec.n_scored
    assert back.to_dict() == rec.to_dict()
