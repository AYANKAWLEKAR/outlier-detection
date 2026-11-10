"""Engine invariants under generated streams (hypothesis).

Review Focus 3 is pinned by ``test_scores_always_finite_in_unit_interval``: brand-new keys
(n = 0), constant fields (IQR = 0), single-tool agents (V = 1), NaN / ±inf and huge ints must
all yield finite scores in [0, 1] with no exception.
"""

from __future__ import annotations

import dataclasses
import math

import orjson
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, DetectorResult, Outcome
from tests.core.strategies import envelope_streams, outcomes

_SETTINGS = settings(deadline=None, max_examples=100, suppress_health_check=[HealthCheck.too_slow])


def strip_latency(r: DetectorResult) -> str:
    """Comparable form of a result without its wall-clock latency (``repr`` so NaN == NaN)."""
    return repr(dataclasses.replace(r, latency_us=0.0))


def snapshot(eng: Engine) -> list[tuple[str, bytes]]:
    """Every record, serialized: catches same-size mutations ``state_bytes`` cannot see."""
    opt = orjson.OPT_SORT_KEYS
    return sorted((k, orjson.dumps(r.to_dict(), option=opt)) for k, r in eng.store.items())


def _in_unit(x: float) -> bool:
    return math.isfinite(x) and 0.0 <= x <= 1.0


@_SETTINGS
@given(envelope_streams())
def test_score_idempotent_and_no_leak(stream: list[ActionEnvelope]) -> None:
    eng = Engine()
    for e in stream[:-1]:
        eng.score(e)
        eng.update(e, Outcome.ALLOWED)
    before, snap = eng.state_bytes(), snapshot(eng)
    a = eng.score(stream[-1])
    b = eng.score(stream[-1])
    assert strip_latency(a) == strip_latency(b)
    assert eng.state_bytes() == before
    assert snapshot(eng) == snap


@_SETTINGS
@given(envelope_streams(include_degenerate=True), st.data())
def test_scores_always_finite_in_unit_interval(
    stream: list[ActionEnvelope], data: st.DataObject
) -> None:
    eng = Engine()
    for e in stream:
        r = eng.score(e)
        assert _in_unit(r.fused_score), r
        for s in r.per_detector:
            assert _in_unit(s.score), s
        assert r.threshold is not None and _in_unit(r.threshold)
        assert r.decision is not None
        assert math.isfinite(r.latency_us) and r.latency_us >= 0.0
        eng.update(e, data.draw(outcomes))
    assert eng.state_bytes() >= 0
