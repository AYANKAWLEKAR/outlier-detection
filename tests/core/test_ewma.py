from __future__ import annotations

import math

import numpy as np
import orjson
import pytest

from agentfw.core.stats import Ewma


def test_ewma_empty_then_first_value() -> None:
    e = Ewma(0.02)
    assert e.value is None
    e.add(5.0)
    assert e.value == 5.0


def test_ewma_matches_closed_form() -> None:
    # adjust=False EWMA: weight (1-a)^(N-1) on x_0 and a(1-a)^(N-1-t) on x_t for t >= 1.
    rng = np.random.default_rng(0)
    xs = rng.normal(3.0, 2.0, 500)
    e = Ewma(0.02)
    for x in xs:
        e.add(float(x))
    a, n = 0.02, len(xs)
    w = a * (1 - a) ** (n - 1 - np.arange(n))
    w[0] = (1 - a) ** (n - 1)
    expected = float(np.dot(w, xs))
    assert e.value == pytest.approx(expected, rel=1e-12)


def test_ewma_converges_to_constant() -> None:
    e = Ewma(0.1)
    e.add(0.0)
    for _ in range(500):
        e.add(10.0)
    assert e.value == pytest.approx(10.0, abs=1e-9)


def test_ewma_ignores_non_finite() -> None:
    e = Ewma(0.5)
    for x in (math.nan, math.inf, 2.0, -math.inf, math.nan, 4.0):
        e.add(x)
    assert e.value == 3.0


def test_ewma_round_trip() -> None:
    e = Ewma(0.02)
    assert Ewma.from_dict(orjson.loads(orjson.dumps(e.to_dict()))) == e
    for x in (1.0, 2.5, -3.25):
        e.add(x)
    clone = Ewma.from_dict(orjson.loads(orjson.dumps(e.to_dict())))
    assert clone == e and clone.value == e.value
    assert Ewma(0.02) != e


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, math.nan])
def test_ewma_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        Ewma(alpha)
