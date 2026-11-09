from __future__ import annotations

import math

import numpy as np
import orjson
import pytest
import scipy.stats
from hypothesis import given
from hypothesis import strategies as st

from agentfw.core.stats import P2Quantiles, p2_tail


def test_p2_converges_on_lognormal_logspace() -> None:
    rng = np.random.default_rng(0)
    est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    for y in rng.normal(3.6, 0.7, 1000):
        est.add(float(y))
    for p, tol in [(0.25, 0.1), (0.5, 0.08), (0.75, 0.1), (0.05, 0.2), (0.95, 0.2)]:
        assert abs(est.quantile(p) - scipy.stats.norm.ppf(p, 3.6, 0.7)) < tol


def test_p2_exact_below_marker_count() -> None:
    est = P2Quantiles((0.5,))
    for x in (3.0, 1.0):
        est.add(x)
    assert est.quantile(0.5) == 2.0


def test_p2_buffer_matches_numpy_linear() -> None:
    est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))  # 7 markers -> exact for n < 7
    xs = [5.0, -2.0, 9.5, 0.25, 3.0, 3.0]
    for i, x in enumerate(xs, start=1):
        est.add(x)
        assert est.n == i
        for p in (0.0, 0.1, 0.33, 0.5, 0.8, 1.0):
            assert est.quantile(p) == pytest.approx(float(np.quantile(xs[:i], p)), abs=1e-12)


def test_p2_tail_995_on_uniform() -> None:
    rng = np.random.default_rng(1)
    est = p2_tail(0.995)
    for x in rng.random(100_000):
        est.add(float(x))
    assert abs(est.quantile(0.995) - 0.995) < 0.003


def test_p2_tail_marker_layout() -> None:
    assert p2_tail(0.99).probs == pytest.approx((0.495, 0.99, 0.995))


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=500))
def test_p2_markers_monotone_and_serializable(xs: list[float]) -> None:
    est = P2Quantiles((0.25, 0.5, 0.75))
    for x in xs:
        est.add(x)
    qs = [est.quantile(p) for p in (0.25, 0.5, 0.75)]
    assert qs == sorted(qs) and P2Quantiles.from_dict(est.to_dict()) == est
    heights = est.to_dict()["q"]
    assert heights == sorted(heights)
    assert min(xs) <= est.quantile(0.0) <= est.quantile(1.0) <= max(xs)


def test_p2_json_round_trip_is_exact() -> None:
    rng = np.random.default_rng(2)
    est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    for y in rng.standard_t(3, 2000):
        est.add(float(y))
    clone = P2Quantiles.from_dict(orjson.loads(orjson.dumps(est.to_dict())))
    assert clone == est
    for y in rng.standard_t(3, 500):  # identical futures after the round trip
        est.add(float(y))
        clone.add(float(y))
    assert clone == est and clone.quantile(0.9) == est.quantile(0.9)


def test_p2_eq_compares_full_state() -> None:
    a, b = P2Quantiles((0.5,)), P2Quantiles((0.5,))
    assert a == b
    a.add(1.0)
    assert a != b
    b.add(1.0)
    assert a == b
    assert P2Quantiles((0.5,)) != P2Quantiles((0.25,))


def test_p2_constant_memory() -> None:
    rng = np.random.default_rng(3)
    est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    xs = rng.normal(0.0, 1.0, 100_000)
    for x in xs[:1000]:
        est.add(float(x))
    d_small = est.to_dict()
    b_small = orjson.dumps(d_small)
    for x in xs[1000:]:
        est.add(float(x))
    d_big = est.to_dict()
    b_big = orjson.dumps(d_big)
    # Same record shape regardless of n; only number formatting (counter digits,
    # shortest float repr) may differ, so the byte size is bounded, not literally equal.
    assert d_small.keys() == d_big.keys()
    for k, v in d_small.items():
        if isinstance(v, list):
            assert len(v) == len(d_big[k]), k
    assert abs(len(b_big) - len(b_small)) <= 32
    assert len(b_big) <= 512


def test_p2_ignores_non_finite() -> None:
    est, ref = P2Quantiles((0.5,)), P2Quantiles((0.5,))
    for x in (1.0, math.nan, 2.0, math.inf, -math.inf, 3.0, 4.0, math.nan, 5.0):
        est.add(x)
    for x in (1.0, 2.0, 3.0, 4.0, 5.0):
        ref.add(x)
    assert est.n == 5 and est == ref
    assert all(math.isfinite(v) for v in est.to_dict()["q"])


def test_p2_empty_is_nan() -> None:
    est = P2Quantiles((0.5,))
    assert est.n == 0
    assert math.isnan(est.quantile(0.5)) and math.isnan(est.percentile_of(1.0))


def test_p2_single_sample() -> None:
    est = P2Quantiles((0.5,))
    est.add(7.0)
    assert est.quantile(0.0) == est.quantile(1.0) == 7.0
    assert est.percentile_of(6.0) == 0.0 and est.percentile_of(8.0) == 1.0
    assert est.percentile_of(7.0) == 0.5


def test_p2_percentile_inverts_quantile_and_clamps() -> None:
    rng = np.random.default_rng(4)
    est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    xs = rng.random(20_000)
    for x in xs:
        est.add(float(x))
    for p in (0.01, 0.05, 0.2, 0.5, 0.6, 0.9, 0.95, 0.99):
        assert est.percentile_of(est.quantile(p)) == pytest.approx(p, abs=1e-9)
    assert est.percentile_of(float(xs.min()) - 1.0) == 0.0
    assert est.percentile_of(float(xs.max()) + 1.0) == 1.0
    assert est.quantile(0.0) == float(xs.min()) and est.quantile(1.0) == float(xs.max())
    ps = [est.percentile_of(float(x)) for x in np.linspace(-0.5, 1.5, 101)]
    assert ps == sorted(ps) and all(0.0 <= v <= 1.0 for v in ps)


def test_p2_constant_stream() -> None:
    est = P2Quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    for _ in range(1000):
        est.add(4.2)
    assert est.quantile(0.5) == 4.2 and est.quantile(0.99) == 4.2
    assert est.percentile_of(4.2) == 0.5
    assert est.percentile_of(4.1) == 0.0 and est.percentile_of(4.3) == 1.0


@pytest.mark.parametrize("probs", [(), (0.0,), (1.0,), (0.5, 0.5), (0.7, 0.3), (math.nan,)])
def test_p2_rejects_bad_probs(probs: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        P2Quantiles(probs)


def test_p2_rejects_bad_quantile_arg() -> None:
    est = P2Quantiles((0.5,))
    est.add(1.0)
    for p in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError):
            est.quantile(p)


def test_p2_from_dict_rejects_inconsistent_state() -> None:
    est = P2Quantiles((0.25, 0.5, 0.75))
    for x in range(20):
        est.add(float(x))
    d = est.to_dict()
    bad = dict(d, q=d["q"][:-1])
    with pytest.raises(ValueError):
        P2Quantiles.from_dict(bad)
