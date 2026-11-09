from __future__ import annotations

import numpy as np
import pytest

from agentfw.core.patterns import is_id_like
from agentfw.synth.dists import IdPool, sample
from agentfw.synth.schema import DistSpec


def _x(spec: DistSpec, rng: np.random.Generator) -> float:
    return float(sample(spec, rng))


def test_lognormal_median() -> None:
    rng = np.random.default_rng(0)
    xs = [_x(DistSpec(family="lognormal", mu=3.6, sigma=0.7), rng) for _ in range(10_000)]
    assert abs(np.median(xs) / np.exp(3.6) - 1) < 0.05


def test_lognormal_is_drawn_in_log_space() -> None:
    rng = np.random.default_rng(1)
    spec = DistSpec(family="lognormal", mu=2.0, sigma=0.5)
    logs = np.log([float(sample(spec, rng)) for _ in range(10_000)])
    assert abs(logs.mean() - 2.0) < 0.02 and abs(logs.std() - 0.5) < 0.02


def test_money_rounded_and_clipped() -> None:
    rng = np.random.default_rng(2)
    spec = DistSpec(family="lognormal", mu=4.0, sigma=1.0, round_cents=True, min=5.0, max=200.0)
    xs = [float(sample(spec, rng)) for _ in range(2_000)]
    assert all(5.0 <= x <= 200.0 for x in xs)
    assert all(abs(x * 100 - round(x * 100)) < 1e-6 for x in xs)


def test_categorical_has_rare_values_and_weights_cached_per_context() -> None:
    spec = DistSpec(
        family="categorical",
        vocab=["duplicate", "fraudulent", "requested_by_customer", "manager_approved"],
        weights=[0.2, 0.1, 0.69, 0.01],
    )
    rng = np.random.default_rng(3)
    cache: dict[str, np.ndarray] = {}
    xs = [sample(spec, rng, cache=cache, key="refund.reason") for _ in range(20_000)]
    assert set(xs) == set(spec.vocab or [])
    assert xs.count("manager_approved") < 0.05 * len(xs)
    assert list(cache) == ["refund.reason"]  # drawn once, reused
    w = cache["refund.reason"].copy()
    sample(spec, rng, cache=cache, key="refund.reason")
    assert np.array_equal(w, cache["refund.reason"])
    other: dict[str, np.ndarray] = {}
    sample(spec, np.random.default_rng(4), cache=other, key="refund.reason")
    assert not np.array_equal(w, other["refund.reason"])  # a new persona context redraws


@pytest.mark.parametrize(
    ("spec", "mean"),
    [
        (DistSpec(family="gamma", shape=2.0, scale=5.0), 10.0),
        (DistSpec(family="poisson", lam=12.0), 12.0),
        (DistSpec(family="uniform", min=2.0, max=4.0), 3.0),
        (DistSpec(family="bernoulli", p=0.3), 0.3),
    ],
)
def test_family_means(spec: DistSpec, mean: float) -> None:
    rng = np.random.default_rng(5)
    xs = [float(sample(spec, rng)) for _ in range(20_000)]
    assert abs(np.mean(xs) - mean) / mean < 0.03


def test_constant_bernoulli_type_and_zipf_range() -> None:
    rng = np.random.default_rng(6)
    assert sample(DistSpec(family="constant", value=20), rng) == 20
    assert isinstance(sample(DistSpec(family="bernoulli", p=0.5), rng), bool)
    zs = [sample(DistSpec(family="zipf", vocab_size=12, s=1.3), rng) for _ in range(5_000)]
    assert min(zs) == 0 and max(zs) == 11 and zs.count(0) > zs.count(5)


def test_idpool_zipf_hot_ids_recur() -> None:
    pool = IdPool("tk_", 1000, 1.1, np.random.default_rng(7))
    draws = [pool.draw() for _ in range(5_000)]
    assert set(draws) <= set(pool.ids)
    assert max(draws.count(i) for i in set(draws)) > 5_000 / 1000 * 20
    assert len(set(pool.ids)) == 1000


@pytest.mark.parametrize("prefix", ["tk_", "ch_", "006", "00T", "cus_"])
def test_idpool_ids_are_id_like_and_fabricate_never_in_pool(prefix: str) -> None:
    pool = IdPool(prefix, 300, 1.1, np.random.default_rng(8))
    rng = np.random.default_rng(9)
    assert all(is_id_like(i) and i.startswith(prefix) for i in pool.ids)
    fakes = [pool.fabricate(rng) for _ in range(2_000)]
    assert all(f.startswith(prefix) and is_id_like(f) and f not in pool.ids for f in fakes)
    assert all(len(f) - len(prefix) >= 10 for f in fakes)


def test_idpool_deterministic() -> None:
    a = IdPool("tk_", 50, 1.1, np.random.default_rng(1))
    b = IdPool("tk_", 50, 1.1, np.random.default_rng(1))
    assert a.ids == b.ids and [a.draw() for _ in range(20)] == [b.draw() for _ in range(20)]
