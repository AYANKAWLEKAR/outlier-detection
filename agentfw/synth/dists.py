"""Distribution sampling and Zipf id pools for the synthetic generator.

Design notes:

* All randomness comes from the ``numpy.random.Generator`` passed in.
* ``zipf`` is a *truncated* Zipf over ``vocab_size`` ranks (p(k) proportional to ``(k+1)**-s``)
  and returns a 0-based rank.
* ``categorical`` draws a Dirichlet(``dirichlet_alpha * weights``) once per *context* and reuses
  it, so rare vocabulary values exist in benign traffic. The context is the caller-owned ``cache``
  dict plus a ``key`` (the walker passes one cache per persona and keys such as
  ``"stripe.create_refund.reason"``). Without a cache each call draws fresh weights.
* ``IdPool`` ids are ``prefix`` + 12+ uppercase alphanumerics, so
  ``agentfw.core.patterns.is_id_like`` recognises them (``tk_...`` via the prefixed form, 3-char
  Salesforce key prefixes via the 15-char form).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from agentfw.synth.schema import DistSpec

__all__ = ["IdPool", "Sampled", "WeightsCache", "new_id", "sample"]

Sampled = float | int | str | bool
WeightsCache = dict[str, NDArray[np.float64]]

_ALPHABET = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
_SUFFIX_MIN = 12


@lru_cache(maxsize=256)
def _zipf_cdf(n: int, s: float) -> NDArray[np.float64]:
    p = np.arange(1, n + 1, dtype=np.float64) ** -s
    return np.cumsum(p / p.sum())


def _zipf_draw(rng: np.random.Generator, n: int, s: float) -> int:
    return min(int(np.searchsorted(_zipf_cdf(n, s), rng.random(), side="right")), n - 1)


def _categorical(
    spec: DistSpec, rng: np.random.Generator, cache: WeightsCache | None, key: str | None
) -> str:
    vocab = spec.vocab or []
    if cache is not None and key is not None and key in cache:
        w = cache[key]
    else:
        base = np.asarray(spec.weights if spec.weights is not None else np.ones(len(vocab)), float)
        alpha = np.maximum(spec.dirichlet_alpha * base / base.sum(), 1e-6)
        w = rng.dirichlet(alpha)
        if cache is not None and key is not None:
            cache[key] = w
    return vocab[
        min(int(np.searchsorted(np.cumsum(w), rng.random(), side="right")), len(vocab) - 1)
    ]


def _clip(x: float, spec: DistSpec) -> float:
    if spec.min is not None:
        x = max(x, spec.min)
    if spec.max is not None:
        x = min(x, spec.max)
    return x


def sample(
    spec: DistSpec,
    rng: np.random.Generator,
    *,
    cache: WeightsCache | None = None,
    key: str | None = None,
) -> Sampled:
    """Draw one value; numeric families clip to ``min``/``max``, ``round_cents`` rounds to 2 dp."""
    fam = spec.family
    if fam == "categorical":
        return _categorical(spec, rng, cache, key)
    if fam == "constant":
        assert spec.value is not None
        return spec.value
    if fam == "bernoulli":
        assert spec.p is not None
        return bool(rng.random() < spec.p)
    if fam == "zipf":
        assert spec.vocab_size is not None and spec.s is not None
        return _zipf_draw(rng, spec.vocab_size, spec.s)
    x: float
    if fam == "lognormal":
        assert spec.mu is not None and spec.sigma is not None
        x = float(np.exp(rng.normal(spec.mu, spec.sigma)))  # drawn in log space
    elif fam == "gamma":
        assert spec.shape is not None and spec.scale is not None
        x = float(rng.gamma(spec.shape, spec.scale))
    elif fam == "poisson":
        assert spec.lam is not None
        return int(_clip(float(rng.poisson(spec.lam)), spec))
    else:  # uniform
        assert spec.min is not None and spec.max is not None
        x = float(rng.uniform(spec.min, spec.max))
    x = _clip(x, spec)
    return round(x, 2) if spec.round_cents else x


def _suffixes(rng: np.random.Generator, n: int, length: int) -> list[str]:
    codes = rng.integers(0, len(_ALPHABET), size=(n, length))
    return ["".join(row) for row in _ALPHABET[codes]]


def _suffix_len(prefix: str) -> int:
    return max(_SUFFIX_MIN, 15 - len(prefix))


def new_id(prefix: str, rng: np.random.Generator) -> str:
    """A fresh id-like string (not tied to any pool)."""
    return prefix + _suffixes(rng, 1, _suffix_len(prefix))[0]


class IdPool:
    """A fixed pool of ids drawn with Zipf popularity so hot ids recur (spec 4.5)."""

    def __init__(self, prefix: str, size: int, s: float, rng: np.random.Generator) -> None:
        self.prefix = prefix
        self.s = s
        self._rng = rng
        length = _suffix_len(prefix)
        seen: dict[str, None] = {}
        while len(seen) < size:
            for suf in _suffixes(rng, size - len(seen), length):
                seen.setdefault(prefix + suf)
        self.ids: tuple[str, ...] = tuple(seen)
        self._members = frozenset(self.ids)

    def draw(self) -> str:
        return self.ids[_zipf_draw(self._rng, len(self.ids), self.s)]

    def fabricate(self, rng: np.random.Generator) -> str:
        """An id with the pool's prefix that is guaranteed not to be a pool member."""
        while (cand := new_id(self.prefix, rng)) in self._members:
            pass  # pragma: no cover - astronomically unlikely
        return cand
