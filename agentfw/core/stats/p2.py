"""P² streaming quantiles (Jain & Chlamtac 1985), generalised to arbitrary marker probabilities.

Markers sit at probabilities ``(0, *probs, 1)``: the two extremes track the exact min and max,
and each interior marker tracks one requested quantile. After ``n`` samples the desired
position of marker ``i`` is ``1 + (n - 1) * p_i`` (i.e. the desired-position increment is
``dn'_i = p_i``); it is computed from ``n`` instead of being accumulated, so no extra state is
stored. Heights move by the piecewise-parabolic (P²) formula, falling back to linear
interpolation whenever the parabolic estimate would not lie strictly between the neighbouring
markers, so marker heights stay non-decreasing.

Until ``len(probs) + 2`` samples have arrived the estimator keeps the sorted samples and is
exact: ``quantile`` then matches ``numpy.quantile(..., method="linear")``.

``quantile(p)`` interpolates linearly between adjacent markers, placing marker ``i`` at its
position-normalised probability ``(pos_i - 1) / (n - 1)`` (0 for the min, 1 for the max).
``percentile_of(x)`` is its inverse, clamped to 0 below the min and 1 above the max; an ``x``
equal to a run of equal markers maps to the midpoint of that run's probabilities.

Non-finite inputs (NaN, ±inf) are ignored by ``add`` — callers filter them, but the
estimator must never be corrupted. Both queries return NaN on an empty estimator.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field
from typing import Any, Self

__all__ = ["P2Quantiles", "p2_tail"]


@dataclass(slots=True)
class P2Quantiles:
    probs: tuple[float, ...]
    _n: int = field(default=0, init=False)
    _q: list[float] = field(default_factory=list, init=False)  # marker heights
    _pos: list[int] = field(default_factory=list, init=False)  # 1-based marker positions
    _buf: list[float] = field(default_factory=list, init=False)  # sorted, until warm
    _dn: tuple[float, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        probs = tuple(float(p) for p in self.probs)
        if not probs:
            raise ValueError("probs must be non-empty")
        if not all(0.0 < p < 1.0 for p in probs):  # also rejects NaN
            raise ValueError(f"probs must lie strictly inside (0, 1), got {probs!r}")
        if any(a >= b for a, b in zip(probs, probs[1:], strict=False)):
            raise ValueError(f"probs must be strictly increasing, got {probs!r}")
        self.probs = probs
        self._dn = (0.0, *probs, 1.0)

    @property
    def n(self) -> int:
        """Number of (finite) samples added."""
        return self._n

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            return
        x = float(x)
        m = len(self._dn)
        if self._n < m:
            insort(self._buf, x)
            self._n += 1
            if self._n == m:
                self._q = self._buf
                self._pos = list(range(1, m + 1))
                self._buf = []
            return

        q, pos, dn = self._q, self._pos, self._dn
        if x < q[0]:
            q[0] = x
            k = 0
        elif x >= q[-1]:
            q[-1] = x
            k = m - 2
        else:
            k = bisect_right(q, x) - 1  # q[k] <= x < q[k+1]
        for i in range(k + 1, m):
            pos[i] += 1
        self._n += 1
        nm1 = self._n - 1

        for i in range(1, m - 1):
            d = 1.0 + nm1 * dn[i] - pos[i]
            gap_up = pos[i + 1] - pos[i]
            gap_down = pos[i] - pos[i - 1]
            if (d >= 1.0 and gap_up > 1) or (d <= -1.0 and gap_down > 1):
                s = 1 if d > 0 else -1
                qi, q_lo, q_hi = q[i], q[i - 1], q[i + 1]
                qp = qi + s / (pos[i + 1] - pos[i - 1]) * (
                    (gap_down + s) * (q_hi - qi) / gap_up + (gap_up - s) * (qi - q_lo) / gap_down
                )
                if not q_lo < qp < q_hi:
                    j = i + s
                    qp = qi + s * (q[j] - qi) / (pos[j] - pos[i])
                # Guard against float rounding (or overflow) breaking marker monotonicity.
                q[i] = min(max(qp, q_lo), q_hi)
                pos[i] += s

    def _grid(self) -> tuple[list[float], list[float]]:
        """(heights, probabilities) of the interpolation knots; requires n >= 1."""
        if self._buf:
            h = self._buf
            k = len(h)
            return h, ([i / (k - 1) for i in range(k)] if k > 1 else [0.5])
        nm1 = self._n - 1
        return self._q, [(p - 1) / nm1 for p in self._pos]

    def quantile(self, p: float) -> float:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        if self._n == 0:
            return math.nan
        h, g = self._grid()
        if p <= g[0]:
            return h[0]
        if p >= g[-1]:
            return h[-1]
        j = bisect_right(g, p) - 1  # g[j] <= p < g[j+1]
        t = (p - g[j]) / (g[j + 1] - g[j])
        return min(max(h[j] + t * (h[j + 1] - h[j]), h[j]), h[j + 1])

    def percentile_of(self, x: float) -> float:
        if self._n == 0 or math.isnan(x):
            return math.nan
        h, g = self._grid()
        if x < h[0]:
            return 0.0
        if x > h[-1]:
            return 1.0
        lo = bisect_left(h, x)  # first knot >= x
        hi = bisect_right(h, x) - 1  # last knot <= x
        if lo <= hi:  # x equals one or more knots
            return (g[lo] + g[hi]) / 2.0
        t = (x - h[hi]) / (h[lo] - h[hi])
        return min(max(g[hi] + t * (g[lo] - g[hi]), g[hi]), g[lo])

    def to_dict(self) -> dict[str, Any]:
        return {
            "probs": list(self.probs),
            "n": self._n,
            "q": list(self._q),
            "pos": list(self._pos),
            "buf": list(self._buf),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        est = cls(tuple(float(p) for p in d["probs"]))
        n = int(d["n"])
        q = [float(v) for v in d["q"]]
        pos = [int(v) for v in d["pos"]]
        buf = [float(v) for v in d["buf"]]
        m = len(est._dn)
        if n < 0:
            raise ValueError("n must be >= 0")
        if n < m:
            ok = not q and not pos and len(buf) == n and buf == sorted(buf)
        else:
            ok = (
                not buf
                and len(q) == len(pos) == m
                and q == sorted(q)
                and pos[0] == 1
                and pos[-1] == n
                and all(a < b for a, b in zip(pos, pos[1:], strict=False))
            )
        if not ok or not all(math.isfinite(v) for v in q + buf):
            raise ValueError("inconsistent P2Quantiles state")
        est._n, est._q, est._pos, est._buf = n, q, pos, buf
        return est


def p2_tail(p: float) -> P2Quantiles:
    """Three-marker estimator focused on the tail quantile ``p``: markers ``(p/2, p, (1+p)/2)``."""
    return P2Quantiles((p / 2.0, p, (1.0 + p) / 2.0))
