"""Sub-detector protocol and shared scoring helpers."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

import xxhash

from agentfw.core.envelope import ActionEnvelope, DetectorScore, Scalar
from agentfw.core.schema_infer import TypeView

if TYPE_CHECKING:
    from agentfw.core.detectors.magnitude import MagRecord
    from agentfw.core.detectors.novelty import NoveltyRecord

__all__ = [
    "IQR_TO_SIGMA",
    "PriorSource",
    "SubDetector",
    "clamp_finite",
    "iter_fields",
    "normal_sf",
    "posterior",
    "signed_expm1",
    "signed_log1p",
    "squash",
    "value_key",
]

IQR_TO_SIGMA = 1.349  # IQR of a standard normal: robust scale = IQR / 1.349
_MAX = sys.float_info.max
_SQRT2 = math.sqrt(2.0)


class SubDetector(Protocol):
    name: str

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        """Score ``e`` against current state; must not mutate state."""
        ...

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        """Fold ``e`` into state; called only for allowed/approved outcomes."""
        ...


class PriorSource(Protocol):
    """Cross-agent pooled priors (§6.10); implemented by ``PooledPriors`` (Task 1.10)."""

    def magnitude_ref(self, tool: str, path: str, direction: str) -> MagRecord | None: ...

    def novelty_ref(self, tool: str, path: str) -> NoveltyRecord | None: ...


def squash(z: float, z0: float, tau: float) -> float:
    """``1 / (1 + exp(-(z - z0) / tau))``, computed without overflow."""
    t = (z - z0) / tau
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))  # exp(-t) <= 1: never overflows
    ex = math.exp(t)  # t < 0: underflows harmlessly to 0.0
    return ex / (1.0 + ex)


def normal_sf(z: float) -> float:
    """Standard normal upper tail ``Q(z) = erfc(z / sqrt(2)) / 2`` (spec A1.2)."""
    return 0.5 * math.erfc(z / _SQRT2)


def posterior(p: float, prior: float) -> float:
    """``prior / (prior + (1 - prior) * min(p, 1))``: the probability that a value with tail
    probability ``p`` is anomalous, when anomalies have rate ``prior`` and make a value this
    extreme certain (spec A1.2). Exactly 1.0 when ``p`` is 0, negative or NaN."""
    if not p > 0.0:
        return 1.0
    return prior / (prior + (1.0 - prior) * min(p, 1.0))


def signed_log1p(x: float) -> float:
    """``sign(x) * log1p(|x|)``."""
    return math.copysign(math.log1p(abs(x)), x)


def signed_expm1(y: float) -> float:
    """Inverse of ``signed_log1p``, saturating at ``±float max`` instead of overflowing."""
    try:
        return math.copysign(math.expm1(abs(y)), y)
    except OverflowError:
        return math.copysign(_MAX, y)


def clamp_finite(v: float) -> float:
    """Clamp ``v`` into ``[-float max, float max]`` (maps ``±inf`` to the extremes)."""
    return max(-_MAX, min(_MAX, v))


def value_key(v: Scalar, max_len: int) -> str:
    """Categorical key of a value: ``str(v)``, or ``#<xxh64 hex>`` of it when that is ``max_len``
    or more characters (long content is never stored). An int beyond the interpreter's
    str-digit limit, which ``str`` refuses, is keyed by its (always hashed) hex form. Equal
    values keep equal keys."""
    try:
        s = str(v)
    except ValueError:  # int beyond the str-digit limit; hex conversion has no limit
        s = hex(v) if isinstance(v, int) else repr(type(v))
    if len(s) >= max_len:
        return "#" + xxhash.xxh64_hexdigest(s.encode("utf-8", "surrogatepass"))
    return s


def iter_fields(e: ActionEnvelope) -> Iterator[tuple[str, tuple[Scalar, ...]]]:
    """Flat argument paths, then derived paths."""
    yield from e.flat_args.items()
    yield from e.derived.items()
