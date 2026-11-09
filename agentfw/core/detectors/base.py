"""Sub-detector protocol and shared scoring helpers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

from agentfw.core.envelope import ActionEnvelope, DetectorScore, Scalar
from agentfw.core.schema_infer import TypeView

if TYPE_CHECKING:
    from agentfw.core.detectors.magnitude import MagRecord

__all__ = ["PriorSource", "SubDetector", "iter_fields", "signed_log1p", "squash"]


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


def squash(z: float, z0: float, tau: float) -> float:
    """``1 / (1 + exp(-(z - z0) / tau))``, computed without overflow."""
    t = (z - z0) / tau
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))  # exp(-t) <= 1: never overflows
    ex = math.exp(t)  # t < 0: underflows harmlessly to 0.0
    return ex / (1.0 + ex)


def signed_log1p(x: float) -> float:
    """``sign(x) * log1p(|x|)``."""
    return math.copysign(math.log1p(abs(x)), x)


def iter_fields(e: ActionEnvelope) -> Iterator[tuple[str, tuple[Scalar, ...]]]:
    """Flat argument paths, then derived paths."""
    yield from e.flat_args.items()
    yield from e.derived.items()
