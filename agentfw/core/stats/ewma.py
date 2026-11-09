"""Exponentially weighted moving average."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

__all__ = ["Ewma"]


@dataclass(slots=True)
class Ewma:
    """``value <- value + alpha * (x - value)``; the first sample initialises ``value``.

    This is the ``adjust=False`` recurrence (pandas ``ewm(alpha=..., adjust=False)``).
    ``value`` is ``None`` until the first finite sample. Non-finite inputs (NaN, ±inf)
    are ignored so the average can never be corrupted; callers are expected to filter
    them anyway.
    """

    alpha: float
    value: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:  # also rejects NaN
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha!r}")

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            return
        if self.value is None:
            self.value = float(x)
        else:
            self.value += self.alpha * (x - self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        v = d["value"]
        return cls(float(d["alpha"]), None if v is None else float(v))
