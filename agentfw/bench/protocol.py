"""Detector protocols the harness drives (spec 2.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentfw.core.envelope import ActionEnvelope, DetectorResult, Outcome

__all__ = ["BatchFit", "Detector"]


class Detector(Protocol):
    def reset(self) -> None: ...

    def score(self, e: ActionEnvelope) -> DetectorResult:
        """Must not mutate detector state."""
        ...

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None: ...

    def state_bytes(self) -> int: ...


@runtime_checkable
class BatchFit(Protocol):
    def fit(self, train: Sequence[ActionEnvelope]) -> None: ...
