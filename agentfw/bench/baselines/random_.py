"""Random baseline: a deterministic per-envelope uniform score (spec 5.5)."""

from __future__ import annotations

import time

from agentfw.core.envelope import ActionEnvelope, DetectorResult, Outcome
from agentfw.synth.sampler import stable_hash

__all__ = ["RandomDetector"]


class RandomDetector:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def reset(self) -> None:
        return None

    def score(self, e: ActionEnvelope) -> DetectorResult:
        t0 = time.perf_counter()
        # Top 53 bits -> exactly representable float in [0, 1).
        u = (stable_hash(f"{self._seed}:{e.envelope_id}") >> 11) / float(1 << 53)
        return DetectorResult(
            fused_score=u,
            per_detector=(),
            reasons=(),
            latency_us=(time.perf_counter() - t0) * 1e6,
        )

    def update(self, e: ActionEnvelope, outcome: Outcome) -> None:
        return None

    def state_bytes(self) -> int:
        return 0
