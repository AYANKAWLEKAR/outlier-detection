"""Per-agent threshold calibration (spec §6.8).

State per agent is a P-squared tail estimator over the fused scores of allowed (or approved)
actions plus ``n_scored``. ``threshold`` is pure: it never creates or mutates a record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import Decision, Outcome
from agentfw.core.state import StateStore, key_cal, register_record
from agentfw.core.stats.p2 import P2Quantiles, p2_tail

__all__ = ["CalRecord", "Calibrator"]


@register_record
@dataclass(slots=True)
class CalRecord:
    KIND: ClassVar[str] = "cal"
    q: P2Quantiles
    n_scored: int

    @classmethod
    def new(cls, cfg: EngineConfig) -> Self:
        return cls(p2_tail(1.0 - cfg.cal_alpha), 0)

    def to_dict(self) -> dict[str, object]:
        return {"q": self.q.to_dict(), "n_scored": self.n_scored}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(P2Quantiles.from_dict(d["q"]), int(d["n_scored"]))


class Calibrator:
    def __init__(self, store: StateStore, cfg: EngineConfig) -> None:
        self.store = store
        self.cfg = cfg

    def _rec(self, agent: str) -> CalRecord | None:
        rec = self.store.get(key_cal(agent))
        return rec if isinstance(rec, CalRecord) else None

    def threshold(self, agent: str) -> float:
        cfg = self.cfg
        rec = self._rec(agent)
        if rec is None or rec.n_scored <= 0:
            return cfg.cal_theta0
        q = rec.q.quantile(1.0 - cfg.cal_alpha)
        if math.isnan(q):
            return cfg.cal_theta0
        w = min(1.0, rec.n_scored / cfg.cal_blend_n)
        return w * q + (1.0 - w) * cfg.cal_theta0

    def decide(self, agent: str, fused: float) -> Decision:
        theta = self.threshold(agent)
        if fused >= theta:
            return Decision.BLOCK
        if fused >= theta - self.cfg.cal_delta:
            return Decision.REQUIRE_APPROVAL
        return Decision.ALLOW

    def observe(self, agent: str, fused: float, outcome: Outcome) -> None:
        if outcome not in (Outcome.ALLOWED, Outcome.APPROVED) or not math.isfinite(fused):
            return
        rec = self._rec(agent)
        if rec is None:
            rec = CalRecord.new(self.cfg)
            self.store.put(key_cal(agent), rec)
        rec.q.add(fused)
        rec.n_scored += 1
