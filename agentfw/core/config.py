"""Engine configuration: every tuned constant of the Tier 1 engine lives here."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

__all__ = ["Direction", "EngineConfig", "PriorsMode"]

PriorsMode = Literal["all", "money_count_only", "off"]
Direction = Literal["up", "both"]


@dataclasses.dataclass(frozen=True, slots=True)
class EngineConfig:
    # Normalization / session (§6.1)
    k_prior_tools: int = 4
    max_session_ids: int = 2000
    freetext_min_len: int = 40
    # Type inference (§6.2)
    type_min_obs: int = 10
    type_sticky_window: int = 100
    reservoir_size: int = 64
    # Magnitude (§6.3)
    mag_probs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    ewma_alpha: float = 0.02
    mag_cold_n: int = 30
    z0: float = 3.0
    tau: float = 0.75
    mag_eps: float = 0.05
    # Sweep switches (Task 1.14, Open Decisions 2 and 5). ``mag_direction_default``: direction of
    # numeric fields with no money/count hint and no up-keyword in their name.
    # ``mag_resp_signal``: score/learn the previous response's ``item_count`` (§2.3).
    mag_direction_default: Direction = "both"
    mag_resp_signal: bool = True
    # Pooled priors (§6.10)
    prior_iqr_inflate: float = 1.5
    prior_novelty_discount: float = 0.67
    priors_mode: PriorsMode = "all"
    # Velocity (§6.4)
    vel_short_min: int = 10
    vel_long_min: int = 1440
    vel_eps: float = 0.25
    vel_cold_n: int = 30
    vel_sum_signal: bool = True  # money-sum windows next to the call count (Task 1.14 sweep)
    # Novelty (§6.5)
    novelty_lru: int = 10_000
    novelty_full_n: int = 200
    cms_width: int = 2048
    cms_depth: int = 4
    # Sequence (§6.6)
    trans_min_ctx: int = 20
    trans_smoothing: float = 0.5
    prereq_min_n: int = 50
    prereq_rate: float = 0.90  # docs/decisions/003-prereq-threshold.md (spec: 0.95)
    # Id flow (§6.7-6.8)
    flow_min_obs: int = 20
    # Fusion (§6.11)
    fusion_bonus: float = 0.15
    fusion_bonus_gate: float = 0.5
    reason_min: float = 0.3
    # Calibration
    cal_alpha: float = 0.005
    cal_theta0: float = 0.6
    cal_blend_n: int = 500
    cal_delta: float = 0.1

    def replace(self, **kw: Any) -> EngineConfig:
        """Return a copy with the given fields changed (``TypeError`` on unknown names)."""
        return dataclasses.replace(self, **kw)
