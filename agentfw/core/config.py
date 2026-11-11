"""Engine configuration: every tuned constant of the Tier 1 engine lives here."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

__all__ = [
    "CAL_CENSOR_MODES",
    "CalCensor",
    "Direction",
    "EngineConfig",
    "PriorsMode",
    "ScoreModel",
    "VelCountStat",
]

PriorsMode = Literal["all", "money_count_only", "off"]
Direction = Literal["up", "both"]
CalCensor = Literal["none", "drop", "winsorize"]
VelCountStat = Literal["logz", "poisson"]
ScoreModel = Literal["z", "tail"]
CAL_CENSOR_MODES: tuple[CalCensor, ...] = ("none", "drop", "winsorize")


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
    # Amendment A1.2: how a value's distance from the median becomes a score. "z" (spec §6.3): the
    # IQR-scaled robust z through ``squash(z, z0, tau)``; "tail": a two-piece robust scale on the
    # side the value falls on, the normal upper-tail probability p of its z, and the posterior
    # ``tail_prior / (tail_prior + (1 - tail_prior) * p)``. No state differs between the two.
    mag_score: ScoreModel = "z"
    # Prior anomaly rate of the tail score. Equals ``cal_alpha`` by definition (D-023); a field of
    # its own so that a ``cal_alpha`` sweep moves thresholds and not scores. Fixed by the spec:
    # never swept, never tuned against benchmark recall.
    tail_prior: float = 0.005
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
    # Task 1.15 switch (decision 008): refused (blocked / denied) calls still count as attempts in
    # the short window's scored count; they never enter the history baseline. Default on by user
    # decision (2026-11-10, decision 008 "User decision"); False = spec.
    vel_count_attempts: bool = True
    # Amendment A1.1: how the count signal turns the short-window count into z. "logz" (spec
    # §6.4): robust z of ln(1 + count); "poisson": the normal deviate of the Poisson upper tail
    # at the key's median window count, over its over-dispersion. The money sum is always "logz".
    vel_count_stat: VelCountStat = "logz"
    # Novelty (§6.5)
    novelty_lru: int = 10_000
    # Novelty ramp min(1, n / novelty_full_n): 50 by user decision (2026-11-10, decision 008);
    # spec §6.5: 200.
    novelty_full_n: int = 50
    cms_width: int = 2048
    cms_depth: int = 4
    # Sequence (§6.6)
    trans_min_ctx: int = 20
    trans_smoothing: float = 0.5
    prereq_min_n: int = 50
    prereq_rate: float = 0.90  # docs/decisions/003-prereq-threshold.md (spec: 0.95)
    # Task 1.15 switch (docs/decisions/008-tier1-gate.md): ``s_prereq`` = the mined rate of the
    # missing prerequisite instead of 1.0. Default on by user decision (2026-11-10, decision 008);
    # False = spec §6.6(b).
    seq_prereq_graded: bool = True
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
    # Task 1.15 switch (docs/decisions/008-tier1-gate.md): what the calibrator learns from a call
    # the engine itself blocked (fused >= the threshold that scored it). "none" (spec §6.8): the
    # fused score of every allowed call; "drop": nothing; "winsorize": the threshold instead of the
    # fused score, also when the call was BLOCKED with fused >= threshold (tail mass kept, value
    # not).
    cal_censor: CalCensor = "none"
    # ... applied only once the agent's calibrator holds this many scores (shadow, then censor).
    cal_censor_min_n: int = 0

    def replace(self, **kw: Any) -> EngineConfig:
        """Return a copy with the given fields changed (``TypeError`` on unknown names)."""
        return dataclasses.replace(self, **kw)
