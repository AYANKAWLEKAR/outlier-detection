"""Engine configuration: every tuned constant of the Tier 1 engine lives here."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

__all__ = [
    "BR_MODES",
    "BrMode",
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
BrMode = Literal["off", "score", "modulate"]
CAL_CENSOR_MODES: tuple[CalCensor, ...] = ("none", "drop", "winsorize")
BR_MODES: tuple[BrMode, ...] = ("off", "score", "modulate")


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
    # Amendment A1.2: how a value's distance from the median becomes a score. "tail" (default):
    # a two-piece robust scale on the side the value falls on, the normal upper-tail probability
    # p of its z, and the posterior ``tail_prior / (tail_prior + (1 - tail_prior) * p)``; "z"
    # (spec §6.3, pre-amendment): the IQR-scaled robust z through ``squash(z, z0, tau)``. No
    # state differs between the two.
    mag_score: ScoreModel = "tail"
    # Prior anomaly rate of the tail score. Equals ``cal_alpha`` by definition (D-023); a field of
    # its own so that a ``cal_alpha`` sweep moves thresholds and not scores. Fixed by the spec:
    # never swept, never tuned against benchmark recall.
    tail_prior: float = 0.005
    # Amendment A1.3: in a BOOL-typed field (a history of only 0 and 1) magnitude also learns the
    # values that parse as finite numbers (never Python bools or "true"/"false") and scores the
    # ones that contradict BOOL (a number outside {0, 1}); novelty then scores such a value only
    # while magnitude has no usable reference for the field. Default True; False: the
    # pre-amendment behaviour (magnitude skips BOOL-typed fields, novelty scores the
    # contradiction as unseen).
    mag_bool_numeric: bool = True
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
    # Amendment A1.1: how the count signal turns the short-window count into z. "poisson"
    # (default): the normal deviate of the Poisson upper tail at the key's median window count,
    # over its over-dispersion; "logz" (spec §6.4, pre-amendment): robust z of ln(1 + count).
    # The money sum is always "logz".
    vel_count_stat: VelCountStat = "poisson"
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
    # Task 1.4 switch (T-04): "present" = a ledger ref on the envelope or a session_ids_seen hit,
    # so ids seeded from the task input count as flowing. False = v3 (session_ids_seen only).
    flow_ledger: bool = False
    # Vital signs (F2, Task 1.5; opt-in through ``Engine(enabled=...)``, never fused: T-07).
    # Loop: exact ``call_fp`` matches in the last ``vital_w_loop`` calls (``vital_w_loop_polling``
    # for read-only + idempotent tools), retries of an errored call exempt up to ``vital_r_max``;
    # an arg-token Jaccard >= ``vital_jaccard`` is a near-duplicate. Spin: ``vital_w_spin`` calls
    # with ``delta_i < vital_spin_di``. Stall: ``vital_w_stall`` calls without a new artifact, or
    # reads with ``delta_i < vital_stall_di``.
    vital_w_loop: int = 8
    vital_w_loop_polling: int = 32
    vital_r_max: int = 3
    vital_jaccard: float = 0.9
    vital_w_spin: int = 4
    vital_spin_di: float = 0.1
    vital_w_stall: int = 6
    vital_stall_di: float = 0.3
    # Blast radius (§6.9 lexical + structural, F3 annotation terms; Task 1.7, core.blast).
    # "off": no b (``TrajectoryInfo.blast_radius`` None); "score": b on every reason and on
    # ``TrajectoryInfo``; "modulate": also decide against ``θ_eff = θ · (1 − br_k · b)``.
    # ``irreversible_b``: a write at or above this b is irreversible (F4's gate input).
    br_mode: BrMode = "off"
    br_k: float = 0.6
    irreversible_b: float = 0.6
    # Session flag window (F7 substrate, Task 2.1; core.trajectory). Every call derives flags
    # (``derive_flags``: a score counts from ``flag_min``); with ``sessions`` (T-34: it opens the
    # engine's one session gate, which later session features also open; shut = no
    # ``sess:`` records, ``TrajectoryInfo.density`` None) ``update`` records them, for every
    # outcome, in a per-session window of the last ``session_window`` calls, and ``density`` =
    # Σ of each call's highest flag weight / ``session_window`` (T-09). Every
    # ``sess_sweep_every`` records, non-halted sessions idle > ``sess_idle_ms`` are evicted
    # (T-15). ``prov_unresolved_weight``: the weight of an ``UNRESOLVED`` flag (Task 2.2).
    sessions: bool = False
    session_window: int = 6
    flag_min: float = 0.5
    sess_idle_ms: int = 1_800_000
    sess_sweep_every: int = 4096
    prov_unresolved_weight: float = 0.125
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
