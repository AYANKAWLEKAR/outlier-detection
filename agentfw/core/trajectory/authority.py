"""Verdict authority scaled by evidence maturity (feature spec F8; T-14, T-44 - T-47, T-50).

The kind of decision the engine may take is tied to how much it knows: a block resting on a key
with a dozen observations, or on a pooled prior, must not carry the authority of one resting on
thousands.

* ``grade`` / ``maturity_of`` grade one call by the evidence its decision rests on: the minimum
  evidence count (``evidence_n``) over the *contributors*, the non-vital scores that made the
  fused score under fusion v1 (top1 + ``fusion_bonus`` when top2 > ``fusion_bonus_gate``): the
  top score(s) when they reach the contributor floor, plus the second score(s) when they earned
  the bonus. Vital scores never fuse (T-07), so they never contribute. The floor
  (``contributor_floor``) is ``CONTRIB_MIN`` (F8's 0.3), lowered to ``θ_eff - cal_delta`` -- the
  lowest fused score at which a band can intervene -- when that is lower (T-50): a decision a
  band took on a score under 0.3 rests on that score, so it is graded on it. Cut points (T-14):
  ``cold`` n < ``mat_warm_n``, ``warm`` n < ``mat_mature_n``, else ``mature``. A contributor
  without a count is ``cold``; a contributor on a pooled prior (``prior_used``) is at most
  ``warm``; the ``bare`` regime forces ``bare``; no contributor (top score under the floor, or
  0) is ``mature``: no band intervened, nothing to cap.
* ``apply_ceiling`` caps a decision (T-14): mature -> ``BLOCK``; warm -> ``REQUIRE_APPROVAL``;
  cold -> ``ALLOW`` with notice ``warn``; bare -> ``ALLOW`` with notice ``inform``. It never
  raises a decision. F4's prerequisite block (``GateOutcome.action == "block"``) is exempt when
  the regime has schemas (``ceiling_exempt``): a missing producer for a destructive tool is a
  fact about the tool's contract, not a statistic.

Pure functions of their arguments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.vital import NAME as VITAL
from agentfw.core.envelope import Decision, DetectorScore, Maturity, Notice, Regime
from agentfw.core.trajectory.escalate import SEVERITY
from agentfw.core.trajectory.gate import SCHEMA_REGIMES, GateOutcome

__all__ = [
    "CEILING",
    "CONTRIB_MIN",
    "Grade",
    "Maturity",
    "apply_ceiling",
    "ceiling_exempt",
    "contributor_floor",
    "contributors",
    "evidence_n",
    "grade",
    "maturity_of",
]

#: A score contributes to a decision from this value on (F8: "contributed >= 0.3").
CONTRIB_MIN = 0.3
#: The most severe decision each maturity may take (T-14).
CEILING: dict[Maturity, Decision] = {
    "mature": Decision.BLOCK,
    "warm": Decision.REQUIRE_APPROVAL,
    "cold": Decision.ALLOW,
    "bare": Decision.ALLOW,
}
_NOTICE: dict[Maturity, Notice | None] = {
    "mature": None,
    "warm": None,
    "cold": "warn",
    "bare": "inform",
}
_RANK: dict[Maturity, int] = {"bare": 0, "cold": 1, "warm": 2, "mature": 3}
#: The sequence part that fired -> the evidence key of that part's own history depth (T-45).
_SEQ_N = {"trans": "n_trans", "prereq": "n_prereq", "flow": "n"}
_N_KEYS = ("n", "n_total", "n_hist")


@dataclass(frozen=True, slots=True)
class Grade:
    """A call's maturity and the evidence count that set it (the minimum-level contributor's
    count; ``None`` when nothing contributed, the regime is bare or that contributor has none)."""

    maturity: Maturity
    n: int | None


def _count(v: object) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def evidence_n(s: DetectorScore) -> int | None:
    """The evidence count of one score: the first of evidence ``n`` (magnitude), ``n_total``
    (novelty), ``n_hist`` (velocity); for the sequence detector, the count of the part that
    fired (``n_trans``, ``n_prereq``, or the flow part's ``n``). ``None`` when absent."""
    ev = s.evidence
    if s.detector == "sequence":
        key = _SEQ_N.get(str(ev.get("fired")))
        if key is not None:
            return _count(ev.get(key))
    for key in _N_KEYS:
        if key in ev:
            return _count(ev[key])
    return None


def contributor_floor(theta_eff: float | None, cfg: EngineConfig) -> float:
    """``min(CONTRIB_MIN, θ_eff - cal_delta)`` (T-50); ``CONTRIB_MIN`` without a threshold."""
    if theta_eff is None or math.isnan(theta_eff):
        return CONTRIB_MIN
    return min(CONTRIB_MIN, theta_eff - cfg.cal_delta)


def contributors(
    per: Sequence[DetectorScore], cfg: EngineConfig, floor: float = CONTRIB_MIN
) -> list[DetectorScore]:
    """The non-vital scores that made the fused score (module docstring); ties count alike. A
    top score of 0 contributes nothing whatever the floor."""
    v1 = [(0.0 if math.isnan(s.score) else s.score, s) for s in per if s.detector != VITAL]
    ranked = sorted({v for v, _ in v1}, reverse=True)
    top1 = ranked[0] if ranked else 0.0
    if top1 <= 0.0 or top1 < floor:
        return []
    out = [s for v, s in v1 if v == top1]  # a tie at the top is top1 and top2 at once
    if len(out) == 1 and len(ranked) > 1 and ranked[1] > cfg.fusion_bonus_gate:
        out += [s for v, s in v1 if v == ranked[1]]  # top2 earned the bonus
    return out


def _level(s: DetectorScore, cfg: EngineConfig) -> Maturity:
    n = evidence_n(s)
    if n is None or n < cfg.mat_warm_n:
        return "cold"
    level: Maturity = "warm" if n < cfg.mat_mature_n else "mature"
    if level == "mature" and s.evidence.get("prior_used") is True:
        return "warm"
    return level


def grade(
    per: Sequence[DetectorScore],
    regime: Regime,
    cfg: EngineConfig,
    theta_eff: float | None = None,
) -> Grade:
    """The call's ``Grade`` (module docstring); ``theta_eff`` sets the contributor floor."""
    if regime == "bare":
        return Grade("bare", None)
    cs = contributors(per, cfg, contributor_floor(theta_eff, cfg))
    if not cs:
        return Grade("mature", None)

    # the lowest level, then the smallest count within it (a missing count first)
    def key(s: DetectorScore) -> tuple[int, int]:
        n = evidence_n(s)
        return _RANK[_level(s, cfg)], -1 if n is None else n

    worst = min(cs, key=key)
    return Grade(_level(worst, cfg), evidence_n(worst))


def maturity_of(
    per: Sequence[DetectorScore],
    fused: float,
    regime: Regime,
    cfg: EngineConfig,
    *,
    theta_eff: float | None = None,
) -> Maturity:
    """The maturity of a call with scores ``per`` (module docstring). ``fused`` is kept for the
    brief's signature: the contributors are read from ``per``, whose top score is the fused
    score below the bonus gate."""
    del fused
    return grade(per, regime, cfg, theta_eff).maturity


def ceiling_exempt(gate: GateOutcome | None, regime: Regime) -> bool:
    """F4's prerequisite block in a schema regime is outside the ceiling (F8)."""
    return gate is not None and gate.action == "block" and regime in SCHEMA_REGIMES


def apply_ceiling(
    decision: Decision, maturity: Maturity, gate: GateOutcome | None, regime: Regime
) -> tuple[Decision, Decision | None, Notice | None]:
    """``(decision, capped_from, notice)`` after the authority ceiling (module docstring):
    ``capped_from`` is the decision before the cap, ``None`` (and no notice) when it stood."""
    if ceiling_exempt(gate, regime):
        return decision, None, None
    ceiling = CEILING[maturity]
    if SEVERITY[decision] <= SEVERITY[ceiling]:
        return decision, None, None
    return ceiling, decision, _NOTICE[maturity]
