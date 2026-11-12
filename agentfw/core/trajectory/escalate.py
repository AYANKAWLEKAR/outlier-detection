"""Two-tier escalation before any intervention (feature spec F5; T-12, T-42).

The fast path scores every call once. Only when it raises a would-be intervention on an
irreversible call does the engine pay for a second, stricter look before the verdict is final,
so no block rests on a stale or approximate score and the cost lands on the rare calls about to
be stopped.

``decide`` runs ``escalate`` (with ``EngineConfig.escalate`` on) after the gate, on an
irreversible call whose decision so far is ``BLOCK`` or ``REQUIRE_APPROVAL`` -- a gate
``escalate`` (held at ``REQUIRE_APPROVAL``, T-41) or a band candidate -- unless the gate itself
blocked it (a missing prerequisite is final, not a candidate). Reversible calls never escalate.

1. **Fresh re-score.** ``rescore()`` recomputes the fused score from the store without the
   engine's per-call cache (T-12: the Python fast path skips nothing, so this is the fast path's
   computation run again on current state). It reads only, so ``score`` stays pure.
2. **Confirm.** The escalated decision is the more severe of the fast decision and the band of
   the fresh score against ``theta_eff`` (``ALLOW < REQUIRE_APPROVAL < BLOCK``): a fresh score at
   or above ``theta_eff`` confirms a block (a gate escalation becomes ``BLOCK`` again), and
   nothing here relaxes the fast decision. A confident fast-path ``BLOCK`` stays ``BLOCK``.
3. **Judge** (optional, injected): consulted only when the escalated decision is
   ``REQUIRE_APPROVAL``, with the envelope and the structured reasons (never response content:
   only ``ResponseMeta`` crosses into this package). A score ``>= judge_block`` -> ``BLOCK``,
   ``<= judge_allow`` -> ``ALLOW``, else (abstain ``None``, NaN, in between) unchanged. It is a
   tie-breaker inside the approval band only, so a judge can never turn a block into an allow.

The engine writes the result with ``TrajectoryInfo.escalated = True`` and both scores
(``fast_score``, ``escalated_score``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from agentfw.core.calibration import band
from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope, Decision, Reason

__all__ = ["SEVERITY", "Judge", "escalate"]

#: Decisions from least to most severe: escalation never moves down this order before the judge.
SEVERITY: dict[Decision, int] = {
    Decision.ALLOW: 0,
    Decision.REQUIRE_APPROVAL: 1,
    Decision.BLOCK: 2,
}


class Judge(Protocol):
    """An optional tie-breaker for the approval band: a 0-1 suspicion score of call ``e`` from its
    structured reasons, or ``None`` to abstain (also on timeout). The implementation owns its
    session history and its 1.5 s budget."""

    def __call__(self, e: ActionEnvelope, reasons: Sequence[Reason]) -> float | None: ...


def escalate(
    e: ActionEnvelope,
    fast_fused: float,
    fast_decision: Decision,
    rescore: Callable[[], float],
    theta_eff: float,
    cfg: EngineConfig,
    judge: Judge | None,
    *,
    reasons: Sequence[Reason] = (),
) -> tuple[Decision, float]:
    """``(escalated decision, escalated score)`` of call ``e`` (module docstring). ``fast_fused``
    is the score behind ``fast_decision``, for the audit only: the caller reports it next to the
    fresh score, and the decision rests on the fresh one."""
    fresh = rescore()
    decision = max(fast_decision, band(fresh, theta_eff, cfg.cal_delta), key=SEVERITY.__getitem__)
    if decision is Decision.REQUIRE_APPROVAL and judge is not None:
        s = judge(e, reasons)
        if s is not None:
            if s >= cfg.judge_block:
                decision = Decision.BLOCK
            elif s <= cfg.judge_allow:
                decision = Decision.ALLOW
    return decision, fresh
