"""The post-fusion decision pipeline (feature spec F4-F8): ``decide`` is what the engine calls.

Stages, in order -- final since Task 3.4: **bands on θ_eff -> gate (irreversible) -> escalation
-> provisional (reversible) -> ceiling -> session overlay**:

1. **Bands** (spec §6.8, ``core.calibration.band``) on the effective threshold
   ``TrajectoryInfo.threshold_eff`` (the agent's θ, or θ_eff under ``br_mode="modulate"``):
   ``fused >= θ_eff`` -> ``BLOCK``; ``fused >= θ_eff - cal_delta`` -> ``REQUIRE_APPROVAL``; else
   ``ALLOW``. Exactly ``Calibrator.decide``, so with every switch off ``decide`` returns the
   calibrator's decision.
2. **Gate** (F4, ``gate``; ``core.trajectory.gate``, T-41). An irreversible call runs ``gate``
   (it needs the call: ``e``, its ``types`` and the sequence detector ``seq``) and the outcome is
   reported as ``TrajectoryInfo.gate``: ``block`` -> ``BLOCK`` plus flag ``PREREQ_MISSING``;
   ``escalate`` -> ``REQUIRE_APPROVAL`` (held for the escalation stage, so a distribution
   anomaly alone never blocks an irreversible call from the fast path); ``pass`` keeps the band.
   A reversible call is never blocked by the engine: with ``provisional`` off its ``BLOCK`` is
   capped at ``REQUIRE_APPROVAL``; with ``provisional`` on the provisional stage below makes it
   ``ALLOW`` + ``PROVISIONAL`` instead (the same call is never both).
3. **Escalation** (F5, ``escalate``; ``core.trajectory.escalate``, T-42). An irreversible call
   whose decision so far is ``BLOCK`` or ``REQUIRE_APPROVAL`` -- unless the gate blocked it, a
   final verdict -- is re-scored fresh (``rescore``, needed with ``e`` here) and the escalated
   decision replaces it; the optional ``judge`` breaks ties in the approval band. Reported as
   ``TrajectoryInfo.escalated`` with ``fast_score`` and ``escalated_score``. Off: the gate's
   ``REQUIRE_APPROVAL`` stand-in stays (T-41).
4. **Provisional** (F6, ``provisional``): a reversible call (``not TrajectoryInfo.irreversible``)
   whose band is ``BLOCK`` becomes ``ALLOW`` with the flag ``PROVISIONAL``; the engine opens its
   ``ProvEntry`` in ``update`` (``core.trajectory.provisional``). ``REQUIRE_APPROVAL`` is left as
   it is. Escalation touches irreversible calls only, so the two never meet.

5. **Ceiling** (F8, ``authority``; ``core.trajectory.authority``, T-14, T-44 - T-50). The call's
   evidence maturity (``TrajectoryInfo.maturity``, which the engine grades on every call from
   the scores its decision rests on) caps what is left: mature -> ``BLOCK``; warm ->
   ``REQUIRE_APPROVAL``; cold -> ``ALLOW`` with ``notice="warn"``; bare -> ``ALLOW`` with
   ``notice="inform"``. The decision before a cap is ``TrajectoryInfo.capped_from``; a gate
   ``block`` is exempt when the regime has schemas; a cold call also gets flag ``COLD_START``
   (weight 0, never counted). It runs after the provisional stage (a provisional call is
   ``ALLOW`` already) and before the overlay, which it does not cap (T-47, pending a user
   decision: the literal F8 audit counts overlay blocks). The engine derives the call's flags
   from the decision *before* the cap (``capped_from``), so given the same outcomes the session
   window, densities and verdicts are the same with ``authority`` on and off (T-46). Off:
   maturity is still reported, nothing is capped.
6. **Session overlay** (F7, ``session_enforce``; ``session_overlay``, T-43): a call whose session
   was already halted *before* it (``SessionRecord.halted``) is ``BLOCK``ed, and is no longer
   ``PROVISIONAL`` (a blocked call opens no entry) nor carries the ceiling's notice; the engine
   adds ``HALT_REASON``. It is the
   last stage and the engine applies it to what ``decide`` returns, once it has derived the
   call's flags and density from that decision: the flag window counts the call's own evidence,
   never the session block (no ``BLOCK`` flag from the overlay), so given the same outcomes the
   densities, verdicts and halt metrics recomputed from them are the same with
   ``session_enforce`` on and off. Off (or not halted): unchanged; the verdict is then
   reported only (shadow).

Pure: reads its arguments (and, through the gate and the re-score, the store) only. The
returned ``TrajectoryInfo`` carries the flags this pipeline raised; the engine adds the derived
and lifecycle flags after it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from agentfw.core.calibration import band
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import SequenceDetector
from agentfw.core.envelope import ActionEnvelope, Decision, Reason, TrajectoryInfo
from agentfw.core.schema_infer import TypeView
from agentfw.core.trajectory.authority import CEILING, apply_ceiling, ceiling_exempt
from agentfw.core.trajectory.escalate import SEVERITY, Judge, escalate
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.gate import gate

__all__ = ["HALT_REASON", "band", "decide", "session_overlay"]

#: The reason of a call blocked by the session overlay (stage 6): no field, no raw id.
HALT_REASON = Reason(
    detector="session",
    field=None,
    observed="halt_session",
    baseline_summary=None,
    percentile=None,
    blast_radius=None,
    prereq_missing=None,
)

#: Decisions that make an irreversible call a candidate for escalation (F5).
_CANDIDATES = frozenset({Decision.BLOCK, Decision.REQUIRE_APPROVAL})


def _gate_stage(
    fused: float,
    decision: Decision,
    traj: TrajectoryInfo,
    cfg: EngineConfig,
    e: ActionEnvelope | None,
    types: TypeView | None,
    seq: SequenceDetector | None,
) -> tuple[Decision, TrajectoryInfo]:
    if not traj.irreversible:
        if decision is Decision.BLOCK and not cfg.provisional:
            decision = Decision.REQUIRE_APPROVAL
        return decision, traj
    if e is None or types is None or seq is None:
        raise ValueError("gate: an irreversible call needs e, types and seq")
    out = gate(e, types, seq, fused, traj.threshold_eff, cfg, regime=traj.regime)
    flags = traj.flags
    if out.action == "block":
        decision = Decision.BLOCK
        flags = flags | {Flag.PREREQ_MISSING}
    elif out.action == "escalate":
        decision = Decision.REQUIRE_APPROVAL
    return decision, dataclasses.replace(traj, flags=flags, gate=out)


def _escalation_stage(
    fused: float,
    decision: Decision,
    traj: TrajectoryInfo,
    cfg: EngineConfig,
    e: ActionEnvelope | None,
    rescore: Callable[[], float] | None,
    judge: Judge | None,
    reasons: Sequence[Reason],
) -> tuple[Decision, TrajectoryInfo]:
    gate_blocked = traj.gate is not None and traj.gate.action == "block"  # final, not a candidate
    if not traj.irreversible or decision not in _CANDIDATES or gate_blocked:
        return decision, traj
    if e is None or rescore is None:
        raise ValueError("escalation: an irreversible candidate needs e and rescore")
    theta_eff = traj.threshold_eff
    decision, fresh = escalate(e, fused, decision, rescore, theta_eff, cfg, judge, reasons=reasons)
    out = dataclasses.replace(traj, escalated=True, fast_score=fused, escalated_score=fresh)
    return decision, out


def _ceiling_stage(decision: Decision, traj: TrajectoryInfo) -> tuple[Decision, TrajectoryInfo]:
    maturity = traj.maturity
    if maturity is None:
        raise ValueError("ceiling: authority needs TrajectoryInfo.maturity")
    exempt = ceiling_exempt(traj.gate, traj.regime)
    held = exempt and SEVERITY[decision] > SEVERITY[CEILING[maturity]]  # a cap was in play
    decision, capped_from, notice = apply_ceiling(decision, maturity, traj.gate, traj.regime)
    flags = (traj.flags | {Flag.COLD_START}) if maturity == "cold" else traj.flags
    out = dataclasses.replace(
        traj, flags=flags, capped_from=capped_from, notice=notice, ceiling_exempt=held
    )
    return decision, out


def decide(
    fused: float,
    traj: TrajectoryInfo,
    cfg: EngineConfig,
    *,
    e: ActionEnvelope | None = None,
    types: TypeView | None = None,
    seq: SequenceDetector | None = None,
    rescore: Callable[[], float] | None = None,
    judge: Judge | None = None,
    reasons: Sequence[Reason] = (),
) -> tuple[Decision, TrajectoryInfo]:
    """``(decision, trajectory)`` of one call after fusion (module docstring). ``e``, ``types``
    and ``seq`` are needed only by the gate (``gate`` on and ``traj.irreversible``); ``e`` and
    ``rescore`` only by the escalation stage (``escalate`` on and an irreversible candidate),
    which shows ``judge`` the call and its ``reasons``; ``traj.maturity`` only by the ceiling
    (``authority`` on)."""
    decision = band(fused, traj.threshold_eff, cfg.cal_delta)
    if cfg.gate:
        decision, traj = _gate_stage(fused, decision, traj, cfg, e, types, seq)
    if cfg.escalate:
        decision, traj = _escalation_stage(fused, decision, traj, cfg, e, rescore, judge, reasons)
    if cfg.provisional and decision is Decision.BLOCK and not traj.irreversible:
        decision = Decision.ALLOW
        traj = dataclasses.replace(traj, flags=traj.flags | {Flag.PROVISIONAL})
    if cfg.authority:
        decision, traj = _ceiling_stage(decision, traj)
    # the session overlay follows in the engine (session_overlay), once flags are derived
    return decision, traj


def session_overlay(
    decision: Decision, traj: TrajectoryInfo, halted: bool, cfg: EngineConfig
) -> tuple[Decision, TrajectoryInfo]:
    """Stage 6 (module docstring): ``BLOCK`` without ``PROVISIONAL`` and without the ceiling's
    notice (a blocked call is not allowed with one; ``capped_from`` stays for the audit) when
    ``session_enforce`` is on and the session was ``halted`` before this call; otherwise
    unchanged."""
    if not (cfg.session_enforce and halted):
        return decision, traj
    flags = traj.flags - {Flag.PROVISIONAL}
    return Decision.BLOCK, dataclasses.replace(traj, flags=flags, notice=None)
