"""The post-fusion decision pipeline (feature spec F4-F8): ``decide`` is what the engine calls.

Stages, in order (later tasks insert theirs at the marked points; the final order is set by
Task 3.4: bands on θ_eff -> gate -> escalation -> provisional -> ceiling -> session overlay):

1. **Bands** (spec §6.8) on the effective threshold ``TrajectoryInfo.threshold_eff`` (the agent's
   θ, or θ_eff under ``br_mode="modulate"``): ``fused >= θ_eff`` -> ``BLOCK``;
   ``fused >= θ_eff - cal_delta`` -> ``REQUIRE_APPROVAL``; else ``ALLOW``. Exactly
   ``Calibrator.decide``, so with every switch off ``decide`` returns the calibrator's decision.
2. **Gate** (F4, ``gate``; ``core.trajectory.gate``, T-41). An irreversible call runs ``gate``
   (it needs the call: ``e``, its ``types`` and the sequence detector ``seq``) and the outcome is
   reported as ``TrajectoryInfo.gate``: ``block`` -> ``BLOCK`` plus flag ``PREREQ_MISSING``;
   ``escalate`` -> ``REQUIRE_APPROVAL`` (held for the escalation stage, Task 3.2, so a
   distribution anomaly alone never blocks an irreversible call from the fast path); ``pass``
   keeps the band. A reversible call is never blocked by the engine: with ``provisional`` off its
   ``BLOCK`` is capped at ``REQUIRE_APPROVAL``; with ``provisional`` on the provisional stage
   below makes it ``ALLOW`` + ``PROVISIONAL`` instead (the same call is never both).
3. **Provisional** (F6, ``provisional``): a reversible call (``not TrajectoryInfo.irreversible``)
   whose band is ``BLOCK`` becomes ``ALLOW`` with the flag ``PROVISIONAL``; the engine opens its
   ``ProvEntry`` in ``update`` (``core.trajectory.provisional``). ``REQUIRE_APPROVAL`` is left as
   it is.

Pure: reads its arguments (and, through the gate, the store) only. The returned
``TrajectoryInfo`` carries the flags this pipeline raised; the engine adds the derived and
lifecycle flags after it.
"""

from __future__ import annotations

import dataclasses

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import SequenceDetector
from agentfw.core.envelope import ActionEnvelope, Decision, TrajectoryInfo
from agentfw.core.schema_infer import TypeView
from agentfw.core.trajectory.flags import Flag
from agentfw.core.trajectory.gate import gate

__all__ = ["band", "decide"]


def band(fused: float, theta_eff: float, cal_delta: float) -> Decision:
    """The §6.8 decision band of ``fused`` against ``theta_eff`` (NaN -> ``ALLOW``)."""
    if fused >= theta_eff:
        return Decision.BLOCK
    if fused >= theta_eff - cal_delta:
        return Decision.REQUIRE_APPROVAL
    return Decision.ALLOW


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


def decide(
    fused: float,
    traj: TrajectoryInfo,
    cfg: EngineConfig,
    *,
    e: ActionEnvelope | None = None,
    types: TypeView | None = None,
    seq: SequenceDetector | None = None,
) -> tuple[Decision, TrajectoryInfo]:
    """``(decision, trajectory)`` of one call after fusion (module docstring). ``e``, ``types``
    and ``seq`` are needed only by the gate (``gate`` on and ``traj.irreversible``)."""
    decision = band(fused, traj.threshold_eff, cfg.cal_delta)
    if cfg.gate:
        decision, traj = _gate_stage(fused, decision, traj, cfg, e, types, seq)
    # -- escalation (Task 3.2) --
    if cfg.provisional and decision is Decision.BLOCK and not traj.irreversible:
        decision = Decision.ALLOW
        traj = dataclasses.replace(traj, flags=traj.flags | {Flag.PROVISIONAL})
    # -- ceiling (Task 3.4), session overlay (Task 3.3) --
    return decision, traj
