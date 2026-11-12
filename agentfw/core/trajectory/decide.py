"""The post-fusion decision pipeline (feature spec F4-F8): ``decide`` is what the engine calls.

Stages, in order (later tasks insert theirs at the marked points; the final order is set by
Task 3.4: bands on θ_eff -> gate -> escalation -> provisional -> ceiling -> session overlay):

1. **Bands** (spec §6.8) on the effective threshold ``TrajectoryInfo.threshold_eff`` (the agent's
   θ, or θ_eff under ``br_mode="modulate"``): ``fused >= θ_eff`` -> ``BLOCK``;
   ``fused >= θ_eff - cal_delta`` -> ``REQUIRE_APPROVAL``; else ``ALLOW``. Exactly
   ``Calibrator.decide``, so with every switch off ``decide`` returns the calibrator's decision.
2. **Provisional** (F6, ``provisional``): a reversible call (``not TrajectoryInfo.irreversible``)
   whose band is ``BLOCK`` becomes ``ALLOW`` with the flag ``PROVISIONAL``; the engine opens its
   ``ProvEntry`` in ``update`` (``core.trajectory.provisional``). ``REQUIRE_APPROVAL`` is left as
   it is.

Pure: reads its arguments only. The returned ``TrajectoryInfo`` carries the flags this pipeline
raised; the engine adds the derived and lifecycle flags after it.
"""

from __future__ import annotations

import dataclasses

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import Decision, TrajectoryInfo
from agentfw.core.trajectory.flags import Flag

__all__ = ["band", "decide"]


def band(fused: float, theta_eff: float, cal_delta: float) -> Decision:
    """The §6.8 decision band of ``fused`` against ``theta_eff`` (NaN -> ``ALLOW``)."""
    if fused >= theta_eff:
        return Decision.BLOCK
    if fused >= theta_eff - cal_delta:
        return Decision.REQUIRE_APPROVAL
    return Decision.ALLOW


def decide(
    fused: float, traj: TrajectoryInfo, cfg: EngineConfig
) -> tuple[Decision, TrajectoryInfo]:
    """``(decision, trajectory)`` of one call after fusion (module docstring)."""
    decision = band(fused, traj.threshold_eff, cfg.cal_delta)
    # -- gate (Task 3.1), escalation (Task 3.2) --
    if cfg.provisional and decision is Decision.BLOCK and not traj.irreversible:
        decision = Decision.ALLOW
        traj = dataclasses.replace(traj, flags=traj.flags | {Flag.PROVISIONAL})
    # -- ceiling (Task 3.4), session overlay (Task 3.3) --
    return decision, traj
