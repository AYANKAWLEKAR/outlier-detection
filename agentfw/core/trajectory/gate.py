"""Pre-execution gate for irreversible calls (feature spec F4; T-11, T-41).

The cheap path scores every call; a stricter synchronous check runs only before an
irreversible call executes, so its cost is paid on the rare actions that warrant it. A gate that
reads cached session state one event old blocks valid calls whose ids the previous call just
produced, so the gate re-reads everything it needs at call time.

``gate`` runs from ``decide`` only when ``EngineConfig.gate`` is on and the call is irreversible:

1. **Fresh state.** It reads the call's own envelope (``ledger_refs`` and ``session_ids_seen``,
   built by the normalizer from the tracker's *current* session state) and the sequence
   detector's ``FlowRecord`` / ``usual_producer`` straight from the store. Nothing is cached
   between calls and nothing is written, so ``score`` stays pure.
2. **Prerequisites** (only when the regime has schemas, ``history+schemas`` / ``schemas_only``;
   without schemas the gate is a distribution check only, G-d). Every id-like string value
   (``is_id_like``: an artifact the ledger can track) of an ID-typed flat path (the sequence
   detector's id-flow paths) that is not *grounded* is *missing*. Grounded (T-59): a
   ``LedgerRef`` at that path whose source is ``"response"`` (an earlier call returned it) or
   ``"task_input"`` (the task seeded it); with no ref at all, membership in
   ``session_ids_seen`` (a Bloom-only id: the ledger overflowed and keeps no producer, T-11). A
   ref with source ``"args"`` is never grounding: an id first seen in an earlier call's
   arguments has no producer (T-36), so a fabricated id cannot be laundered through a lookup
   that used it. A value inference types ID but that no
   ledger could ever record (free text such as a short ``subject``) is never missing. A missing
   value on a warm key (``FlowRecord.total >= flow_min_obs``) whose historical presence rate
   ``present / total >= gate_flow_rate`` -> ``block``, naming the path and the key's usual
   producer tool (``None`` when no producer was ever tallied, e.g. without ``flow_ledger``).
   A missing value on a cold key -> ``escalate`` (T-11); a warm key whose ids usually do not
   flow is left to the distribution check. Paths are visited in sorted order; the first block
   wins, else the first cold path is reported.
3. **Distribution.** ``fused >= theta_eff`` -> ``escalate``. Otherwise ``pass``.

``decide`` maps the outcome (T-41): ``block`` -> ``BLOCK`` with flag ``PREREQ_MISSING``;
``escalate`` -> ``REQUIRE_APPROVAL`` (the escalation stage, Task 3.2, decides it later);
``pass`` keeps the band's decision. Raw ids never leave this module: the outcome carries a path
and a tool name only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.sequence import SequenceDetector
from agentfw.core.envelope import ActionEnvelope, FieldType, Regime
from agentfw.core.patterns import is_id_like
from agentfw.core.schema_infer import TypeView

__all__ = ["SCHEMA_REGIMES", "GateAction", "GateOutcome", "gate"]

GateAction = Literal["pass", "block", "escalate"]
#: Regimes with a captured schema behind the tool: the prerequisite check runs only there (G-d).
SCHEMA_REGIMES: frozenset[Regime] = frozenset({"history+schemas", "schemas_only"})


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """What the gate decided for one irreversible call. ``prereq_missing`` is the usual producer
    tool of ``missing_path``'s ids (a tool name, never an id)."""

    action: GateAction
    prereq_missing: str | None
    missing_path: str | None


_PASS = GateOutcome("pass", None, None)


#: Ledger sources that ground an id (T-59): produced by an earlier response or seeded by the task.
_GROUNDING = frozenset({"response", "task_input"})


def _missing(e: ActionEnvelope, path: str, refs: Mapping[str, str]) -> bool:
    """Does ``path`` carry an id-like value that is not grounded (module docstring, step 2)?
    ``refs`` maps the path's ref'd ids to their ledger source."""
    seen = e.session_ids_seen
    for v in e.flat_args[path]:
        if not isinstance(v, str) or not is_id_like(v):
            continue
        src = refs.get(v)
        if src is None:
            if v not in seen:  # no ref: present only through bare (Bloom-only) membership
                return True
        elif src not in _GROUNDING:  # first seen in an earlier call's args: no producer
            return True
    return False


def gate(
    e: ActionEnvelope,
    types: TypeView,
    seq: SequenceDetector,
    fused: float,
    theta_eff: float,
    cfg: EngineConfig,
    *,
    regime: Regime,
) -> GateOutcome:
    """The gate's verdict on irreversible call ``e`` (module docstring). Pure: reads only."""
    cold: str | None = None
    if regime in SCHEMA_REGIMES:
        refs: dict[str, dict[str, str]] = {}
        for r in e.ledger_refs:
            refs.setdefault(r.path, {})[r.artifact_id] = r.source
        agent, tool = e.agent_id, e.tool_id
        for path in sorted(e.flat_args):
            if types(path) != FieldType.ID or not _missing(e, path, refs.get(path, {})):
                continue
            rec = seq.flow_record(agent, tool, path)
            if rec is None or rec.total == 0 or rec.total < cfg.flow_min_obs:
                if cold is None:
                    cold = path
            elif rec.present / rec.total >= cfg.gate_flow_rate:
                return GateOutcome("block", seq.usual_producer(agent, tool, path), path)
        if cold is not None:
            return GateOutcome("escalate", seq.usual_producer(agent, tool, cold), cold)
    if fused >= theta_eff:
        return GateOutcome("escalate", None, None)
    return _PASS
