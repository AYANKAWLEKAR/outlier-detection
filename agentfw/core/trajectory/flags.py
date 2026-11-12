"""Per-call flags (feature spec F7): the discrete per-step signals a session window counts.

The session-level abort policy of arXiv:2610.12375 (§6.3) reacts to the density of *severe*
per-step flags in a short rolling window rather than to an aggregate score, so each call is first
reduced to a set of named flags. ``derive_flags`` reads only the call's detector scores, its
decision, its irreversibility and its tool hint, so it is pure. A score counts from
``EngineConfig.flag_min``:

* ``BLOCK``: the decision is ``BLOCK``.
* ``PREREQ_MISSING``: the sequence score fired on its prerequisite part and is >= ``flag_min``.
* ``ID_FLOW_VIOLATION``: an irreversible call whose sequence evidence names a missing id path
  with ``r_flow >= flag_min``.
* ``LOOP`` / ``SPIN`` / ``STALL``: the vital ``__loop__`` / ``__spin__`` / ``__stall__`` score is
  1.0. ``LOOP`` is exempt on a tool annotated read-only *and* idempotent (polling is expected
  there; F2's risk mitigation): the vital score still reports the repeat.
* ``VELOCITY_BURST``: a velocity score >= ``flag_min``.
* ``NOVEL_READ``: a novelty score >= ``flag_min`` on a reversible call.

``OFF_TRACK``, ``PROVISIONAL``, ``UNRESOLVED`` and ``COLD_START`` are raised by later stages.
The first eight members are severe (weight 1.0); ``UNRESOLVED`` weighs
``prov_unresolved_weight``; the rest weigh 0 (reported, never counted).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import Decision, DetectorScore
from agentfw.core.hints import ToolHint

__all__ = ["SEVERE", "Flag", "derive_flags", "flag_weight"]


class Flag(StrEnum):
    BLOCK = "BLOCK"
    PREREQ_MISSING = "PREREQ_MISSING"
    ID_FLOW_VIOLATION = "ID_FLOW_VIOLATION"
    LOOP = "LOOP"
    SPIN = "SPIN"
    STALL = "STALL"
    OFF_TRACK = "OFF_TRACK"
    VELOCITY_BURST = "VELOCITY_BURST"
    PROVISIONAL = "PROVISIONAL"
    UNRESOLVED = "UNRESOLVED"
    NOVEL_READ = "NOVEL_READ"
    COLD_START = "COLD_START"


SEVERE: frozenset[Flag] = frozenset(list(Flag)[:8])

_VITAL = {"__loop__": Flag.LOOP, "__spin__": Flag.SPIN, "__stall__": Flag.STALL}


def flag_weight(f: Flag, cfg: EngineConfig) -> float:
    """1.0 for a severe flag, ``prov_unresolved_weight`` for ``UNRESOLVED``, else 0."""
    if f in SEVERE:
        return 1.0
    if f is Flag.UNRESOLVED:
        return cfg.prov_unresolved_weight
    return 0.0


def _polling(hint: ToolHint | None) -> bool:
    ann = None if hint is None else hint.annotations
    return ann is not None and ann.read_only is True and ann.idempotent is True


def _num(v: object) -> float:
    return float(v) if isinstance(v, int | float) else 0.0


def derive_flags(
    per: Sequence[DetectorScore],
    decision: Decision,
    irreversible: bool,
    hint: ToolHint | None,
    cfg: EngineConfig,
) -> set[Flag]:
    """The call's flags (module docstring) from its scores, decision and tool facts."""
    lo = cfg.flag_min
    out: set[Flag] = set()
    if decision is Decision.BLOCK:
        out.add(Flag.BLOCK)
    for s in per:
        det, ev = s.detector, s.evidence
        if det == "sequence":
            if ev.get("fired") == "prereq" and s.score >= lo:
                out.add(Flag.PREREQ_MISSING)
            missing = ev.get("missing_id_path") is not None
            if irreversible and missing and _num(ev.get("r_flow")) >= lo:
                out.add(Flag.ID_FLOW_VIOLATION)
        elif det == "vital":
            flag = _VITAL.get(s.field or "")
            exempt = flag is Flag.LOOP and _polling(hint)
            if flag is not None and s.score == 1.0 and not exempt:
                out.add(flag)
        elif det == "velocity" and s.score >= lo:
            out.add(Flag.VELOCITY_BURST)
        elif det == "novelty" and s.score >= lo and not irreversible:
            out.add(Flag.NOVEL_READ)
    return out
