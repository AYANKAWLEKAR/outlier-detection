"""One readable line per reason (feature spec F8; T-48).

``render_reason(r, info, decision=...)`` reads, in order: the decision and the authority it rests
on (maturity and the evidence count that set it, pooled prior, the decision before a cap -- or,
when the session overlay blocked a call the ceiling had capped, that the halt overrode the cap
(T-62) -- F4's prerequisite block when it was kept above a ceiling that would have capped it),
what the reason
says, the gate's missing producer when the gate blocked, and the regime::

    blocked (mature: 1,140 observations); amount at 99.97th percentile (history+schemas,
    irreversible)

Raw ids never appear (global constraint): a reason is rendered from its detector, its field (an
argument *path*), its numbers and tool names only; ``observed`` is shown only when it is a number,
and the gate contributes its path and producer tool, never a value.
"""

from __future__ import annotations

from agentfw.core.envelope import Decision, Reason, TrajectoryInfo

__all__ = ["render_reason"]

_VERB = {
    Decision.BLOCK: "blocked",
    Decision.REQUIRE_APPROVAL: "held for approval",
    Decision.ALLOW: "allowed",
}
_NOTICE_VERB = {"warn": "allowed with a warning", "inform": "allowed, audit only"}
_VITAL = {
    "__loop__": "repeated identical calls",
    "__spin__": "no new information",
    "__stall__": "stalled progress",
}


def _count(v: object) -> str | None:
    """A numeric ``observed`` as text: an int exactly (it may exceed a float's range), a float
    with ``:g``; anything else ``None``."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return str(v) if isinstance(v, int) else f"{v:g}"


def _ordinal(p: float) -> str:
    """``0.9997`` -> ``"99.97th"``, ``0.01`` -> ``"1st"``, ``0.22`` -> ``"22nd"``."""
    text = f"{p * 100:.2f}".rstrip("0").rstrip(".")
    if "." in text:
        return f"{text}th"
    n = int(text)
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{text}{suffix}"


def _what(r: Reason) -> str:
    field = r.field
    observed = _count(r.observed)
    if r.detector == "magnitude":
        if r.percentile is not None:
            return f"{field} at {_ordinal(r.percentile)} percentile"
        return f"unusual {field}"
    if r.detector == "novelty":
        return f"unseen {field} value"
    if r.detector == "velocity":
        count = "" if observed is None else f": {observed} calls in the short window"
        return f"{field} total burst{count}" if field else f"call burst{count}"
    if r.detector == "sequence":
        if r.prereq_missing is not None:
            return f"no prior {r.prereq_missing}"
        return "unusual tool sequence"
    if r.detector == "vital":
        return _VITAL.get(field or "", f"vital sign {field}")
    if r.detector == "session":
        return "session halted"
    return f"{r.detector} anomaly" + ("" if field is None else f" on {field}")


def _authority(r: Reason, info: TrajectoryInfo, decision: Decision | None) -> str:
    maturity = info.maturity or r.maturity
    parts: list[str] = []
    if maturity is not None:  # the count that graded the call, never this reason's own (T-48)
        n = info.maturity_n if info.maturity is not None else None
        parts.append(maturity if n is None else f"{maturity}: {n:,} observations")
    if r.prior_used:
        parts.append("pooled prior")
    if info.capped_from is not None:
        if decision is Decision.BLOCK:  # only the session overlay blocks over a cap (T-47)
            parts.append(f"session halt over a cap from {info.capped_from.value}")
        else:
            parts.append(f"capped from {info.capped_from.value}")
    if decision is Decision.BLOCK and info.ceiling_exempt:
        parts.append("prerequisite exempt from the ceiling")
    if decision is None:
        return ", ".join(parts)
    verb = _NOTICE_VERB[info.notice] if info.notice is not None else _VERB[decision]
    return f"{verb} ({', '.join(parts)})" if parts else verb


def render_reason(r: Reason, info: TrajectoryInfo, *, decision: Decision | None = None) -> str:
    """The reason as one line (module docstring); without ``decision`` the line starts with the
    authority alone."""
    segments = [_authority(r, info, decision), _what(r)]
    gate = info.gate
    if gate is not None and gate.action == "block" and gate.missing_path is not None:
        producer = "" if gate.prereq_missing is None else f" (usually {gate.prereq_missing})"
        segments.append(f"{gate.missing_path} has no producer{producer}")
    regime = r.regime or info.regime
    tail = f"{regime}, irreversible" if info.irreversible else regime
    return "; ".join(s for s in segments if s) + f" ({tail})"
