"""Provisional flags for reversible calls (feature spec F6): grace TTL and exploration budget.

The step lifecycle of arXiv:2610.12375 (§4.5), paraphrased: a new step that looks wrong is not
judged at once. It stays open for a short grace period of later events; if a later step uses what
it produced, it was useful; if the grace period runs out first, it is counted as unresolved
exploration, and a session whose unresolved share of the total grows past a budget is treated as
off track.

Here a reversible call the decision bands would block is allowed with the flag ``PROVISIONAL``
(``decide``) and, once it has executed (outcome ``ALLOWED`` / ``APPROVED``: a call refused
outside the engine produces nothing to consume), opens a ``ProvEntry`` (``open_entry``) with
``ttl = prov_ttl`` and ``mass = fused``. ``advance`` runs once per later call of the session,
in this order:

(a) each open entry becomes ``validated`` when this call has a ledger ref with
    ``source == "response"`` and ``producer_seq == entry.seq_no`` (ruling R7: a ``task_input``
    ref carries ``producer_seq`` 0 whatever call 0 was, and an ``args`` ref names the first call
    that *used* an ungrounded id, not a producer); else ``absorbed`` when this call is on the
    entry's server (the tool id's part before the first ``.``), a novelty score of this call on
    the entry's field carries the entry's ``value_hash``, and ``fused < theta_eff`` (the novel
    value was a vocabulary shift);
(b) every still-open entry loses one ``ttl``; at 0 it is ``unresolved``, its mass is added to
    ``unresolved_mass`` and the call gets ``UNRESOLVED``;
(c) ``total_mass += fused``; from the ``prov_min_calls``-th call of the session on, an
    ``unresolved_mass / total_mass`` above ``prov_budget`` raises ``OFF_TRACK`` once
    (``off_track_raised``), cleared when the ratio is back at or under the budget.

Closed entries stay on the record until the next ``advance`` (so the transition is visible right
after the call that made it) and are dropped then, which bounds ``exploration`` to
``prov_ttl + 1`` entries. ``advance`` mutates the record it is given: the engine's ``score`` hands
it a copy (``SessionRecord.copy``), ``update`` the stored record. Only hashes and field paths are
kept, never a raw value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Self

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.trajectory.flags import Flag

if TYPE_CHECKING:
    from agentfw.core.trajectory.session import SessionRecord

__all__ = ["ProvEntry", "ProvState", "advance", "check_config", "open_entry"]

ProvState = Literal["provisional", "validated", "absorbed", "unresolved"]
_STATES: tuple[ProvState, ...] = ("provisional", "validated", "absorbed", "unresolved")
_NOVELTY = "novelty"


@dataclass(slots=True, frozen=True)
class ProvEntry:
    seq_no: int
    ttl: int
    mass: float
    server: str
    field: str | None
    value_hash: str | None
    state: ProvState = "provisional"

    def to_dict(self) -> dict[str, object]:
        return {
            "seq_no": self.seq_no,
            "ttl": self.ttl,
            "mass": self.mass,
            "server": self.server,
            "field": self.field,
            "value_hash": self.value_hash,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        state = d["state"]
        if state not in _STATES:
            raise ValueError(f"unknown provisional state {state!r}")
        return cls(
            int(d["seq_no"]),
            int(d["ttl"]),
            float(d["mass"]),
            str(d["server"]),
            None if d["field"] is None else str(d["field"]),
            None if d["value_hash"] is None else str(d["value_hash"]),
            state,
        )


def check_config(cfg: EngineConfig) -> None:
    """Reject provisional settings that make no sense (only called when ``provisional`` is on)."""
    if cfg.prov_ttl < 1:
        raise ValueError(f"prov_ttl must be >= 1, got {cfg.prov_ttl}")
    if not 0.0 <= cfg.prov_budget <= 1.0:
        raise ValueError(f"prov_budget must be in [0, 1], got {cfg.prov_budget}")
    if cfg.prov_min_calls < 0:
        raise ValueError(f"prov_min_calls must be >= 0, got {cfg.prov_min_calls}")


def _server(tool_id: str) -> str:
    return tool_id.split(".", 1)[0]


def _novel_value(per: Sequence[DetectorScore], cfg: EngineConfig) -> tuple[str | None, str | None]:
    """(field, value_hash) of the call's top novelty score when it fired (>= ``flag_min``)."""
    best: DetectorScore | None = None
    for s in per:
        fired = s.detector == _NOVELTY and s.score >= cfg.flag_min
        if fired and (best is None or s.score > best.score):
            best = s
    if best is None:
        return None, None
    vh = best.evidence.get("value_hash")
    return best.field, vh if isinstance(vh, str) else None


def open_entry(
    rec: SessionRecord,
    e: ActionEnvelope,
    fused: float,
    per: Sequence[DetectorScore],
    cfg: EngineConfig,
) -> None:
    """Append the ``ProvEntry`` of a call ``decide`` made provisional."""
    field, value_hash = _novel_value(per, cfg)
    rec.exploration.append(
        ProvEntry(e.seq_no, cfg.prov_ttl, fused, _server(e.tool_id), field, value_hash)
    )


def _validated(entry: ProvEntry, e: ActionEnvelope) -> bool:
    return any(r.source == "response" and r.producer_seq == entry.seq_no for r in e.ledger_refs)


def _absorbed(
    entry: ProvEntry,
    e: ActionEnvelope,
    fused: float,
    theta_eff: float,
    per: Sequence[DetectorScore],
) -> bool:
    if entry.value_hash is None or _server(e.tool_id) != entry.server or not fused < theta_eff:
        return False
    return any(
        s.detector == _NOVELTY
        and s.field == entry.field
        and s.evidence.get("value_hash") == entry.value_hash
        for s in per
    )


def advance(
    rec: SessionRecord,
    e: ActionEnvelope,
    fused: float,
    theta_eff: float,
    per: Sequence[DetectorScore],
    cfg: EngineConfig,
) -> set[Flag]:
    """One step of the lifecycle for call ``e`` (module docstring); mutates ``rec``."""
    out: set[Flag] = set()
    kept: list[ProvEntry] = []
    for entry in rec.exploration:
        if entry.state != "provisional":
            continue  # closed on an earlier call: dropped now
        if _validated(entry, e):
            entry = replace(entry, state="validated")
        elif _absorbed(entry, e, fused, theta_eff, per):
            entry = replace(entry, state="absorbed")
        else:
            entry = replace(entry, ttl=entry.ttl - 1)
            if entry.ttl <= 0:
                entry = replace(entry, ttl=0, state="unresolved")
                rec.unresolved_mass += entry.mass
                out.add(Flag.UNRESOLVED)
        kept.append(entry)
    rec.exploration = kept
    if math.isfinite(fused):
        rec.total_mass += fused
    over = rec.total_mass > 0.0 and rec.unresolved_mass / rec.total_mass > cfg.prov_budget
    if not over:
        rec.off_track_raised = False
    elif rec.n_calls + 1 >= cfg.prov_min_calls and not rec.off_track_raised:
        rec.off_track_raised = True
        out.add(Flag.OFF_TRACK)
    return out
