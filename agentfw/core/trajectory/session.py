"""Session records, the per-session flag window and session verdicts (feature spec F7).

arXiv:2610.12375 (§1 S4, §6.3) argues that an aggregate score's meaning drifts with trajectory
length, so session-level decisions should come from the share of recent steps that raised a
severe per-step flag; it reports that stopping at the first flag stops almost every run, while
a density threshold degrades gradually as the threshold moves. This module keeps that share and
the verdict it gives. One ``SessionRecord`` per session id (store key ``sess:<session_id>``, kind
``sess``) holds the last ``session_window`` calls as ``(seq_no, sorted flag names)``, the
session's call count and the timestamp of its latest call, the verdict of its latest call and
whether it is halted, the session's fused mass, and the provisional lifecycle state of F6
(``core.trajectory.provisional``): the open ``exploration`` entries, the session's scored
``total_mass``, its ``unresolved_mass`` and whether ``OFF_TRACK`` is currently raised.

``density`` (T-09) = Σ over the window's calls of that call's *highest* flag weight
(``flag_weight``), divided by ``session_window``, so it stays in [0, 1] and a call raising two
severe flags counts once. ``SessionBook.density_with`` computes it for a call that is not yet
recorded (the window's newest ``session_window - 1`` entries plus this call), reading the store
only: the engine's ``score`` stays pure. ``record`` appends in ``update``, for every outcome (a
blocked call is still a step of the session).

Verdicts (Task 3.3, T-43): ``ladder`` maps the density including this call to
``SessionVerdict``: ``>= theta_halt`` -> ``HALT``, ``>= theta_warn`` -> ``WARN``, else ``OK``;
a session halted before this call reports ``HALT`` whatever its density. ``record`` stores the
call's verdict and makes ``HALT`` sticky (``halted``) until ``release`` clears the halt, the
verdict and the window. ``fused_mass`` (Σ of the session's finite fused scores, reported as
``TrajectoryInfo.session_mass``) is the session aggregate: reported, never thresholded -- no
verdict reads it.

Eviction (T-15): every ``sess_sweep_every`` records, ``maybe_evict`` drops the records of
sessions without a call for more than ``sess_idle_ms`` (by the calls' own timestamps), except
halted ones; an evicted session that calls again starts a fresh record. The sweep counter is
bookkeeping, not detector state: it lives on the book, not in the store, and ``reset`` zeroes it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Self

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope
from agentfw.core.state import StateStore, key_sess, register_record
from agentfw.core.trajectory.flags import Flag, flag_weight
from agentfw.core.trajectory.provisional import ProvEntry

__all__ = ["SessionBook", "SessionRecord", "SessionVerdict", "ladder"]


class SessionVerdict(StrEnum):
    OK = "ok"
    WARN = "warn_session"
    HALT = "halt_session"


def ladder(density: float, halted: bool, cfg: EngineConfig) -> SessionVerdict:
    """The verdict of a call whose session density, this call included, is ``density``;
    ``HALT`` whatever the density once the session was ``halted`` before it."""
    if halted or density >= cfg.theta_halt:
        return SessionVerdict.HALT
    return SessionVerdict.WARN if density >= cfg.theta_warn else SessionVerdict.OK


@register_record
@dataclass(slots=True)
class SessionRecord:
    KIND: ClassVar[str] = "sess"
    last_ts: int = 0
    n_calls: int = 0
    window: list[tuple[int, list[str]]] = field(default_factory=list)  # (seq_no, sorted names)
    verdict: str = "ok"
    halted: bool = False
    exploration: list[ProvEntry] = field(default_factory=list)
    total_mass: float = 0.0
    unresolved_mass: float = 0.0
    off_track_raised: bool = False
    fused_mass: float = 0.0  # the session aggregate (reported, never thresholded)

    def copy(self) -> SessionRecord:
        """An independent copy (``ProvEntry`` is frozen, so copying the lists suffices)."""
        return dataclasses.replace(
            self,
            window=[(seq, list(names)) for seq, names in self.window],
            exploration=list(self.exploration),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "last_ts": self.last_ts,
            "n_calls": self.n_calls,
            "window": [[seq, list(names)] for seq, names in self.window],
            "verdict": self.verdict,
            "halted": self.halted,
            "exploration": [p.to_dict() for p in self.exploration],
            "total_mass": self.total_mass,
            "unresolved_mass": self.unresolved_mass,
            "off_track_raised": self.off_track_raised,
            "fused_mass": self.fused_mass,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            int(d["last_ts"]),
            int(d["n_calls"]),
            [(int(seq), [str(n) for n in names]) for seq, names in d["window"]],
            str(d["verdict"]),
            bool(d["halted"]),
            [ProvEntry.from_dict(p) for p in d.get("exploration", ())],
            float(d.get("total_mass", 0.0)),
            float(d.get("unresolved_mass", 0.0)),
            bool(d.get("off_track_raised", False)),
            float(d.get("fused_mass", 0.0)),
        )


class SessionBook:
    def __init__(self, store: StateStore, cfg: EngineConfig) -> None:
        if cfg.session_window < 1:
            raise ValueError(f"session_window must be >= 1, got {cfg.session_window}")
        if cfg.sess_sweep_every < 1:
            raise ValueError(f"sess_sweep_every must be >= 1, got {cfg.sess_sweep_every}")
        if not 0.0 < cfg.theta_warn <= cfg.theta_halt <= 1.0:
            raise ValueError(
                "need 0 < theta_warn <= theta_halt <= 1, got "
                f"theta_warn={cfg.theta_warn}, theta_halt={cfg.theta_halt}"
            )
        self._store = store
        self._cfg = cfg
        self._since_sweep = 0

    def reset(self) -> None:
        self._since_sweep = 0

    def peek(self, session_id: str) -> SessionRecord | None:
        rec = self._store.get(key_sess(session_id))
        return rec if isinstance(rec, SessionRecord) else None

    def _top(self, names: Iterable[str]) -> float:
        return max((flag_weight(Flag(n), self._cfg) for n in names), default=0.0)

    def density_with(self, rec: SessionRecord | None, flags: Iterable[Flag]) -> float:
        """Density of the window that recording a call with ``flags`` would leave (T-09)."""
        w = self._cfg.session_window
        prior = rec.window[-(w - 1) :] if rec is not None and w > 1 else []
        total = sum(self._top(names) for _, names in prior) + self._top(flags)
        return total / w

    def ensure(self, session_id: str) -> SessionRecord:
        """The session's stored record, created (and stored) when absent. ``update`` only."""
        key = key_sess(session_id)
        rec = self._store.get(key)
        if not isinstance(rec, SessionRecord):
            rec = SessionRecord()
            self._store.put(key, rec)
        return rec

    def verdict_with(self, rec: SessionRecord | None, flags: Iterable[Flag]) -> SessionVerdict:
        """The verdict recording a call with ``flags`` would give (pure; ``ladder``)."""
        halted = rec is not None and rec.halted
        return ladder(self.density_with(rec, flags), halted, self._cfg)

    @staticmethod
    def mass_with(rec: SessionRecord | None, fused: float) -> float:
        """The session's fused mass once a call scoring ``fused`` is recorded (pure)."""
        mass = 0.0 if rec is None else rec.fused_mass
        return mass + fused if math.isfinite(fused) else mass

    def record(self, e: ActionEnvelope, flags: Iterable[Flag], fused: float = math.nan) -> None:
        """Append the call to its session's window with its verdict (``verdict_with``, read
        before the append; ``HALT`` sets ``halted``) and add its finite ``fused`` to the mass."""
        rec = self.ensure(e.session_id)
        flags = list(flags)
        verdict = self.verdict_with(rec, flags)
        rec.fused_mass = self.mass_with(rec, fused)
        rec.window.append((e.seq_no, sorted(f.value for f in flags)))
        del rec.window[: max(0, len(rec.window) - self._cfg.session_window)]
        rec.last_ts = max(rec.last_ts, e.ts_unix_ms) if rec.n_calls else e.ts_unix_ms
        rec.n_calls += 1
        rec.verdict = verdict.value
        rec.halted = rec.halted or verdict is SessionVerdict.HALT
        self._since_sweep += 1

    def release(self, session_id: str) -> None:
        """Clear a session's halt, verdict and window (``Engine.release_session``); its call
        count, mass and provisional state stay. No-op for a session without a record."""
        rec = self.peek(session_id)
        if rec is None:
            return
        rec.halted = False
        rec.verdict = SessionVerdict.OK.value
        rec.window.clear()

    def maybe_evict(self, now_ms: int) -> int:
        """Every ``sess_sweep_every`` records: drop non-halted sessions idle > ``sess_idle_ms``."""
        if self._since_sweep < self._cfg.sess_sweep_every:
            return 0
        self._since_sweep = 0
        idle = self._cfg.sess_idle_ms
        stale = [
            k
            for k, r in self._store.items()
            if isinstance(r, SessionRecord) and not r.halted and now_ms - r.last_ts > idle
        ]
        for k in stale:
            self._store.delete(k)
        return len(stale)
