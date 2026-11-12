"""Session records and the per-session flag window (feature spec F7 substrate).

arXiv:2610.12375 (§1 S4, §6.3) argues that an aggregate score's meaning drifts with trajectory
length, so session-level decisions should come from the share of recent steps that raised a
severe per-step flag. This module keeps that share. One ``SessionRecord`` per session id (store
key ``sess:<session_id>``, kind ``sess``) holds the last ``session_window`` calls as
``(seq_no, sorted flag names)``, the session's call count and the timestamp of its latest call;
later tasks add the verdict and the provisional ledger with defaults.

``density`` (T-09) = Σ over the window's calls of that call's *highest* flag weight
(``flag_weight``), divided by ``session_window``, so it stays in [0, 1] and a call raising two
severe flags counts once. ``SessionBook.density_with`` computes it for a call that is not yet
recorded (the window's newest ``session_window - 1`` entries plus this call), reading the store
only: the engine's ``score`` stays pure. ``record`` appends in ``update``, for every outcome (a
blocked call is still a step of the session).

Eviction (T-15): every ``sess_sweep_every`` records, ``maybe_evict`` drops the records of
sessions without a call for more than ``sess_idle_ms`` (by the calls' own timestamps), except
halted ones. The sweep counter is bookkeeping, not detector state: it lives on the book, not in
the store, and ``reset`` zeroes it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope
from agentfw.core.state import StateStore, key_sess, register_record
from agentfw.core.trajectory.flags import Flag, flag_weight

__all__ = ["SessionBook", "SessionRecord"]


@register_record
@dataclass(slots=True)
class SessionRecord:
    KIND: ClassVar[str] = "sess"
    last_ts: int = 0
    n_calls: int = 0
    window: list[tuple[int, list[str]]] = field(default_factory=list)  # (seq_no, sorted names)
    verdict: str = "ok"
    halted: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "last_ts": self.last_ts,
            "n_calls": self.n_calls,
            "window": [[seq, list(names)] for seq, names in self.window],
            "verdict": self.verdict,
            "halted": self.halted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            int(d["last_ts"]),
            int(d["n_calls"]),
            [(int(seq), [str(n) for n in names]) for seq, names in d["window"]],
            str(d["verdict"]),
            bool(d["halted"]),
        )


class SessionBook:
    def __init__(self, store: StateStore, cfg: EngineConfig) -> None:
        if cfg.session_window < 1:
            raise ValueError(f"session_window must be >= 1, got {cfg.session_window}")
        if cfg.sess_sweep_every < 1:
            raise ValueError(f"sess_sweep_every must be >= 1, got {cfg.sess_sweep_every}")
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

    def record(self, e: ActionEnvelope, flags: Iterable[Flag]) -> None:
        key = key_sess(e.session_id)
        rec = self._store.get(key)
        if not isinstance(rec, SessionRecord):
            rec = SessionRecord()
            self._store.put(key, rec)
        rec.window.append((e.seq_no, sorted(f.value for f in flags)))
        del rec.window[: max(0, len(rec.window) - self._cfg.session_window)]
        rec.last_ts = max(rec.last_ts, e.ts_unix_ms) if rec.n_calls else e.ts_unix_ms
        rec.n_calls += 1
        self._since_sweep += 1

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
