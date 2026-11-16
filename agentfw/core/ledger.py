"""Per-session artifact ledger: where each id in a session first came from.

The ledger records, for every id-like value a session has touched, its first sighting: the
call that produced it (an id first seen in a call's args, or first returned in a response) or
the task input that seeded it. Later calls that use the id are recorded as consumers. This is
the ledger lookup of Definition 1 in arXiv:2610.12375, paraphrased: a call's id argument is
"grounded" when the session ledger already holds a producer for it, and ungrounded otherwise.

Rules:
* The first sighting wins; consumers only append. One exception (T-59): an exact entry first
  seen in a call's *args* has no producer, so when a response later returns that id (the same
  call's included: a lookup that echoes the id it was given) the response becomes its producer
  (source ``"response"``, that call's seq and tool; consumers kept). It is not a new artifact.
  An id only ever used in args (e.g. a fabricated id whose lookup errored) stays ``"args"``.
* An id first seen in a call's own args is not its own producer: refs are computed from the
  ledger state *before* the call is recorded (see ``normalize``).
* At most ``max_entries`` exact entries per session; beyond that ids go to a Bloom filter
  (membership only, no producer), ``degraded`` turns True and a warning is logged once.
* Ledgers are strictly per session; the tracker owns one per ``session_id``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentfw.core.envelope import ActionEnvelope, LedgerRef
from agentfw.core.patterns import is_id_like
from agentfw.core.sketch.bloom import BloomFilter

__all__ = ["ArtifactLedger", "LedgerEntry", "LedgerRef"]

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class LedgerEntry:
    producer_seq: int
    producer_tool: str | None
    source: str
    consumers: list[int] = field(default_factory=list)


class ArtifactLedger:
    def __init__(self, max_entries: int = 2000, session_id: str = "") -> None:
        self.max_entries = max_entries
        self.session_id = session_id
        self.degraded = False
        self._entries: dict[str, LedgerEntry] = {}
        self._bloom: BloomFilter | None = None

    def lookup(self, v: str) -> LedgerEntry | None:
        return self._entries.get(v)

    def __contains__(self, v: object) -> bool:
        if not isinstance(v, str):
            return False
        return v in self._entries or (self._bloom is not None and v in self._bloom)

    def __len__(self) -> int:
        return len(self._entries)

    def _first_sighting(self, v: str, seq: int, tool: str | None, source: str) -> bool:
        """Record ``v``'s first sighting; True when it was new (exact entry or Bloom add)."""
        if v in self:
            return False
        if len(self._entries) < self.max_entries:
            self._entries[v] = LedgerEntry(seq, tool, source)
            return True
        if self._bloom is None:
            self._bloom = BloomFilter(capacity=8 * self.max_entries)
            self.degraded = True
            _log.warning(
                "artifact ledger for session %r exceeded %d exact entries; "
                "falling back to Bloom membership",
                self.session_id,
                self.max_entries,
            )
        self._bloom.add(v)
        return True

    def seed_task(self, ids: Iterable[str]) -> None:
        for v in ids:
            self._first_sighting(v, 0, None, "task_input")

    def record(self, e: ActionEnvelope) -> int:
        """Record one call: arg ids are consumed (or first-seen), response ids first-seen.

        Returns how many new artifacts the call created (first sightings, Bloom ones included;
        task-input seeds are not the call's).
        """
        created = 0
        used: set[str] = set()
        for path in sorted(e.flat_args):
            for v in e.flat_args[path]:
                if not isinstance(v, str) or not is_id_like(v) or v in used:
                    continue
                used.add(v)
                entry = self._entries.get(v)
                if entry is not None:
                    entry.consumers.append(e.seq_no)
                else:
                    created += self._first_sighting(v, e.seq_no, e.tool_id, "args")
        if e.response_meta is not None:
            for v in e.response_meta.ids:
                entry = self._entries.get(v)
                if entry is not None and entry.source == "args":  # now it has a producer
                    entry.producer_seq, entry.producer_tool = e.seq_no, e.tool_id
                    entry.source = "response"
                    continue
                created += self._first_sighting(v, e.seq_no, e.tool_id, "response")
        return created

    def refs_for(
        self, flat_args: dict[str, tuple[Any, ...]] | Any, task_ids: Iterable[str] = ()
    ) -> tuple[LedgerRef, ...]:
        """Refs for the id-like args, in sorted path order, from the current state plus seeds."""
        seeds = frozenset(task_ids)
        out: list[LedgerRef] = []
        for path in sorted(flat_args):
            seen: set[str] = set()
            for v in flat_args[path]:
                if not isinstance(v, str) or v in seen or not is_id_like(v):
                    continue
                seen.add(v)
                entry = self._entries.get(v)
                if entry is not None:
                    src: Any = entry.source
                    out.append(LedgerRef(v, path, entry.producer_seq, entry.producer_tool, src))
                elif v in seeds:
                    out.append(LedgerRef(v, path, 0, None, "task_input"))
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "session_id": self.session_id,
            "degraded": self.degraded,
            "entries": {
                k: [en.producer_seq, en.producer_tool, en.source, list(en.consumers)]
                for k, en in sorted(self._entries.items())
            },
            "bloom": self._bloom.to_dict() if self._bloom is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactLedger:
        led = cls(int(d["max_entries"]), str(d.get("session_id", "")))
        led.degraded = bool(d["degraded"])
        for k, (seq, tool, source, consumers) in d["entries"].items():
            led._entries[k] = LedgerEntry(int(seq), tool, source, [int(c) for c in consumers])
        if d.get("bloom") is not None:
            led._bloom = BloomFilter.from_dict(d["bloom"])
        return led
