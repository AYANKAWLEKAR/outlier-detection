"""Normalizer (spec 6.1): flatten args, derive fields, attach causal session context."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import orjson

from agentfw.core.envelope import (
    ActionEnvelope,
    IdSnapshot,
    PrevResponse,
    RawCall,
    Scalar,
)
from agentfw.core.ledger import ArtifactLedger
from agentfw.core.patterns import email_domain, id_prefix, is_id_like, url_host
from agentfw.core.sketch.bloom import BloomFilter

__all__ = ["SessionTracker", "flatten", "normalize"]

MAX_DEPTH = 16
FREETEXT_MIN_LEN = 40

_EMPTY_LEDGER = ArtifactLedger()
_Out = dict[str, list[Scalar]]


def _emit_scalar(path: str, v: object, flat: _Out, derived: _Out) -> None:
    if v is None or isinstance(v, bool | int | float):
        flat.setdefault(path, []).append(v)
        return
    s = v if isinstance(v, str) else str(v)
    if len(s) >= FREETEXT_MIN_LEN:
        derived.setdefault(f"{path}.__strlen__", []).append(len(s))
    else:
        flat.setdefault(path, []).append(s)
    if (dom := email_domain(s)) is not None:
        derived.setdefault(f"{path}.__domain__", []).append(dom)
    if (host := url_host(s)) is not None:
        derived.setdefault(f"{path}.__host__", []).append(host)
    if len(s) < FREETEXT_MIN_LEN and (pre := id_prefix(s)) is not None:
        derived.setdefault(f"{path}.__prefix__", []).append(pre)


def _walk(path: str, v: object, depth: int, flat: _Out, derived: _Out) -> None:
    if isinstance(v, Mapping):
        if depth >= MAX_DEPTH:
            derived.setdefault(f"{path}.__truncated__", []).append(1)
            return
        for k, child in v.items():
            _walk(f"{path}.{k}" if path else str(k), child, depth + 1, flat, derived)
    elif isinstance(v, list | tuple):
        if depth >= MAX_DEPTH:
            derived.setdefault(f"{path}.__truncated__", []).append(1)
            return
        derived.setdefault(f"{path}.__len__", []).append(len(v))
        for child in v:
            _walk(f"{path}[]", child, depth + 1, flat, derived)
    else:
        _emit_scalar(path, v, flat, derived)


def flatten(
    args: Mapping[str, object],
) -> tuple[dict[str, tuple[Scalar, ...]], dict[str, tuple[Scalar, ...]]]:
    """Return (flat_args, derived). Never raises on odd shapes."""
    flat: _Out = {}
    derived: _Out = {}
    _walk("", args, 0, flat, derived)
    return (
        {k: tuple(v) for k, v in flat.items()},
        {k: tuple(v) for k, v in derived.items()},
    )


@dataclass(slots=True)
class _Session:
    tools: deque[str]
    ids: set[str] = field(default_factory=set)
    bloom: BloomFilter | None = None
    prev: PrevResponse | None = None
    snap: IdSnapshot | None = None
    ledger: ArtifactLedger = field(default_factory=ArtifactLedger)


class SessionTracker:
    """Per-session causal context: ring buffer of tools, bounded id set (Bloom on overflow)."""

    def __init__(self, k: int = 4, max_ids: int = 2000) -> None:
        self.k = k
        self.max_ids = max_ids
        self._sessions: dict[str, _Session] = {}

    def snapshot(self, session_id: str) -> tuple[tuple[str, ...], IdSnapshot, PrevResponse | None]:
        s = self._sessions.get(session_id)
        if s is None:
            return (), IdSnapshot(frozenset(), None), None
        if s.snap is None:
            s.snap = IdSnapshot(frozenset(s.ids), s.bloom.copy() if s.bloom else None)
        return tuple(s.tools), s.snap, s.prev

    def ledger(self, session_id: str) -> ArtifactLedger | None:
        s = self._sessions.get(session_id)
        return None if s is None else s.ledger

    def _add_id(self, s: _Session, v: str) -> None:
        if v in s.ids or (s.bloom is not None and v in s.bloom):
            return
        if len(s.ids) < self.max_ids:
            s.ids.add(v)
        else:
            if s.bloom is None:
                s.bloom = BloomFilter(capacity=8 * self.max_ids)
            s.bloom.add(v)
        s.snap = None

    def observe(self, e: ActionEnvelope) -> None:
        s = self._sessions.get(e.session_id)
        if s is None:
            s = self._sessions[e.session_id] = _Session(
                tools=deque(maxlen=self.k),
                ledger=ArtifactLedger(self.max_ids, e.session_id),
            )
        s.ledger.seed_task(e.task_ids)
        s.ledger.record(e)
        s.tools.append(e.tool_id)
        for vals in e.flat_args.values():
            for v in vals:
                if isinstance(v, str) and is_id_like(v):
                    self._add_id(s, v)
        rm = e.response_meta
        if rm is not None:
            for v in rm.ids:
                self._add_id(s, v)
        s.prev = PrevResponse(e.tool_id, rm.item_count if rm is not None else None)

    def dump(self, path: str | Path) -> None:
        """Write one JSON line per session (sorted by session id, sorted keys)."""
        lines = []
        for sid in sorted(self._sessions):
            s = self._sessions[sid]
            rec: dict[str, Any] = {
                "session_id": sid,
                "k": self.k,
                "max_ids": self.max_ids,
                "tools": list(s.tools),
                "ids": sorted(s.ids),
                "bloom": s.bloom.to_dict() if s.bloom is not None else None,
                "prev": None if s.prev is None else [s.prev.tool_id, s.prev.item_count],
                "ledger": s.ledger.to_dict(),
            }
            lines.append(orjson.dumps(rec, option=orjson.OPT_SORT_KEYS))
        Path(path).write_bytes(b"\n".join(lines) + (b"\n" if lines else b""))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        recs = [orjson.loads(ln) for ln in Path(path).read_bytes().splitlines() if ln.strip()]
        tr = cls(k=recs[0]["k"], max_ids=recs[0]["max_ids"]) if recs else cls()
        for r in recs:
            s = _Session(
                tools=deque(r["tools"], maxlen=tr.k),
                ids=set(r["ids"]),
                bloom=BloomFilter.from_dict(r["bloom"]) if r["bloom"] is not None else None,
                prev=None if r["prev"] is None else PrevResponse(r["prev"][0], r["prev"][1]),
                ledger=ArtifactLedger.from_dict(r["ledger"]),
            )
            tr._sessions[r["session_id"]] = s
        return tr


def normalize(raw: RawCall, tracker: SessionTracker) -> ActionEnvelope:
    """Build an envelope from a raw call. Reads the tracker; never observes."""
    flat, derived = flatten(raw.args)
    prior_tools, ids_seen, prev = tracker.snapshot(raw.session_id)
    ledger = tracker.ledger(raw.session_id)
    refs = (_EMPTY_LEDGER if ledger is None else ledger).refs_for(flat, raw.task_ids)
    return ActionEnvelope(
        envelope_id=raw.envelope_id,
        schema_version=raw.schema_version,
        session_id=raw.session_id,
        agent_id=raw.agent_id,
        principal_id=raw.principal_id,
        seq_no=raw.seq_no,
        ts_unix_ms=raw.ts_unix_ms,
        tool_id=raw.tool_id,
        args=raw.args,
        response_meta=raw.response_meta,
        flat_args=flat,
        derived=derived,
        prior_tools=prior_tools,
        session_ids_seen=ids_seen,
        prev_response=prev,
        ledger_refs=refs,
        task_ids=raw.task_ids,
    )
