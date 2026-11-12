"""Sequence detector: transition surprise and mined prerequisites (spec §6.6 a, b).

Transitions (a): the context is the last ``k`` entries of ``prior_tools`` (left-padded with
``"<s>"``). Back-off: k=2 if that context's total >= ``trans_min_ctx``, else k=1 if its total is,
else k=0 (unigram). With ``V = len(tools) + 1`` (the +1 is an "unseen" slot) the smoothed
probability is ``P = (c + a) / (total + a*V)``; ``H`` is the entropy of that distribution over the
V outcomes and ``s_trans = (1 - P) * (1 - H / ln V)`` (0 when ``V <= 2``). Evidence: ``k``, ``p``,
``entropy_ratio``.

Prerequisites (b): per (agent, tool Y) we count Y-observations in which tool X was anywhere in the
``prior_tools`` window (once per observation). X is a prerequisite when ``n_y >= prereq_min_n`` and
``cnt[X] / n_y >= prereq_rate``. ``s_prereq = 1`` if any prerequisite is absent from
``prior_tools`` (evidence ``prereq_missing`` = first missing tool, sorted) under the spec's
``seq_prereq_graded=False``; with ``seq_prereq_graded`` (Task 1.15, decision 008; the default by
user decision 2026-11-10) it is instead the highest ``cnt[X] / n_y`` among the missing
prerequisites. The mined set is cached
on the record; ``update`` refreshes it and ``score`` never writes (it recomputes if stale).

Id flow (c): for a *write* tool, each ID-typed flat value absent from ``session_ids_seen`` scores
``present / total`` of that (agent, tool, path)'s history (0 until ``total >= flow_min_obs``); the
max over missing values is ``s_flow``. Evidence: ``missing_id_path``, ``r_flow``, ``n`` (the
history's ``total``; the deepest over ID paths when nothing is missing), ``ref_source`` (never
raw ids).
With ``flow_ledger`` (Task 1.4) "present" means the value has a ``LedgerRef`` on this envelope
(an id produced earlier in the session, or seeded from the task input) or is in
``session_ids_seen`` (ids known only through the Bloom overflow); the history counts presence
the same way and ``FlowRecord.producers`` tallies the producer tool of each ref
(``usual_producer`` names the commonest). ``ref_source`` is the source of the first ref on the
flagged path (or, when nothing is missing, on any ID-typed path); None without one or without the
switch.
A tool is a write if its hint says so, else if it has been seen with an ID-typed argument and never
with a response ``item_count`` -- learned only from past ``update`` calls, so scoring is causal.

``score`` is pure; ``update`` is for allowed outcomes only. The emitted score is the max over the
parts (first part wins ties); ``fired`` is None when that max is 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Self

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType, LedgerRef
from agentfw.core.hints import ToolHint
from agentfw.core.schema_infer import TypeView
from agentfw.core.state import (
    StateStore,
    key_flow,
    key_kind,
    key_prereq,
    key_trans,
    register_record,
)

__all__ = [
    "NAME",
    "FlowRecord",
    "PrereqRecord",
    "SequenceDetector",
    "ToolKindRecord",
    "TransRecord",
]

NAME = "sequence"
_PAD = "<s>"

Part = tuple[str, float, dict[str, object]]


@register_record
@dataclass(slots=True)
class TransRecord:
    KIND: ClassVar[str] = "trans"
    c2: dict[str, dict[str, int]]
    c1: dict[str, dict[str, int]]
    c0: dict[str, int]
    tools: set[str]

    def __init__(
        self,
        c2: dict[str, dict[str, int]] | None = None,
        c1: dict[str, dict[str, int]] | None = None,
        c0: dict[str, int] | None = None,
        tools: set[str] | None = None,
    ) -> None:
        self.c2 = {} if c2 is None else c2
        self.c1 = {} if c1 is None else c1
        self.c0 = {} if c0 is None else c0
        self.tools = set() if tools is None else tools

    def to_dict(self) -> dict[str, object]:
        return {
            "c2": {k: dict(sorted(v.items())) for k, v in sorted(self.c2.items())},
            "c1": {k: dict(sorted(v.items())) for k, v in sorted(self.c1.items())},
            "c0": dict(sorted(self.c0.items())),
            "tools": sorted(self.tools),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            {k: {t: int(n) for t, n in v.items()} for k, v in d["c2"].items()},
            {k: {t: int(n) for t, n in v.items()} for k, v in d["c1"].items()},
            {t: int(n) for t, n in d["c0"].items()},
            {str(t) for t in d["tools"]},
        )


@register_record
@dataclass(slots=True)
class PrereqRecord:
    KIND: ClassVar[str] = "prereq"
    n_y: int
    cnt: dict[str, int]
    cached: tuple[int, frozenset[str]]  # (n_y at compute time, mined set)

    def __init__(
        self,
        n_y: int = 0,
        cnt: dict[str, int] | None = None,
        cached: tuple[int, frozenset[str]] | None = None,
    ) -> None:
        self.n_y = n_y
        self.cnt = {} if cnt is None else cnt
        self.cached = (0, frozenset()) if cached is None else cached

    def to_dict(self) -> dict[str, object]:
        return {
            "n_y": self.n_y,
            "cnt": dict(sorted(self.cnt.items())),
            "cached": [self.cached[0], sorted(self.cached[1])],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        n, items = d["cached"]
        return cls(
            int(d["n_y"]),
            {t: int(n_) for t, n_ in d["cnt"].items()},
            (int(n), frozenset(str(t) for t in items)),
        )


@register_record
@dataclass(slots=True)
class ToolKindRecord:
    KIND: ClassVar[str] = "toolkind"
    has_item_count: bool = False
    has_id_arg: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"has_item_count": self.has_item_count, "has_id_arg": self.has_id_arg}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(bool(d["has_item_count"]), bool(d["has_id_arg"]))


@register_record
@dataclass(slots=True)
class FlowRecord:
    KIND: ClassVar[str] = "flow"
    present: int = 0
    total: int = 0
    producers: dict[str, int] = field(default_factory=dict)  # producer tool -> count (flow_ledger)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"present": self.present, "total": self.total}
        if self.producers:  # omitted while empty: dumped state bytes match pre-ledger snapshots
            out["producers"] = dict(sorted(self.producers.items()))
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        prod = {str(t): int(n) for t, n in d.get("producers", {}).items()}
        return cls(int(d["present"]), int(d["total"]), prod)


def _contexts(prior: tuple[str, ...]) -> tuple[str, str]:
    """(k=2 key ``"a|b"``, k=1 key) from the last two prior tools, padded with ``<s>``."""
    last = prior[-2:]
    a, b = (_PAD,) * (2 - len(last)) + tuple(last)
    return f"{a}|{b}", b


class SequenceDetector:
    name: str = NAME

    def __init__(
        self,
        store: StateStore,
        cfg: EngineConfig,
        hints: Mapping[str, ToolHint] = MappingProxyType({}),
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._hints = hints

    # --- transitions -------------------------------------------------------------------

    def _trans(self, agent: str) -> TransRecord | None:
        rec = self._store.get(key_trans(agent))
        return rec if isinstance(rec, TransRecord) else None

    def _s_trans(self, e: ActionEnvelope) -> tuple[float, dict[str, object]]:
        rec = self._trans(e.agent_id)
        ev: dict[str, object] = {"k": 0, "p": 0.0, "entropy_ratio": 0.0}
        if rec is None:
            return 0.0, ev
        v = len(rec.tools) + 1
        k2, k1 = _contexts(e.prior_tools)
        min_ctx = self._cfg.trans_min_ctx
        dist = rec.c2.get(k2, {})
        k = 2
        if sum(dist.values()) < min_ctx:
            dist, k = rec.c1.get(k1, {}), 1
            if sum(dist.values()) < min_ctx:
                dist, k = rec.c0, 0
        a = self._cfg.trans_smoothing
        total = sum(dist.values())
        denom = total + a * v
        p = (dist.get(e.tool_id, 0) + a) / denom
        ev["k"], ev["p"] = k, p
        if v <= 2:
            return 0.0, ev
        h = -sum(((c + a) / denom) * math.log((c + a) / denom) for c in dist.values())
        zeros = v - len(dist)  # outcomes (incl. the unseen slot) with no counts
        if zeros > 0:
            q = a / denom
            h -= zeros * q * math.log(q)
        ratio = min(1.0, max(0.0, h / math.log(v)))
        ev["entropy_ratio"] = ratio
        return max(0.0, min(1.0, (1.0 - p) * (1.0 - ratio))), ev

    # --- prerequisites -----------------------------------------------------------------

    def _mine(self, rec: PrereqRecord) -> frozenset[str]:
        if rec.n_y < self._cfg.prereq_min_n:
            return frozenset()
        return frozenset(x for x, c in rec.cnt.items() if c / rec.n_y >= self._cfg.prereq_rate)

    def _mined(self, rec: PrereqRecord) -> frozenset[str]:
        """Cached set if fresh, else recomputed on the fly (never written here)."""
        return rec.cached[1] if rec.cached[0] == rec.n_y else self._mine(rec)

    def prerequisites(self, agent: str, tool: str) -> frozenset[str]:
        rec = self._store.get(key_prereq(agent, tool))
        return self._mined(rec) if isinstance(rec, PrereqRecord) else frozenset()

    def _s_prereq(self, e: ActionEnvelope) -> tuple[float, dict[str, object]]:
        missing = sorted(self.prerequisites(e.agent_id, e.tool_id) - set(e.prior_tools))
        if not missing:
            return 0.0, {"prereq_missing": None}
        s = 1.0
        if self._cfg.seq_prereq_graded:  # decision 008: strongest missing rule
            rec = self._store.get(key_prereq(e.agent_id, e.tool_id))
            assert isinstance(rec, PrereqRecord)  # a mined prerequisite implies a record
            s = max(rec.cnt[x] / rec.n_y for x in missing)
        return s, {"prereq_missing": missing[0]}

    # --- id flow -----------------------------------------------------------------------

    def is_write(self, agent: str, tool: str) -> bool:
        hint = self._hints.get(tool)
        if hint is not None and hint.kind is not None:
            return hint.kind == "write"
        rec = self._store.get(key_kind(agent, tool))
        return isinstance(rec, ToolKindRecord) and rec.has_id_arg and not rec.has_item_count

    def usual_producer(self, agent: str, tool: str, path: str) -> str | None:
        """Commonest producer tool of this (agent, tool, path)'s ids; ties by name. Read-only."""
        rec = self._store.get(key_flow(agent, tool, path))
        if not isinstance(rec, FlowRecord) or not rec.producers:
            return None
        return min(rec.producers.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def _is_present(self, e: ActionEnvelope, v: str, ref_ids: frozenset[str] | None) -> bool:
        """Is id ``v`` already known? ``ref_ids`` is this path's ledger-ref ids (None = v3)."""
        if ref_ids is not None and v in ref_ids:
            return True
        return v in e.session_ids_seen

    def _refs_by_path(self, e: ActionEnvelope) -> dict[str, list[LedgerRef]] | None:
        if not self._cfg.flow_ledger:
            return None
        out: dict[str, list[LedgerRef]] = {}
        for r in e.ledger_refs:
            out.setdefault(r.path, []).append(r)
        return out

    def _s_flow(self, e: ActionEnvelope, types: TypeView) -> tuple[float, dict[str, object]]:
        ev: dict[str, object] = {"missing_id_path": None, "r_flow": 0.0, "n": 0, "ref_source": None}
        if not self.is_write(e.agent_id, e.tool_id):
            return 0.0, ev
        refs = self._refs_by_path(e)
        best = -1.0
        first_ref: str | None = None
        n_any = 0
        for path in sorted(e.flat_args):
            if types(path) != FieldType.ID:
                continue
            seen_rec = self._store.get(key_flow(e.agent_id, e.tool_id, path))
            if isinstance(seen_rec, FlowRecord):
                n_any = max(n_any, seen_rec.total)
            prefs = refs.get(path, []) if refs is not None else []
            if first_ref is None and prefs:
                first_ref = prefs[0].source
            ids = frozenset(r.artifact_id for r in prefs) if refs is not None else None
            if not any(
                isinstance(v, str) and not self._is_present(e, v, ids) for v in e.flat_args[path]
            ):
                continue
            rec = self._store.get(key_flow(e.agent_id, e.tool_id, path))
            r = 0.0
            n = 0
            if isinstance(rec, FlowRecord):
                n = rec.total
                if rec.total >= self._cfg.flow_min_obs:
                    r = rec.present / rec.total
            if r > best:
                best = r
                ev["missing_id_path"], ev["r_flow"], ev["n"] = path, r, n
                ev["ref_source"] = prefs[0].source if prefs else None
        if best < 0.0:  # nothing missing: report the first ref and the deepest history
            ev["ref_source"], ev["n"] = first_ref, n_any
        return max(0.0, best), ev

    # --- protocol ----------------------------------------------------------------------

    def _parts(self, e: ActionEnvelope, types: TypeView) -> list[Part]:
        """All sub-scores in tie-break order."""
        s_t, ev_t = self._s_trans(e)
        s_p, ev_p = self._s_prereq(e)
        s_f, ev_f = self._s_flow(e, types)
        return [("trans", s_t, ev_t), ("prereq", s_p, ev_p), ("flow", s_f, ev_f)]

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        parts = self._parts(e, types)
        fired = max(parts, key=lambda p: p[1])  # first part wins ties
        evidence: dict[str, object] = {}
        for _, _, ev in parts:
            evidence.update(ev)
        evidence["fired"] = fired[0] if fired[1] > 0.0 else None
        return [DetectorScore(NAME, None, fired[1], evidence)]

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        key = key_trans(e.agent_id)
        trec = self._trans(e.agent_id)
        if trec is None:
            trec = TransRecord()
            self._store.put(key, trec)
        k2, k1 = _contexts(e.prior_tools)
        t = e.tool_id
        d2 = trec.c2.setdefault(k2, {})
        d2[t] = d2.get(t, 0) + 1
        d1 = trec.c1.setdefault(k1, {})
        d1[t] = d1.get(t, 0) + 1
        trec.c0[t] = trec.c0.get(t, 0) + 1
        trec.tools.add(t)

        pkey = key_prereq(e.agent_id, t)
        prec = self._store.get(pkey)
        if not isinstance(prec, PrereqRecord):
            prec = PrereqRecord()
            self._store.put(pkey, prec)
        prec.n_y += 1
        for x in set(e.prior_tools):
            prec.cnt[x] = prec.cnt.get(x, 0) + 1
        prec.cached = (prec.n_y, self._mine(prec))

        self._update_flow(e, types)

    def _update_flow(self, e: ActionEnvelope, types: TypeView) -> None:
        kkey = key_kind(e.agent_id, e.tool_id)
        krec = self._store.get(kkey)
        if not isinstance(krec, ToolKindRecord):
            krec = ToolKindRecord()
            self._store.put(kkey, krec)
        if e.response_meta is not None and e.response_meta.item_count is not None:
            krec.has_item_count = True
        pr = e.prev_response
        if pr is not None and pr.item_count is not None:
            # D-004: the previous call's response arrives on this envelope (pre-response path).
            pkey = key_kind(e.agent_id, pr.tool_id)
            prec = self._store.get(pkey)
            if not isinstance(prec, ToolKindRecord):
                prec = ToolKindRecord()
                self._store.put(pkey, prec)
            prec.has_item_count = True
        refs = self._refs_by_path(e)
        for path, vals in e.flat_args.items():
            if types(path) != FieldType.ID:
                continue
            strs = [v for v in vals if isinstance(v, str)]
            if not strs:
                continue
            krec.has_id_arg = True
            fkey = key_flow(e.agent_id, e.tool_id, path)
            frec = self._store.get(fkey)
            if not isinstance(frec, FlowRecord):
                frec = FlowRecord()
                self._store.put(fkey, frec)
            ids: frozenset[str] | None = None
            by_id: dict[str, LedgerRef] = {}
            if refs is not None:
                by_id = {r.artifact_id: r for r in refs.get(path, [])}
                ids = frozenset(by_id)
            for v in strs:
                frec.total += 1
                if self._is_present(e, v, ids):
                    frec.present += 1
                ref = by_id.get(v)
                if ref is not None and ref.producer_tool is not None:
                    frec.producers[ref.producer_tool] = frec.producers.get(ref.producer_tool, 0) + 1
