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
``prior_tools`` (evidence ``prereq_missing`` = first missing tool, sorted). The mined set is cached
on the record; ``update`` refreshes it and ``score`` never writes (it recomputes if stale).

``score`` is pure; ``update`` is for allowed outcomes only. Task 1.8 adds the ``flow`` part in
``_parts``; the emitted score is the max over parts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Self

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.hints import ToolHint
from agentfw.core.schema_infer import TypeView
from agentfw.core.state import DictStore, key_prereq, key_trans, register_record

__all__ = ["NAME", "PrereqRecord", "SequenceDetector", "TransRecord"]

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


def _contexts(prior: tuple[str, ...]) -> tuple[str, str]:
    """(k=2 key ``"a|b"``, k=1 key) from the last two prior tools, padded with ``<s>``."""
    last = prior[-2:]
    a, b = (_PAD,) * (2 - len(last)) + tuple(last)
    return f"{a}|{b}", b


class SequenceDetector:
    name: str = NAME

    def __init__(
        self,
        store: DictStore,
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
        if missing:
            return 1.0, {"prereq_missing": missing[0]}
        return 0.0, {"prereq_missing": None}

    # --- protocol ----------------------------------------------------------------------

    def _parts(self, e: ActionEnvelope) -> list[Part]:
        """All sub-scores; Task 1.8 appends the ``flow`` part here."""
        s_t, ev_t = self._s_trans(e)
        s_p, ev_p = self._s_prereq(e)
        return [("trans", s_t, ev_t), ("prereq", s_p, ev_p)]

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        parts = self._parts(e)
        fired = max(parts, key=lambda p: p[1])  # first part wins ties
        evidence: dict[str, object] = {}
        for _, _, ev in parts:
            evidence.update(ev)
        evidence["fired"] = fired[0]
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
