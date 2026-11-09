"""Novelty detector: Good-Turing unseen-mass per (agent, tool, path) (spec §6.5).

Scored paths: CATEGORICAL and ID typed paths, plus the derived ``__domain__`` / ``__host__`` /
``__prefix__`` paths (categorical by construction). Values are keyed by ``str(v)``.

A value is *seen* when it is in the record's LRU (``novelty_lru`` entries) or its count-min
estimate in the lazily allocated ``evicted`` sketch is > 0 (D-011). Seen scores 0; unseen scores
``(1 - n_singletons / n_total) * min(1, n_total / novelty_full_n)``; a path's score is the max
over its values. Evidence: ``p_unseen``, ``n_total``, ``n_distinct``, ``value_hash`` (xxhash64 hex),
``prior_used``; the raw value is added only for non-ID types.

Cold keys (``n_total < mag_cold_n``, ruling NV1) may consult the pooled prior (§6.10) via
``PriorSource.novelty_ref``: a value is then seen if either record has it, and an unseen value is
scored with the prior's counts times ``prior_novelty_discount`` (D-012). With no prior the
behaviour is exactly the formula above.

``score`` is pure: it never reorders the LRU. Only ``update`` touches LRU order. When an evicted
value reappears it is re-inserted with count ``estimate + 1`` (and ``n_singletons`` decremented if
the estimate was 1).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import xxhash

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import PriorSource, iter_fields
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType, Scalar
from agentfw.core.schema_infer import TypeView
from agentfw.core.sketch.cms import CountMinSketch
from agentfw.core.state import DictStore, key_nov, register_record

__all__ = ["NAME", "NoveltyDetector", "NoveltyRecord"]

NAME = "novelty"
_SCORED = frozenset({FieldType.CATEGORICAL, FieldType.ID})


@register_record
@dataclass(slots=True)
class NoveltyRecord:
    KIND: ClassVar[str] = "nov"
    seen: OrderedDict[str, int]
    n_total: int
    n_singletons: int
    n_distinct: int
    evicted: CountMinSketch | None

    def __init__(
        self,
        seen: OrderedDict[str, int] | None = None,
        n_total: int = 0,
        n_singletons: int = 0,
        n_distinct: int = 0,
        evicted: CountMinSketch | None = None,
    ) -> None:
        self.seen = OrderedDict() if seen is None else seen
        self.n_total = n_total
        self.n_singletons = n_singletons
        self.n_distinct = n_distinct
        self.evicted = evicted

    def is_seen(self, v: str) -> bool:
        return v in self.seen or (self.evicted is not None and self.evicted.estimate(v) > 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "seen": [[v, c] for v, c in self.seen.items()],  # LRU order, oldest first
            "n_total": self.n_total,
            "n_singletons": self.n_singletons,
            "n_distinct": self.n_distinct,
            "evicted": None if self.evicted is None else self.evicted.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        ev = d.get("evicted")
        return cls(
            OrderedDict((str(v), int(c)) for v, c in d["seen"]),
            int(d["n_total"]),
            int(d["n_singletons"]),
            int(d["n_distinct"]),
            None if ev is None else CountMinSketch.from_dict(ev),
        )


def _vhash(v: str) -> str:
    return xxhash.xxh64_hexdigest(v.encode("utf-8", "surrogatepass"))


class NoveltyDetector:
    name: str = NAME

    def __init__(
        self, store: DictStore, cfg: EngineConfig, priors: PriorSource | None = None
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._priors = priors

    def _record(self, key: str) -> NoveltyRecord | None:
        rec = self._store.get(key)
        return rec if isinstance(rec, NoveltyRecord) else None

    def _prior(self, rec: NoveltyRecord | None, tool: str, path: str) -> NoveltyRecord | None:
        """Pooled prior for a cold key (the 1.10 plug-in point); None when not applicable."""
        own_n = rec.n_total if rec is not None else 0
        if self._priors is None or own_n >= self._cfg.mag_cold_n:
            return None
        return self._priors.novelty_ref(tool, path)

    def _score_path(
        self,
        rec: NoveltyRecord | None,
        prior: NoveltyRecord | None,
        values: tuple[Scalar, ...],
        field_name: str,
        is_id: bool,
    ) -> DetectorScore | None:
        basis = prior if prior is not None else rec
        n_total = basis.n_total if basis is not None else 0
        n_single = basis.n_singletons if basis is not None else 0
        n_distinct = basis.n_distinct if basis is not None else 0
        if n_total > 0:
            s_unseen = (1.0 - n_single / n_total) * min(1.0, n_total / self._cfg.novelty_full_n)
        else:
            s_unseen = 0.0
        if prior is not None:
            s_unseen *= self._cfg.prior_novelty_discount
        best: tuple[float, str] | None = None
        for raw in values:
            if raw is None:
                continue
            v = str(raw)
            seen = (rec is not None and rec.is_seen(v)) or (prior is not None and prior.is_seen(v))
            s = 0.0 if seen else s_unseen
            if best is None or s > best[0]:
                best = (s, v)
        if best is None:
            return None
        s, v = best
        ev: dict[str, object] = {
            "p_unseen": s_unseen,
            "n_total": n_total,
            "n_distinct": n_distinct,
            "value_hash": _vhash(v),
            "prior_used": prior is not None,
        }
        if not is_id:
            ev["value"] = v
        return DetectorScore(NAME, field_name, s, ev)

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        out: list[DetectorScore] = []
        for path, values in iter_fields(e):
            ft = types(path)
            if ft not in _SCORED:
                continue
            rec = self._record(key_nov(e.agent_id, e.tool_id, path))
            prior = self._prior(rec, e.tool_id, path)
            ds = self._score_path(rec, prior, values, path, ft is FieldType.ID)
            if ds is not None:
                out.append(ds)
        return out

    def _push(self, rec: NoveltyRecord, v: str) -> None:
        rec.n_total += 1
        count = rec.seen.get(v)
        if count is not None:
            if count == 1:
                rec.n_singletons -= 1
            rec.seen[v] = count + 1
            rec.seen.move_to_end(v)
            return
        est = rec.evicted.estimate(v) if rec.evicted is not None else 0
        if est > 0:  # evicted earlier: already counted in n_distinct
            if est == 1:
                rec.n_singletons -= 1
            rec.seen[v] = est + 1
        else:
            rec.seen[v] = 1
            rec.n_singletons += 1
            rec.n_distinct += 1
        while len(rec.seen) > self._cfg.novelty_lru:
            old, c = rec.seen.popitem(last=False)
            if rec.evicted is None:
                rec.evicted = CountMinSketch(self._cfg.cms_width, self._cfg.cms_depth)
            rec.evicted.add(old, c)

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        for path, values in iter_fields(e):
            if types(path) not in _SCORED:
                continue
            key = key_nov(e.agent_id, e.tool_id, path)
            rec = self._record(key)
            for raw in values:
                if raw is None:
                    continue
                if rec is None:
                    rec = NoveltyRecord()
                    self._store.put(key, rec)
                self._push(rec, str(raw))
