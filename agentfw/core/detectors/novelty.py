"""Novelty detector: Good-Turing unseen-mass per (agent, tool, path) (spec §6.5).

Scored paths: CATEGORICAL and ID typed paths, plus the derived ``__domain__`` / ``__host__`` /
``__prefix__`` paths (categorical by construction). Values are keyed by
``base.value_key(v, freetext_min_len)``: ``str(v)``, hashed when it is that long or longer (so a
long derived host or a 400-digit int is never stored) or when ``str`` refuses a huge int.

A value is *seen* when it is in the record's LRU (``novelty_lru`` entries) or its count-min
estimate in the lazily allocated ``evicted`` sketch is > 0 (D-011). Seen scores 0; unseen scores
``(1 - n_singletons / n_total) * min(1, n_total / novelty_full_n)`` (``novelty_full_n`` = 50 by
user decision, decision 008; spec §6.5: 200); a path's score is the max
over its values. Evidence: ``p_unseen`` (= n_singletons / n_total of the record used),
``unseen_score`` (the unseen formula), ``n_total``, ``n_distinct``, ``value_hash`` (xxhash64
hex), ``prior_used``; the raw value is added only for non-ID types.

BOOL-typed paths (FR2 N1) are tracked too, but only a value that *contradicts* BOOL
(``schema_infer.contradicts_bool``: a number outside {0, 1}) is scored there: as an unseen value
against the field's observed boolean vocabulary, with the same formula and ramp, no prior, and
evidence ``type_violation="bool"``. Boolean-like values in a BOOL path produce no score. Without
this, a count field with only 0/1 history is typed BOOL and its first large value (scored before
the update that retypes the field) would be invisible to every detector.

Amendment A1.3 (``mag_bool_numeric``, wired by the engine as ``numeric_ref``): that score is only
a fallback. ``numeric_ref(agent, tool, path)`` is ``MagnitudeDetector.has_ref``: True when
magnitude has a usable reference for the field (a warm own record, which it now learns from the
BOOL-typed 0 / 1 numbers, or a warm pooled prior) and therefore scores the contradicting value
by its size; novelty then emits nothing for the path. It is asked only when a value contradicts
BOOL. Tracking of BOOL paths in ``update`` is the same either way. With ``numeric_ref=None``
(the switch off, or the engine without magnitude) the FR2 N1 score is always emitted.

Remaining limits with the switch on (pinned in ``tests/core/test_engine.py``): (1) the cold
window: while the field holds fewer than ``mag_cold_n`` learned numbers and no warm pooled prior
exists (the first observations, or a history of ``True`` / ``False`` only, which is never
pushed), magnitude is cold and the fallback scores the first number outside {0, 1} as unseen,
whatever its size (0.30 after ``[1]*25``, 1.0 after ``[True]*60``). (2) A constant history flags
the next integer: after ``[1]*60`` the field never varied, so magnitude's scale is the
``mag_eps`` floor and a ``2`` scores about 1.0 (z = 8.1), as for any constant numeric field; it
stays so on later ones while the first was BLOCKED, since a blocked call learns nothing. With the
switch off (pre-amendment) the first number outside {0, 1} scores by novelty alone, and after a
learned BOOL -> NUMERIC switch magnitude starts cold (it never pushed the BOOL-typed history).

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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import xxhash

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import PriorSource, iter_fields, value_key
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType, Scalar
from agentfw.core.schema_infer import TypeView, contradicts_bool
from agentfw.core.sketch.cms import CountMinSketch
from agentfw.core.state import StateStore, key_nov, register_record

__all__ = ["NAME", "NoveltyDetector", "NoveltyRecord"]

NAME = "novelty"
_SCORED = frozenset({FieldType.CATEGORICAL, FieldType.ID})
_TRACKED = _SCORED | {FieldType.BOOL}  # BOOL: only type-violating values are scored


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

    def push(self, v: str, cfg: EngineConfig) -> None:
        """Fold one value in: LRU + singleton bookkeeping, evicting to the sketch (D-011)."""
        self.n_total += 1
        count = self.seen.get(v)
        if count is not None:
            if count == 1:
                self.n_singletons = max(0, self.n_singletons - 1)
            self.seen[v] = count + 1
            self.seen.move_to_end(v)
            return
        est = self.evicted.estimate(v) if self.evicted is not None else 0
        if est > 0:  # evicted earlier: already counted in n_distinct
            if est == 1:
                self.n_singletons = max(0, self.n_singletons - 1)
            self.seen[v] = est + 1
        else:
            self.seen[v] = 1
            self.n_singletons += 1
            self.n_distinct += 1
        while len(self.seen) > cfg.novelty_lru:
            old, c = self.seen.popitem(last=False)
            if self.evicted is None:
                self.evicted = CountMinSketch(cfg.cms_width, cfg.cms_depth)
            self.evicted.add(old, c)

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
        self,
        store: StateStore,
        cfg: EngineConfig,
        priors: PriorSource | None = None,
        numeric_ref: Callable[[str, str, str], bool] | None = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._priors = priors
        # spec A1.3: ``numeric_ref(agent, tool, path)`` is True when magnitude can score a
        # BOOL-contradicting number of that field by its size; novelty then stays silent.
        self._numeric_ref = numeric_ref

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
        violation: str | None = None,
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
            v = value_key(raw, self._cfg.freetext_min_len)
            seen = (rec is not None and rec.is_seen(v)) or (prior is not None and prior.is_seen(v))
            s = 0.0 if seen else s_unseen
            if best is None or s > best[0]:
                best = (s, v)
        if best is None:
            return None
        s, v = best
        ev: dict[str, object] = {
            "p_unseen": n_single / n_total if n_total > 0 else 0.0,
            "unseen_score": s_unseen,
            "n_total": n_total,
            "n_distinct": n_distinct,
            "value_hash": _vhash(v),
            "prior_used": prior is not None,
        }
        if not is_id:
            ev["value"] = v
        if violation is not None:
            ev["type_violation"] = violation
        return DetectorScore(NAME, field_name, s, ev)

    def score(self, e: ActionEnvelope, types: TypeView) -> list[DetectorScore]:
        out: list[DetectorScore] = []
        numeric_ref = self._numeric_ref
        for path, values in iter_fields(e):
            ft = types(path)
            if ft is FieldType.BOOL:
                bad = tuple(v for v in values if v is not None and contradicts_bool(v))
                if bad and not (
                    numeric_ref is not None and numeric_ref(e.agent_id, e.tool_id, path)
                ):
                    rec = self._record(key_nov(e.agent_id, e.tool_id, path))
                    ds = self._score_path(rec, None, bad, path, False, violation="bool")
                    if ds is not None:
                        out.append(ds)
                continue
            if ft not in _SCORED:
                continue
            rec = self._record(key_nov(e.agent_id, e.tool_id, path))
            prior = self._prior(rec, e.tool_id, path)
            ds = self._score_path(rec, prior, values, path, ft is FieldType.ID)
            if ds is not None:
                out.append(ds)
        return out

    def update(self, e: ActionEnvelope, types: TypeView) -> None:
        for path, values in iter_fields(e):
            if types(path) not in _TRACKED:
                continue
            key = key_nov(e.agent_id, e.tool_id, path)
            rec = self._record(key)
            for raw in values:
                if raw is None:
                    continue
                if rec is None:
                    rec = NoveltyRecord()
                    self._store.put(key, rec)
                rec.push(value_key(raw, self._cfg.freetext_min_len), self._cfg)
