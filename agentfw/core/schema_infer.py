"""Sticky per-agent field type inference (spec §6.2).

State per ``(agent, tool, path)`` is a :class:`TypeRecord`: cumulative counts, a bounded set of
distinct values, a reservoir sample and a window of the last ``type_sticky_window``
observations.

Rules (first match wins) over counts ``n`` (cumulative for the first assignment, the window
for changes):

* bool: every value is a boolean-like: Python ``bool``, int/float ``0``/``1``, or a string that
  stripped and lowercased is one of ``true``, ``false``, ``0``, ``1``.
* numeric: >= 95% parse via ``try_number`` and are not id-like (numeric strings count).
* id: >= 90% id-like, or (``n >= 50``, distinct/n >= 0.9 and mean length <= 40).
* freetext: mean length > 40 and distinct/n > 0.9.
* categorical: otherwise.

Lifecycle: the field is ``UNKNOWN`` until ``n_total >= type_min_obs``; the first assignment uses
the cumulative counts and is *not* reported as a change. Afterwards a change requires a full
window whose rule result differs from the current type *and* that holds no residual evidence for
the old type (no numeric flag if it was numeric, no bool flag if bool, no id-like value if id).
This makes the new rule hold over the last 100 observations on their own, so a handful of odd
values never flips a type.

``None`` values carry no information and are skipped. Strings of ``freetext_min_len`` or more
characters are never stored: the distinct set, reservoir and window keep ``#<xxh64 hex>``
instead (their true length still feeds ``sum_len``). The reservoir RNG is stateless (splitmix64
of ``(seed, n_total)``, ``seed = xxh64(key)``, the same value as ``synth.sampler.stable_hash``),
so it needs no persisted generator state and ``type_of`` stays pure. Distinct cardinality beyond
the 1024 cap is estimated as ``len(distinct) + distinct_overflow`` (an upper bound)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import xxhash

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import FieldType, Scalar
from agentfw.core.patterns import is_id_like, try_number
from agentfw.core.state import DictStore, key_type, register_record

__all__ = ["TypeChange", "TypeInference", "TypeRecord", "TypeView"]

_DISTINCT_CAP = 1024
_BOOL_STRINGS = frozenset({"true", "false", "0", "1"})
_NUMERIC_FRAC = 0.95
_ID_FRAC = 0.9
_ID_CARD_MIN_N = 50
_ID_CARD_RATIO = 0.9
_ID_CARD_MAX_MEAN_LEN = 40
_FREETEXT_MEAN_LEN = 40
_FREETEXT_RATIO = 0.9
_MASK = 0xFFFFFFFFFFFFFFFF

_NUMERIC_SUFFIXES = ("__len__", "__strlen__", "__truncated__")
_CATEGORICAL_SUFFIXES = ("__domain__", "__host__", "__prefix__")

Obs = tuple[bool, bool, bool, int, str]  # (is_numeric, is_bool, is_idlike, length, stored_str)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


@register_record
@dataclass(slots=True)
class TypeRecord:
    KIND: ClassVar[str] = "type"
    seed: int = 0
    n_numeric: int = 0
    n_bool: int = 0
    n_idlike: int = 0
    n_total: int = 0
    sum_len: int = 0
    distinct: set[str] = field(default_factory=set)
    distinct_overflow: int = 0
    reservoir: list[str] = field(default_factory=list)
    window: deque[Obs] = field(default_factory=lambda: deque(maxlen=100))
    type: FieldType = FieldType.UNKNOWN
    confidence: float = 0.0
    changes: int = 0
    reservoir_size: int = 64

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_numeric": self.n_numeric,
            "n_bool": self.n_bool,
            "n_idlike": self.n_idlike,
            "n_total": self.n_total,
            "sum_len": self.sum_len,
            "distinct": sorted(self.distinct),
            "distinct_overflow": self.distinct_overflow,
            "reservoir": list(self.reservoir),
            "window": [list(o) for o in self.window],
            "window_max": self.window.maxlen,
            "type": self.type.value,
            "confidence": self.confidence,
            "changes": self.changes,
            "reservoir_size": self.reservoir_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            seed=d["seed"],
            n_numeric=d["n_numeric"],
            n_bool=d["n_bool"],
            n_idlike=d["n_idlike"],
            n_total=d["n_total"],
            sum_len=d["sum_len"],
            distinct=set(d["distinct"]),
            distinct_overflow=d["distinct_overflow"],
            reservoir=list(d["reservoir"]),
            window=deque((tuple(o) for o in d["window"]), maxlen=d["window_max"]),
            type=FieldType(d["type"]),
            confidence=d["confidence"],
            changes=d["changes"],
            reservoir_size=d["reservoir_size"],
        )


@dataclass(slots=True)
class TypeChange:
    key: str
    old: FieldType
    new: FieldType
    n_total: int


def _is_boolish(v: Scalar) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, int | float):
        return v == 0 or v == 1
    return isinstance(v, str) and v.strip().lower() in _BOOL_STRINGS


def _render(v: Scalar) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return v if isinstance(v, str) else repr(v)


def _classify(
    n: int, n_num: int, n_bool: int, n_id: int, sum_len: int, n_distinct: int
) -> tuple[FieldType, float]:
    """Apply the §6.2 rules to counts; returns ``(type, confidence)``."""
    if n_bool == n:
        return FieldType.BOOL, 1.0
    if n_num >= _NUMERIC_FRAC * n:
        return FieldType.NUMERIC, n_num / n
    ratio = min(1.0, n_distinct / n)
    mean_len = sum_len / n
    if n_id >= _ID_FRAC * n:
        return FieldType.ID, n_id / n
    if n >= _ID_CARD_MIN_N and ratio >= _ID_CARD_RATIO and mean_len <= _ID_CARD_MAX_MEAN_LEN:
        return FieldType.ID, ratio
    if mean_len > _FREETEXT_MEAN_LEN and ratio > _FREETEXT_RATIO:
        return FieldType.FREETEXT, ratio
    return FieldType.CATEGORICAL, 1.0 - ratio


class TypeInference:
    def __init__(self, store: DictStore, cfg: EngineConfig) -> None:
        self._store = store
        self._cfg = cfg

    def _get(self, key: str) -> TypeRecord | None:
        rec = self._store.get(key)
        return rec if isinstance(rec, TypeRecord) else None

    def record(self, agent: str, tool: str, path: str) -> TypeRecord | None:
        return self._get(key_type(agent, tool, path))

    def type_of(self, agent: str, tool: str, path: str) -> FieldType:
        """Current type; pure (reads only, never creates or mutates state)."""
        rec = self._get(key_type(agent, tool, path))
        return FieldType.UNKNOWN if rec is None else rec.type

    def observe(
        self, agent: str, tool: str, path: str, values: tuple[Scalar, ...]
    ) -> TypeChange | None:
        """Record observations; returns the last real type change (first assignment is not one)."""
        key = key_type(agent, tool, path)
        rec = self._get(key)
        change: TypeChange | None = None
        for v in values:
            if v is None:
                continue
            if rec is None:
                rec = TypeRecord(
                    seed=xxhash.xxh64_intdigest(key.encode()),
                    window=deque(maxlen=self._cfg.type_sticky_window),
                    reservoir_size=self._cfg.reservoir_size,
                )
                self._store.put(key, rec)
            ch = self._push(key, rec, v)
            if ch is not None:
                change = ch
        return change

    def _push(self, key: str, rec: TypeRecord, v: Scalar) -> TypeChange | None:
        s = _render(v)
        length = len(s)
        idlike = isinstance(v, str) and is_id_like(v)
        num = try_number(v) is not None and not idlike
        boolish = _is_boolish(v)
        stored = (
            s if length < self._cfg.freetext_min_len else f"#{xxhash.xxh64_hexdigest(s.encode())}"
        )

        rec.n_total += 1
        rec.n_numeric += num
        rec.n_bool += boolish
        rec.n_idlike += idlike
        rec.sum_len += length
        if stored in rec.distinct:
            pass
        elif len(rec.distinct) < _DISTINCT_CAP:
            rec.distinct.add(stored)
        else:
            rec.distinct_overflow += 1
        if len(rec.reservoir) < rec.reservoir_size:
            rec.reservoir.append(stored)
        else:
            j = _splitmix64(rec.seed ^ (rec.n_total * 0x9E3779B97F4A7C15 & _MASK)) % rec.n_total
            if j < rec.reservoir_size:
                rec.reservoir[j] = stored
        rec.window.append((num, boolish, idlike, length, stored))

        if rec.type is FieldType.UNKNOWN:
            if rec.n_total >= self._cfg.type_min_obs:
                n_distinct = min(rec.n_total, len(rec.distinct) + rec.distinct_overflow)
                rec.type, rec.confidence = _classify(
                    rec.n_total, rec.n_numeric, rec.n_bool, rec.n_idlike, rec.sum_len, n_distinct
                )
            return None
        return self._maybe_change(key, rec)

    def _maybe_change(self, key: str, rec: TypeRecord) -> TypeChange | None:
        w = rec.window
        if w.maxlen is None or len(w) < w.maxlen:
            return None
        n = len(w)
        n_num = sum(o[0] for o in w)
        n_bool = sum(o[1] for o in w)
        n_id = sum(o[2] for o in w)
        new, conf = _classify(n, n_num, n_bool, n_id, sum(o[3] for o in w), len({o[4] for o in w}))
        old = rec.type
        if new is old:
            rec.confidence = conf
            return None
        residual = {FieldType.NUMERIC: n_num, FieldType.BOOL: n_bool, FieldType.ID: n_id}.get(
            old, 0
        )
        if residual > 0:
            return None
        rec.type, rec.confidence = new, conf
        rec.changes += 1
        return TypeChange(key=key, old=old, new=new, n_total=rec.n_total)


class TypeView:
    """Callable ``path -> FieldType`` bound to one ``(agent, tool)``.

    Derived paths are typed by construction: ``__len__``/``__strlen__``/``__truncated__`` are
    numeric; ``__domain__``/``__host__``/``__prefix__`` are categorical.
    """

    def __init__(self, inference: TypeInference, agent: str, tool: str) -> None:
        self._inference = inference
        self._agent = agent
        self._tool = tool

    def __call__(self, path: str) -> FieldType:
        last = path.rsplit(".", 1)[-1]
        if last in _NUMERIC_SUFFIXES:
            return FieldType.NUMERIC
        if last in _CATEGORICAL_SUFFIXES:
            return FieldType.CATEGORICAL
        return self._inference.type_of(self._agent, self._tool, path)
