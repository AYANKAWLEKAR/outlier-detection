"""Keyed state store, record registry and JSONL snapshots (spec §7).

Every detector keeps small per-key records in a :class:`StateStore`. Records are
mutable objects that detectors mutate in place after ``get``; ``put`` only stores
the reference (O(1), no serialization on the hot path).

``nbytes()`` is therefore measured on demand as the sum of
``len(orjson.dumps(rec.to_dict(), option=OPT_SORT_KEYS))`` over all records, so it is
always accurate even after in-place mutation. It is meant for reports, strict-mode
leak checks and ``state_bytes()``, not for the per-call path.

``dump`` writes deterministic JSONL (rows ``{"key","kind","data"}`` sorted by key,
sorted JSON keys). ``load`` *replaces* the store's contents and dispatches each row by
``kind`` through the registry; an unknown kind raises ``ValueError`` and leaves the
store unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self

import orjson

__all__ = [
    "DictStore",
    "Record",
    "StateStore",
    "key_cal",
    "key_flow",
    "key_kind",
    "key_mag",
    "key_nov",
    "key_prereq",
    "key_prior_mag",
    "key_prior_nov",
    "key_trans",
    "key_type",
    "key_vel",
    "register_record",
    "split_key",
]

_OPTS = orjson.OPT_SORT_KEYS


class Record(Protocol):
    KIND: ClassVar[str]

    def to_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self: ...


_REGISTRY: dict[str, type[Any]] = {}


def register_record[R: type[Any]](cls: R) -> R:
    """Register a record class under its ``KIND``; duplicate kinds (other classes) raise."""
    kind: str = cls.KIND
    existing = _REGISTRY.get(kind)
    if existing is not None and existing is not cls:
        raise ValueError(f"record kind {kind!r} already registered by {existing.__qualname__}")
    _REGISTRY[kind] = cls
    return cls


class StateStore(Protocol):
    def get(self, key: str) -> Record | None: ...
    def put(self, key: str, rec: Record) -> None: ...
    def items(self) -> Iterator[tuple[str, Record]]: ...
    def nbytes(self) -> int: ...
    def dump(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def clear(self) -> None: ...


class DictStore:
    """In-memory store backed by a dict."""

    def __init__(self) -> None:
        self._d: dict[str, Record] = {}

    def get(self, key: str) -> Record | None:
        return self._d.get(key)

    def put(self, key: str, rec: Record) -> None:
        self._d[key] = rec

    def items(self) -> Iterator[tuple[str, Record]]:
        return iter(list(self._d.items()))

    def nbytes(self) -> int:
        return sum(len(orjson.dumps(r.to_dict(), option=_OPTS)) for r in self._d.values())

    def dump(self, path: Path) -> None:
        lines = [
            orjson.dumps({"key": k, "kind": r.KIND, "data": r.to_dict()}, option=_OPTS)
            for k, r in sorted(self._d.items())
        ]
        Path(path).write_bytes(b"".join(line + b"\n" for line in lines))

    def load(self, path: Path) -> None:
        new: dict[str, Record] = {}
        for line in Path(path).read_bytes().splitlines():
            if not line.strip():
                continue
            row = orjson.loads(line)
            kind = row["kind"]
            cls = _REGISTRY.get(kind)
            if cls is None:
                raise ValueError(f"unknown record kind {kind!r} for key {row['key']!r}")
            new[row["key"]] = cls.from_dict(row["data"])
        self._d = new

    def clear(self) -> None:
        self._d = {}

    def __len__(self) -> int:
        return len(self._d)


def _esc(seg: str) -> str:
    return seg.replace("%", "%25").replace(":", "%3A")


def _unesc(seg: str) -> str:
    return seg.replace("%3A", ":").replace("%25", "%")


def _key(*segs: str) -> str:
    key = ":".join(segs)
    # Hot path (every scored field builds keys): segments almost never need escaping, so join
    # once and escape per segment only when a separator or escape char is present.
    if "%" not in key and key.count(":") == len(segs) - 1:
        return key
    return ":".join([_esc(s) for s in segs])


def split_key(key: str) -> list[str]:
    """Inverse of the key builders: split on ``:`` and unescape each segment."""
    return [_unesc(s) for s in key.split(":")]


def key_type(a: str, t: str, p: str) -> str:
    return "t:" + _key(a, t, p)


def key_mag(a: str, t: str, p: str) -> str:
    return "m:" + _key(a, t, p)


def key_vel(a: str, t: str, p: str | None = None) -> str:
    return "v:" + (_key(a, t) if p is None else _key(a, t, p))


def key_nov(a: str, t: str, p: str) -> str:
    return "n:" + _key(a, t, p)


def key_trans(a: str) -> str:
    return f"s:{_esc(a)}:trans"


def key_prereq(a: str, t: str) -> str:
    return f"s:{_esc(a)}:prereq:{_esc(t)}"


def key_flow(a: str, t: str, p: str) -> str:
    return "f:" + _key(a, t, p)


def key_cal(a: str) -> str:
    return "c:" + _esc(a)


def key_kind(a: str, t: str) -> str:
    return "k:" + _key(a, t)


def key_prior_mag(t: str, p: str) -> str:
    return "p:m:" + _key(t, p)


def key_prior_nov(t: str, p: str) -> str:
    return "p:n:" + _key(t, p)
