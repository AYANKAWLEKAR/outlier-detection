from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import orjson
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentfw.core.state import (
    DictStore,
    Record,
    key_cal,
    key_flow,
    key_kind,
    key_mag,
    key_nov,
    key_prereq,
    key_prior_mag,
    key_prior_nov,
    key_trans,
    key_type,
    key_vel,
    register_record,
    split_key,
)


@register_record
@dataclass(slots=True)
class Toy:
    KIND: ClassVar[str] = "toy"
    n: int = 0
    x: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"n": self.n, "x": self.x, "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(int(d["n"]), float(d["x"]), [str(t) for t in d["tags"]])


def _size(rec: Record) -> int:
    return len(orjson.dumps(rec.to_dict(), option=orjson.OPT_SORT_KEYS))


def _sum(store: DictStore) -> int:
    return sum(_size(r) for _, r in store.items())


toys = st.builds(
    Toy,
    n=st.integers(-(2**40), 2**40),
    x=st.floats(allow_nan=False, allow_infinity=False),
    tags=st.lists(st.text(max_size=8), max_size=4),
)
keys = st.text(alphabet="abc:%/ é", min_size=1, max_size=6)


def test_key_escaping_round_trips() -> None:
    k = key_type("agent:1", "tool%x", "a:b/c%3A")
    assert k.startswith("t:")
    assert k.count(":") == 3
    assert split_key(k) == ["t", "agent:1", "tool%x", "a:b/c%3A"]
    assert "%25" in k and "%3A" in k


def test_key_patterns() -> None:
    assert key_mag("a", "t", "p") == "m:a:t:p"
    assert key_nov("a", "t", "p") == "n:a:t:p"
    assert key_vel("a", "t") == "v:a:t"
    assert key_vel("a", "t", "p") == "v:a:t:p"
    assert key_trans("a") == "s:a:trans"
    assert key_prereq("a", "t") == "s:a:prereq:t"
    assert key_flow("a", "t", "p") == "f:a:t:p"
    assert key_cal("a") == "c:a"
    assert key_kind("a", "t") == "k:a:t"
    assert key_prior_mag("t", "p") == "p:m:t:p"
    assert key_prior_nov("t", "p") == "p:n:t:p"


def test_pooled_prior_keys_distinct() -> None:
    assert key_prior_mag("t", "p") != key_prior_nov("t", "p")


def test_duplicate_kind_registration_raises() -> None:
    with pytest.raises(ValueError, match="toy"):

        @register_record
        @dataclass(slots=True)
        class Other:
            KIND: ClassVar[str] = "toy"

            def to_dict(self) -> dict[str, object]:
                return {}

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> Self:
                return cls()


def test_get_put_items_clear() -> None:
    s = DictStore()
    assert s.get("a") is None
    r = Toy(1, 2.0, ["q"])
    s.put("a", r)
    assert s.get("a") is r
    assert dict(s.items()) == {"a": r}
    assert s.nbytes() > 0
    s.clear()
    assert s.get("a") is None
    assert s.nbytes() == 0


@given(st.lists(st.tuples(keys, toys), max_size=30))
def test_nbytes_equals_recomputed_sum(ops: list[tuple[str, Toy]]) -> None:
    s = DictStore()
    for k, rec in ops:
        s.put(k, rec)
        assert s.nbytes() == _sum(s)
    s.clear()
    assert s.nbytes() == 0


def test_nbytes_reflects_in_place_mutation() -> None:
    s = DictStore()
    r = Toy()
    s.put("k", r)
    before = s.nbytes()
    r.tags.extend(["alpha", "beta", "gamma"])
    r.n = 10**9
    assert s.nbytes() > before
    assert s.nbytes() == _sum(s)


@settings(deadline=None, max_examples=50)
@given(st.dictionaries(keys, toys, max_size=20))
def test_dump_load_round_trip(tmp_path_factory: pytest.TempPathFactory, d: dict[str, Toy]) -> None:
    path = tmp_path_factory.mktemp("snap") / "s.jsonl"
    s = DictStore()
    for k, v in d.items():
        s.put(k, v)
    s.dump(path)
    t = DictStore()
    t.put("stale", Toy(9))
    t.load(path)
    assert dict(t.items()) == dict(s.items()) == d
    assert t.nbytes() == s.nbytes()


def test_dump_is_deterministic_sorted(tmp_path: Path) -> None:
    a, b = DictStore(), DictStore()
    for k in ("z", "a", "m"):
        a.put(k, Toy(1, 0.5, ["x"]))
    for k in ("m", "z", "a"):
        b.put(k, Toy(1, 0.5, ["x"]))
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.dump(pa)
    b.dump(pb)
    assert pa.read_bytes() == pb.read_bytes()
    rows = [orjson.loads(line) for line in pa.read_bytes().splitlines()]
    assert [r["key"] for r in rows] == ["a", "m", "z"]
    assert set(rows[0]) == {"key", "kind", "data"} and rows[0]["kind"] == "toy"


def test_load_unknown_kind_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_bytes(orjson.dumps({"key": "k", "kind": "nope", "data": {}}) + b"\n")
    with pytest.raises(ValueError, match="nope"):
        DictStore().load(p)
