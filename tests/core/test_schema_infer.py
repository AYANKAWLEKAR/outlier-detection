from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import FieldType
from agentfw.core.schema_infer import TypeChange, TypeInference, TypeRecord, TypeView
from agentfw.core.state import DictStore, key_type
from agentfw.synth.sampler import stable_hash


@pytest.fixture
def store() -> DictStore:
    return DictStore()


@pytest.fixture
def ti(store: DictStore) -> TypeInference:
    return TypeInference(store, EngineConfig())


def test_unknown_until_ten_obs(ti: TypeInference) -> None:
    for i in range(9):
        assert ti.observe("a", "t", "amount", (float(i),)) is None
    assert ti.type_of("a", "t", "amount") is FieldType.UNKNOWN
    assert ti.observe("a", "t", "amount", (9.0,)) is None  # first assignment is not a change
    assert ti.type_of("a", "t", "amount") is FieldType.NUMERIC


def test_unseen_key_is_unknown(ti: TypeInference) -> None:
    assert ti.type_of("a", "t", "nope") is FieldType.UNKNOWN


def test_numeric_strings_infer_numeric(ti: TypeInference) -> None:
    for _ in range(20):
        ti.observe("a", "t", "amt", ("37.50",))
    assert ti.type_of("a", "t", "amt") is FieldType.NUMERIC


def test_numeric_tolerates_five_percent_junk(ti: TypeInference) -> None:
    for i in range(19):
        ti.observe("a", "t", "f", (f"{i}.5",))
    ti.observe("a", "t", "f", ("n/a",))
    assert ti.type_of("a", "t", "f") is FieldType.NUMERIC  # 95% parse


def test_huge_ints_and_inf_do_not_crash(ti: TypeInference) -> None:
    for i in range(12):
        ti.observe("a", "t", "f", (10**400 + i, float("inf"), float("nan")))
    assert ti.type_of("a", "t", "f") is FieldType.NUMERIC


def test_ids_emails_categoricals_bools(ti: TypeInference) -> None:
    for i in range(20):
        ti.observe("a", "t", "charge", (f"ch_3Nabc{i:08d}",))
        ti.observe("a", "t", "to", (f"user{i}@acme.example",))
        ti.observe("a", "t", "cur", (["usd", "eur", "gbp"][i % 3],))
        ti.observe("a", "t", "flag", (i % 2 == 0,))
    assert ti.type_of("a", "t", "charge") is FieldType.ID
    assert ti.type_of("a", "t", "to") is FieldType.ID
    assert ti.type_of("a", "t", "cur") is FieldType.CATEGORICAL
    assert ti.type_of("a", "t", "flag") is FieldType.BOOL


@pytest.mark.parametrize(
    "vals",
    [
        [0, 1],
        ["true", "false"],
        ["TRUE", " False "],
        [True, 0, "1", "false"],
    ],
)
def test_bool_value_forms(ti: TypeInference, vals: list[object]) -> None:
    for i in range(20):
        ti.observe("a", "t", "f", (vals[i % len(vals)],))  # type: ignore[arg-type]
    assert ti.type_of("a", "t", "f") is FieldType.BOOL


def test_bool_requires_all_values_in_set(ti: TypeInference) -> None:
    for i in range(20):
        ti.observe("a", "t", "f", (i % 3,))  # 0, 1, 2
    assert ti.type_of("a", "t", "f") is FieldType.NUMERIC


def test_id_by_cardinality(ti: TypeInference) -> None:
    # Assigned at n=10 (too few for the n>=50 cardinality rule -> categorical), then sticky:
    # the full 100-observation window re-evaluates to id.
    for i in range(99):
        ti.observe("a", "t", "ref", (f"zz{i * 7919 % 100003}",))  # not id-like, all distinct
    assert ti.type_of("a", "t", "ref") is FieldType.CATEGORICAL
    ch = ti.observe("a", "t", "ref", ("zz-last",))
    assert ch is not None and ch.new is FieldType.ID


def test_freetext(ti: TypeInference) -> None:
    for i in range(20):
        ti.observe("a", "t", "body", (f"this is a long free text message number {i} " * 2,))
    assert ti.type_of("a", "t", "body") is FieldType.FREETEXT


def test_none_values_are_skipped(ti: TypeInference) -> None:
    for _ in range(30):
        ti.observe("a", "t", "f", (None,))
    assert ti.type_of("a", "t", "f") is FieldType.UNKNOWN


def test_type_is_sticky_then_changes(ti: TypeInference) -> None:
    for i in range(200):
        ti.observe("a", "t", "f", (float(i),))
    assert ti.type_of("a", "t", "f") is FieldType.NUMERIC
    for _ in range(99):
        assert ti.observe("a", "t", "f", ("x",)) is None
        assert ti.type_of("a", "t", "f") is FieldType.NUMERIC
    ch = ti.observe("a", "t", "f", ("x",))
    assert isinstance(ch, TypeChange)
    assert ch.new is FieldType.CATEGORICAL
    assert ch.old is FieldType.NUMERIC
    assert ch.key == key_type("a", "t", "f")
    assert ch.n_total == 300
    assert ti.type_of("a", "t", "f") is FieldType.CATEGORICAL
    rec = ti.record("a", "t", "f")
    assert rec is not None and rec.changes == 1


def test_per_agent_independent_and_type_of_is_pure(
    ti: TypeInference, store: DictStore, tmp_path: Path
) -> None:
    for i in range(30):
        ti.observe("a1", "t", "f", (["usd", "eur"][i % 2],))
        ti.observe("a2", "t", "f", (f"ch_3Nabc{i:08d}",))
    assert ti.type_of("a1", "t", "f") is FieldType.CATEGORICAL
    assert ti.type_of("a2", "t", "f") is FieldType.ID
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    store.dump(p1)
    before = store.nbytes()
    for _ in range(5):
        ti.type_of("a1", "t", "f")
        ti.type_of("a1", "t", "missing")
    assert store.nbytes() == before
    store.dump(p2)
    assert p1.read_bytes() == p2.read_bytes()
    assert store.get(key_type("a1", "t", "missing")) is None


def test_reservoir_bounded_deterministic_and_seeded(store: DictStore) -> None:
    def run() -> TypeRecord:
        s = DictStore()
        inf = TypeInference(s, EngineConfig())
        for i in range(500):
            inf.observe("a", "t", "f", (f"v{i}",))
        rec = inf.record("a", "t", "f")
        assert rec is not None
        return rec

    r1, r2 = run(), run()
    assert len(r1.reservoir) == 64
    assert r1.reservoir == r2.reservoir
    assert set(r1.reservoir) <= {f"v{i}" for i in range(500)}
    assert r1.seed == stable_hash(key_type("a", "t", "f"))
    # Algorithm R: not just the first 64
    assert r1.reservoir != [f"v{i}" for i in range(64)]


def test_distinct_capped_with_overflow(ti: TypeInference) -> None:
    for i in range(1500):
        ti.observe("a", "t", "f", (f"v{i}",))
    rec = ti.record("a", "t", "f")
    assert rec is not None
    assert len(rec.distinct) == 1024
    assert rec.distinct_overflow == 476
    assert ti.type_of("a", "t", "f") is FieldType.ID  # high cardinality still detected


def test_window_bounded(ti: TypeInference) -> None:
    for i in range(250):
        ti.observe("a", "t", "f", (float(i),))
    rec = ti.record("a", "t", "f")
    assert rec is not None
    assert len(rec.window) == 100
    assert rec.n_total == 250


def test_record_roundtrip_finite_json(ti: TypeInference, store: DictStore, tmp_path: Path) -> None:
    for i in range(120):
        ti.observe("a", "t", "f", (float(i), "x", float("nan"), float("inf")))
    p = tmp_path / "s.jsonl"
    store.dump(p)
    assert b"NaN" not in p.read_bytes() and b"null" not in p.read_bytes()
    s2 = DictStore()
    s2.load(p)
    p2 = tmp_path / "s2.jsonl"
    s2.dump(p2)
    assert p.read_bytes() == p2.read_bytes()
    ti2 = TypeInference(s2, EngineConfig())
    assert ti2.type_of("a", "t", "f") is ti.type_of("a", "t", "f")
    for row in p.read_bytes().splitlines():
        assert orjson.loads(row)["kind"] == "type"
    # continued observation after load behaves identically
    ti.observe("a", "t", "f", ("zzz",))
    ti2.observe("a", "t", "f", ("zzz",))
    assert ti.record("a", "t", "f") == ti2.record("a", "t", "f")


def test_type_view(ti: TypeInference) -> None:
    for i in range(20):
        ti.observe("a", "t", "amount", (float(i) + 0.5,))
    view = TypeView(ti, "a", "t")
    assert view("amount") is FieldType.NUMERIC
    assert view("nope") is FieldType.UNKNOWN
    for suffix in ("__len__", "__strlen__", "__truncated__"):
        assert view(f"x.{suffix}") is FieldType.NUMERIC
    for suffix in ("__domain__", "__host__", "__prefix__"):
        assert view(f"to[].{suffix}") is FieldType.CATEGORICAL


def test_freetext_content_never_stored(ti: TypeInference) -> None:
    text = "secret customer message that is definitely longer than forty chars"
    for i in range(30):
        ti.observe("a", "t", "body", (f"{text} {i}",))
    rec = ti.record("a", "t", "body")
    assert rec is not None
    blob = orjson.dumps(rec.to_dict())
    assert b"secret" not in blob


def test_huge_int_observe_does_not_raise(ti: TypeInference) -> None:
    for _ in range(12):
        ti.observe("a", "t", "g", (10**5000, -(10**5000)))
    assert ti.type_of("a", "t", "g") is FieldType.NUMERIC


def test_huge_int_end_to_end_through_normalizer(ti: TypeInference) -> None:
    from agentfw.core.normalize import SessionTracker, normalize
    from tests.factories import make_raw

    for _ in range(12):
        env = normalize(make_raw(args={"n": 10**5000, "l": [10**5000]}), SessionTracker())
        for path, vals in {**env.flat_args, **env.derived}.items():
            ti.observe("a", "t", path, vals)
    assert ti.type_of("a", "t", "n") is FieldType.NUMERIC


# -- FR I3: a count field that starts as 0/1 must not stay BOOL ------------------------------------


def test_bool_field_switches_to_numeric_on_first_value_outside_0_1(ti: TypeInference) -> None:
    for _ in range(50):
        ti.observe("a", "t", "qty", (1,))
    assert ti.type_of("a", "t", "qty") is FieldType.BOOL
    ch = ti.observe("a", "t", "qty", (3,))  # immediately, not after a 100-observation window
    assert isinstance(ch, TypeChange) and (ch.old, ch.new) == (FieldType.BOOL, FieldType.NUMERIC)
    assert ti.type_of("a", "t", "qty") is FieldType.NUMERIC
    for _ in range(300):  # and it stays numeric under 0/1 traffic (numeric residual)
        ti.observe("a", "t", "qty", (1,))
    assert ti.type_of("a", "t", "qty") is FieldType.NUMERIC


@pytest.mark.parametrize("odd", [2.5, -1, "4", 10**400])
def test_bool_switch_covers_floats_negatives_numeric_strings_huge_ints(
    ti: TypeInference, odd: object
) -> None:
    for i in range(20):
        ti.observe("a", "t", "f", (["0", "1"][i % 2],))
    assert ti.type_of("a", "t", "f") is FieldType.BOOL
    ti.observe("a", "t", "f", (odd,))  # type: ignore[arg-type]
    assert ti.type_of("a", "t", "f") is FieldType.NUMERIC


def test_genuine_bool_field_stays_bool(ti: TypeInference) -> None:
    forms: list[object] = [True, False, "true", "False", 0, 1, "0", "1"]
    for i in range(400):
        ti.observe("a", "t", "flag", (forms[i % len(forms)],))  # type: ignore[arg-type]
    for v in (float("nan"), float("inf"), "nan"):  # non-finite: no evidence of a count
        ti.observe("a", "t", "flag", (v,))
    assert ti.type_of("a", "t", "flag") is FieldType.BOOL
    rec = ti.record("a", "t", "flag")
    assert rec is not None and rec.changes == 0
