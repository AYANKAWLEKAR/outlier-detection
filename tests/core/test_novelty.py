from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import orjson
import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.novelty import NoveltyDetector, NoveltyRecord
from agentfw.core.envelope import ActionEnvelope, DetectorScore, FieldType
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_nov
from tests.factories import make_env

AGENT = "a1"
TOOL = "stripe.create_refund"


class StubTypes(TypeView):
    """TypeView stand-in: explicit types, derived categorical paths by default."""

    def __init__(self, types: Mapping[str, FieldType]) -> None:
        super().__init__(TypeInference(DictStore(), EngineConfig()), AGENT, TOOL)
        self._t = dict(types)

    def __call__(self, path: str) -> FieldType:
        if path in self._t:
            return self._t[path]
        if path.endswith(("__domain__", "__host__", "__prefix__")):
            return FieldType.CATEGORICAL
        return FieldType.UNKNOWN


CAT = StubTypes(
    {
        "currency": FieldType.CATEGORICAL,
        "to[]": FieldType.ID,
        "cid": FieldType.ID,
        "ccy[]": FieldType.CATEGORICAL,
    }
)


@pytest.fixture
def nov() -> NoveltyDetector:
    return NoveltyDetector(DictStore(), EngineConfig())


def env(args: dict[str, Any]) -> ActionEnvelope:
    return make_env(agent_id=AGENT, tool_id=TOOL, args=args)


def feed(det: NoveltyDetector, path: str, values: Iterable[Any]) -> None:
    for v in values:
        det.update(env({path: v}), CAT)


def scores(det: NoveltyDetector, args: dict[str, Any]) -> list[DetectorScore]:
    return det.score(env(args), CAT)


def score(det: NoveltyDetector, path: str, v: Any) -> float:
    out = [s for s in scores(det, {path: v}) if s.field == path]
    assert len(out) == 1
    return out[0].score


def test_second_sighting_is_not_novel(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["usd", "eur", "gbp"] * 100)
    assert score(nov, "currency", "usd") == 0.0
    feed(nov, "currency", ["jpy"])
    assert score(nov, "currency", "jpy") == 0.0


def test_stable_vocab_exact_formula(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["usd", "eur", "gbp"] * 100)  # n=300, no singletons
    assert score(nov, "currency", "jpy") == pytest.approx(1.0)


def test_half_warm_formula(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["usd", "eur"] * 50)  # n=100
    assert score(nov, "currency", "jpy") == pytest.approx(0.5)


def test_cold_key_scores_zero_without_division_error(nov: NoveltyDetector) -> None:
    assert score(nov, "currency", "jpy") == 0.0


def test_singleton_heavy_field_scores_low(nov: NoveltyDetector) -> None:
    feed(nov, "cid", [f"cus_{i}" for i in range(300)])
    assert score(nov, "cid", "cus_new") < 0.05


def test_singleton_decrements_on_second_sighting(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["usd"] * 50 + ["eur", "gbp"] + ["usd"] * 148)
    rec = nov._store.get(key_nov(AGENT, TOOL, "currency"))
    assert isinstance(rec, NoveltyRecord)
    assert (rec.n_total, rec.n_singletons, rec.n_distinct) == (200, 2, 3)
    feed(nov, "currency", ["eur"])
    assert (rec.n_total, rec.n_singletons, rec.n_distinct) == (201, 1, 3)


def test_new_address_known_domain_vs_new_domain(nov: NoveltyDetector) -> None:
    people = ["ann", "bob", "cy", "di"]
    for i in range(300):
        nov.update(env({"to": [f"{people[i % 4]}@acme.example"]}), CAT)
    known = {s.field: s for s in scores(nov, {"to": ["eve@acme.example"]})}
    # new address on a known domain: the address is a singleton-free vocab, so it is "novel" by
    # the formula only via to[]; the domain is not novel at all.
    assert known["to[].__domain__"].score == 0.0
    other = {s.field: s for s in scores(nov, {"to": ["eve@evil.example"]})}
    assert other["to[].__domain__"].score == pytest.approx(1.0)
    assert other["to[].__domain__"].score > known["to[].__domain__"].score


def test_case_variant_domain_not_novel(nov: NoveltyDetector) -> None:
    for _ in range(300):
        nov.update(env({"to": ["bob@acme.example"]}), CAT)
    out = {s.field: s for s in scores(nov, {"to": ["Bob@ACME.example"]})}
    assert out["to[].__domain__"].score == 0.0


def test_lru_bound_and_evicted_still_seen(nov: NoveltyDetector) -> None:
    feed(nov, "currency", [f"v{i}" for i in range(20_000)])
    rec = nov._store.get(key_nov(AGENT, TOOL, "currency"))
    assert isinstance(rec, NoveltyRecord)
    assert len(rec.seen) <= 10_000
    assert rec.evicted is not None
    assert "v0" not in rec.seen  # evicted from the LRU ...
    assert score(nov, "currency", "v0") == 0.0  # ... yet still seen via the CMS
    assert score(nov, "currency", "never-seen") >= 0.0


def test_evicted_value_reinserted_with_estimate_plus_one() -> None:
    cfg = EngineConfig(novelty_lru=2)
    det = NoveltyDetector(DictStore(), cfg)
    feed(det, "currency", ["a", "b", "c"])  # "a" evicted (count 1 -> singleton)
    rec = det._store.get(key_nov(AGENT, TOOL, "currency"))
    assert isinstance(rec, NoveltyRecord)
    assert rec.evicted is not None and "a" not in rec.seen
    assert (rec.n_singletons, rec.n_distinct) == (3, 3)
    feed(det, "currency", ["a"])
    assert rec.seen["a"] == 2
    assert (rec.n_total, rec.n_singletons, rec.n_distinct) == (4, 2, 3)
    assert len(rec.seen) <= 2


def test_cms_not_allocated_until_first_eviction(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["a", "b"])
    rec = nov._store.get(key_nov(AGENT, TOOL, "currency"))
    assert isinstance(rec, NoveltyRecord) and rec.evicted is None


def test_score_is_pure(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["usd", "eur", "gbp"] * 100)
    rec = nov._store.get(key_nov(AGENT, TOOL, "currency"))
    assert isinstance(rec, NoveltyRecord)
    before = (list(rec.seen.items()), rec.n_total, rec.n_singletons, rec.n_distinct)
    for v in ("usd", "jpy", "eur"):
        score(nov, "currency", v)
    assert (list(rec.seen.items()), rec.n_total, rec.n_singletons, rec.n_distinct) == before
    assert nov._store.get(key_nov(AGENT, TOOL, "currency")) is rec
    assert nov._store.get(key_nov(AGENT, TOOL, "other")) is None


def test_evidence_and_id_never_leaks_raw_value(nov: NoveltyDetector) -> None:
    feed(nov, "cid", [f"cus_{i}" for i in range(300)])
    [s] = [s for s in scores(nov, {"cid": "cus_secret"}) if s.field == "cid"]
    assert s.detector == "novelty"
    assert set(s.evidence) >= {"p_unseen", "n_total", "n_distinct", "value_hash"}
    assert "cus_secret" not in repr(s.evidence)
    assert s.evidence["n_total"] == 300 and s.evidence["n_distinct"] == 300
    feed(nov, "currency", ["usd"] * 5)
    [c] = [s for s in scores(nov, {"currency": "usd"}) if s.field == "currency"]
    assert len(str(c.evidence["value_hash"])) == 16


def test_untyped_paths_and_numeric_are_skipped(nov: NoveltyDetector) -> None:
    assert scores(nov, {"amount": 5, "memo": "hi"}) == []


def test_list_values_take_max(nov: NoveltyDetector) -> None:
    for _ in range(100):
        nov.update(env({"ccy": ["usd", "eur", "gbp"]}), CAT)
    e = env({"ccy": ["usd", "jpy"]})
    [s] = [s for s in nov.score(e, CAT) if s.field == "ccy[]"]
    assert s.score == pytest.approx(1.0)


def test_record_roundtrip_preserves_lru_order_and_cms() -> None:
    cfg = EngineConfig(novelty_lru=3)
    det = NoveltyDetector(DictStore(), cfg)
    feed(det, "currency", ["a", "b", "c", "d", "b", "e"])
    rec = det._store.get(key_nov(AGENT, TOOL, "currency"))
    assert isinstance(rec, NoveltyRecord) and rec.evicted is not None
    d = rec.to_dict()
    raw = orjson.dumps(d, option=orjson.OPT_SORT_KEYS)
    back = NoveltyRecord.from_dict(orjson.loads(raw))
    assert list(back.seen.items()) == list(rec.seen.items())
    assert orjson.dumps(back.to_dict(), option=orjson.OPT_SORT_KEYS) == raw
    assert back.evicted is not None and back.evicted.estimate("a") == rec.evicted.estimate("a")


def test_record_without_cms_roundtrips() -> None:
    rec = NoveltyRecord()
    rec.seen["x"] = 1
    back = NoveltyRecord.from_dict(orjson.loads(orjson.dumps(rec.to_dict())))
    assert back.evicted is None and list(back.seen.items()) == [("x", 1)]


class _Prior:
    def __init__(self, rec: NoveltyRecord | None) -> None:
        self.rec = rec

    def novelty_ref(self, tool: str, path: str) -> NoveltyRecord | None:
        return self.rec

    def magnitude_ref(self, tool: str, path: str, direction: str) -> None:
        return None


def test_cold_key_uses_pooled_prior_with_discount() -> None:
    pooled = NoveltyRecord()
    for v in ("usd", "eur", "gbp"):
        pooled.seen[v] = 100
    pooled.n_total, pooled.n_distinct = 300, 3
    det = NoveltyDetector(DictStore(), EngineConfig(), priors=_Prior(pooled))
    [s] = [s for s in scores(det, {"currency": "jpy"}) if s.field == "currency"]
    assert s.score == pytest.approx(0.67)
    assert s.evidence["prior_used"] is True
    assert score(det, "currency", "usd") == 0.0  # known to the pool


def test_no_prior_matches_plain_formula(nov: NoveltyDetector) -> None:
    [s] = [s for s in scores(nov, {"currency": "jpy"}) if s.field == "currency"]
    assert s.evidence["prior_used"] is False
