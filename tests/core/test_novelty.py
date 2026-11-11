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


# Spec §6.5 ramp min(1, n / 200); the default is 50 by user decision (decision 008).
SPEC_RAMP = EngineConfig().replace(novelty_full_n=200)


@pytest.fixture
def nov_spec() -> NoveltyDetector:
    return NoveltyDetector(DictStore(), SPEC_RAMP)


def test_half_warm_formula(nov_spec: NoveltyDetector) -> None:
    feed(nov_spec, "currency", ["usd", "eur"] * 50)  # n=100
    assert score(nov_spec, "currency", "jpy") == pytest.approx(0.5)


def test_default_ramp_is_50_observations(nov: NoveltyDetector) -> None:
    feed(nov, "currency", ["usd", "eur"] * 12)  # n=24
    assert score(nov, "currency", "jpy") == pytest.approx(24 / 50)
    feed(nov, "currency", ["usd", "eur"] * 13)  # n=50: fully warm
    assert score(nov, "currency", "jpy") == pytest.approx(1.0)


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
    for i in range(300):  # 300 distinct addresses, one domain
        nov.update(env({"to": [f"user{i}@acme.example"]}), CAT)
    known = {s.field: s for s in scores(nov, {"to": ["eve@acme.example"]})}
    assert known["to[]"].score < 0.05  # singleton-heavy address field: new address is normal
    assert known["to[].__domain__"].score == 0.0
    other = {s.field: s for s in scores(nov, {"to": ["eve@evil.example"]})}
    assert other["to[].__domain__"].score == pytest.approx(1.0)
    assert other["to[]"].score < 0.05


def test_p_unseen_is_good_turing_singleton_fraction(nov_spec: NoveltyDetector) -> None:
    feed(nov_spec, "currency", ["usd"] * 98 + ["eur", "gbp"])  # n=100, 2 singletons
    [s] = [s for s in scores(nov_spec, {"currency": "jpy"}) if s.field == "currency"]
    assert s.evidence["p_unseen"] == pytest.approx(0.02)
    assert s.evidence["unseen_score"] == pytest.approx(0.98 * 0.5)
    assert s.score == pytest.approx(0.49)


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


@pytest.mark.parametrize(("n", "prior_used"), [(29, True), (30, False)])
def test_prior_boundary_is_mag_cold_n(n: int, prior_used: bool) -> None:
    pooled = NoveltyRecord()
    for v in ("usd", "eur", "gbp"):
        pooled.seen[v] = 1000
    pooled.n_total, pooled.n_distinct = 3000, 3
    det = NoveltyDetector(DictStore(), SPEC_RAMP, priors=_Prior(pooled))
    feed(det, "currency", ["usd"] * n)
    [s] = [s for s in scores(det, {"currency": "jpy"}) if s.field == "currency"]
    assert s.evidence["prior_used"] is prior_used
    # own record: no singletons, n/full_n warm factor; prior: full warm x 0.67
    expected = 0.67 if prior_used else n / 200
    assert s.score == pytest.approx(expected)


def test_int_beyond_str_digit_limit_does_not_raise(nov: NoveltyDetector) -> None:
    """Review Focus 3 (found by the engine's hypothesis test): ``str(10**5000)`` raises."""
    huge, other = 10**5000, 10**5000 + 1
    feed(nov, "currency", ["usd"] * 30 + [huge])
    assert score(nov, "currency", huge) == 0.0  # seen: equal huge ints share one key
    assert score(nov, "currency", other) > 0.0  # a different huge int is unseen
    assert all(len(v) < 40 for v in nov._store.get(key_nov(AGENT, TOOL, "currency")).seen)  # type: ignore[union-attr]


# -- FR2 N1: a number outside {0, 1} in a BOOL-typed field is novel ------------------------------

BOOL = StubTypes({"qty": FieldType.BOOL})


def _feed_bool(det: NoveltyDetector, values: Iterable[Any]) -> None:
    for v in values:
        det.update(env({"qty": v}), BOOL)


def _bool_score(det: NoveltyDetector, v: Any) -> DetectorScore | None:
    out = [s for s in det.score(env({"qty": v}), BOOL) if s.field == "qty"]
    assert len(out) <= 1
    return out[0] if out else None


def test_bool_field_scores_a_contradicting_number_as_unseen(nov: NoveltyDetector) -> None:
    _feed_bool(nov, [0, 1] * 100)
    s = _bool_score(nov, 10000)
    assert s is not None and s.score == pytest.approx(1.0)  # stable {0,1}: no singletons
    assert s.evidence["type_violation"] == "bool"
    assert s.evidence["n_total"] == 200 and s.evidence["p_unseen"] == 0.0


def test_young_bool_field_is_discounted_by_the_ramp(nov: NoveltyDetector) -> None:
    _feed_bool(nov, [1] * 20)
    s = _bool_score(nov, 7)
    assert s is not None and s.score == pytest.approx(20 / 50)


@pytest.mark.parametrize("v", [True, False, 0, 1, "true", "0", float("nan"), None])
def test_bool_field_with_boolish_values_is_not_scored(nov: NoveltyDetector, v: Any) -> None:
    _feed_bool(nov, [True, False] * 100)
    assert _bool_score(nov, v) is None


def test_bool_violation_score_is_pure(nov: NoveltyDetector) -> None:
    _feed_bool(nov, [0, 1] * 100)
    before = orjson.dumps(nov._store.get(key_nov(AGENT, TOOL, "qty")).to_dict())  # type: ignore[union-attr]
    for _ in range(3):
        _bool_score(nov, 10000)
    assert orjson.dumps(nov._store.get(key_nov(AGENT, TOOL, "qty")).to_dict()) == before  # type: ignore[union-attr]
