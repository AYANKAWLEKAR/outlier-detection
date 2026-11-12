from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentfw.bench.sweeps import PRE_A1_SWITCHES
from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import iter_fields, normal_sf, posterior, signed_log1p, squash
from agentfw.core.detectors.magnitude import MagnitudeDetector, MagRecord, direction_for
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_mag
from tests.factories import make_env

AGENT = "a1"
TOOL = "stripe.create_refund"
DEFAULT = EngineConfig()
Z = EngineConfig().replace(mag_score="z")  # the §6.3 rule, whatever the default is
TAIL = EngineConfig().replace(mag_score="tail")  # Amendment A1.2


@dataclass
class Trained:
    det: MagnitudeDetector
    ti: TypeInference
    store: DictStore

    def view(self, tool: str = TOOL) -> TypeView:
        return TypeView(self.ti, AGENT, tool)


def _learn(t: Trained, e: ActionEnvelope) -> None:
    """Engine order: infer types, then update the detector."""
    for path, vals in e.flat_args.items():
        t.ti.observe(e.agent_id, e.tool_id, path, vals)
    t.det.update(e, t.view(e.tool_id))


def build(
    values: Iterable[Any], path: str = "amount", cfg: EngineConfig = DEFAULT, **det_kw: Any
) -> Trained:
    store = DictStore()
    t = Trained(MagnitudeDetector(store, cfg, **det_kw), TypeInference(store, cfg), store)
    for v in values:
        _learn(t, make_env(agent_id=AGENT, tool_id=TOOL, args={path: v}))
    return t


def scores_for(t: Trained, args: dict[str, Any]) -> list[DetectorScore]:
    e = make_env(agent_id=AGENT, tool_id=TOOL, args=args)
    return t.det.score(e, t.view())


def score_field(t: Trained, path: str, x: Any) -> float:
    found = [s for s in scores_for(t, {path: x}) if s.field == path]
    assert len(found) == 1
    return found[0].score


def score_amount(t: Trained, x: float) -> float:
    return score_field(t, "amount", x)


def record(t: Trained, path: str) -> MagRecord:
    rec = t.store.get(key_mag(AGENT, TOOL, path))
    assert isinstance(rec, MagRecord)
    return rec


def _lognormal(n: int = 500, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.lognormal(3.6, 0.7, n)]


@pytest.fixture(scope="module")
def trained_mag() -> Trained:
    """Default config. Since Task 1.21 that is ``mag_score="tail"`` (A1.2), so the tests on this
    fixture (warm spike, warm evidence, max over values, ...) run the tail rule, not §6.3's z
    rule. The z rule is pinned by the tests that build with ``Z`` / ``PRE_A1_SWITCHES``."""
    return build(_lognormal())


def _uniform() -> list[float]:
    rng = np.random.default_rng(1)
    return [float(v) for v in rng.uniform(30.0, 50.0, 300)]


@pytest.fixture(scope="module")
def trained_both() -> Trained:
    return build(_uniform(), path="latitude")


@pytest.fixture(scope="module")
def trained_tail() -> Trained:
    return build(_lognormal(), cfg=TAIL)


@pytest.fixture(scope="module")
def trained_both_tail() -> Trained:
    return build(_uniform(), path="latitude", cfg=TAIL)


def _dump(store: DictStore) -> bytes:
    return orjson.dumps(
        [(k, r.KIND, r.to_dict()) for k, r in sorted(store.items())], option=orjson.OPT_SORT_KEYS
    )


# --- base helpers -------------------------------------------------------------------------


def test_squash_anchor_points() -> None:
    # Brief asked |squash(5.25) - 0.95| < 1e-3, but sigma(3) = 0.95257 exactly; pin the true value.
    assert squash(3.0, 3.0, 0.75) == 0.5
    assert abs(squash(5.25, 3.0, 0.75) - 1.0 / (1.0 + math.exp(-3.0))) < 1e-12
    assert abs(squash(5.25, 3.0, 0.75) - 0.95) < 3e-3


def test_squash_extremes_do_not_overflow() -> None:
    assert squash(-1e6, 3.0, 0.75) == 0.0
    assert squash(1e6, 3.0, 0.75) == 1.0


def test_signed_log1p() -> None:
    assert signed_log1p(0.0) == 0.0
    assert signed_log1p(math.e - 1) == pytest.approx(1.0)
    assert signed_log1p(-(math.e - 1)) == pytest.approx(-1.0)


def test_iter_fields_flat_then_derived() -> None:
    e = make_env(args={"to": ["x@acme.example"], "amount": 5})
    paths = [p for p, _ in iter_fields(e)]
    assert paths == [*e.flat_args, *e.derived]


@pytest.mark.parametrize(
    "path",
    [
        "amount",
        "meta.item_count",
        "lines.__len__",
        "Total",
        "total_amount",
        "QTY",
        "x.amount_cents",
        "body.__strlen__",
    ],
)
def test_direction_up_on_segment_boundary(path: str) -> None:
    assert direction_for(path, {}) == "up"


@pytest.mark.parametrize(
    "path", ["latitude", "account", "discount", "calendar", "meta.account_id", "x.recount"]
)
def test_direction_both_when_keyword_not_at_segment_start(path: str) -> None:
    assert direction_for(path, {}) == "both"


def test_direction_hints_override_name() -> None:
    assert direction_for("latitude", {"latitude": "money"}) == "up"
    assert direction_for("discount", {"discount": "count"}) == "up"


# --- scoring --------------------------------------------------------------------------------


def test_warm_spike_and_normal(trained_mag: Trained) -> None:  # 500 lognormal(3.6,0.7) refunds
    assert score_amount(trained_mag, 37.0) < 0.1
    # Default (tail) rule since Task 1.21. Thresholds date from the §6.3 z rule, where 20x gives
    # z~4.2-4.9 (s~0.83-0.93 by seed); the brief asked 20x > 0.9.
    assert score_amount(trained_mag, 37.0 * 20) > 0.8
    assert score_amount(trained_mag, 37.0 * 50) > 0.9
    assert score_amount(trained_mag, 37.0 / 20) < 0.05  # direction=up ignores small


def test_warm_evidence(trained_mag: Trained) -> None:
    (s,) = [s for s in scores_for(trained_mag, {"amount": 740.0}) if s.field == "amount"]
    ev = s.evidence
    assert s.detector == "magnitude"
    assert ev["direction"] == "up" and ev["prior_used"] is False
    assert ev["n"] == 491  # the first 9 obs arrive before the type is NUMERIC
    assert 25.0 < float(ev["median_raw"]) < 50.0  # type: ignore[arg-type]
    assert float(ev["iqr_raw"]) > 0.0  # type: ignore[arg-type]
    assert ev["percentile"] == 1.0
    assert "cold" not in ev


def test_direction_both_flags_small(trained_both: Trained) -> None:  # field "latitude"
    assert score_field(trained_both, "latitude", 40.0) < 0.1
    assert score_field(trained_both, "latitude", 2.0) > 0.9
    assert score_field(trained_both, "latitude", -40.0) > 0.9
    assert score_field(trained_both, "latitude", 400.0) > 0.9


def test_max_over_values_of_a_path(trained_mag: Trained) -> None:
    e = make_env(agent_id=AGENT, tool_id=TOOL, args={"amount": 37.0})
    spiky = make_env(agent_id=AGENT, tool_id=TOOL, args={"amount": 3700.0})
    both = dataclasses.replace(e, flat_args={"amount": (37.0, 3700.0)})
    (s_both,) = trained_mag.det.score(both, trained_mag.view())
    (s_spike,) = trained_mag.det.score(spiky, trained_mag.view())
    assert s_both.score == s_spike.score


@given(st.floats(0, 1e7), st.floats(0, 1e7))
def test_monotone_in_x_for_direction_up(trained_mag: Trained, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert score_amount(trained_mag, lo) <= score_amount(trained_mag, hi)


def test_score_is_idempotent_and_pure(trained_mag: Trained) -> None:
    store = trained_mag.store
    before, n_keys = _dump(store), len(store)
    tracker = SessionTracker()
    prev = make_env(agent_id=AGENT, tool_id="never.seen", args={}, response_meta={"item_count": 5})
    tracker.observe(prev)
    e = make_env(tracker=tracker, agent_id=AGENT, tool_id=TOOL, args={"amount": 9999.0})
    first = trained_mag.det.score(e, trained_mag.view())
    second = trained_mag.det.score(e, trained_mag.view())
    assert first == second
    assert {s.field for s in first} == {"amount", "__resp__.item_count"}
    assert _dump(store) == before and len(store) == n_keys  # no mutation, no new (cold) keys


def test_cold_without_prior_scores_zero() -> None:
    t = build([10.0 + i for i in range(20)])  # 11 pushed: n < mag_cold_n
    rec = t.store.get(key_mag(AGENT, TOOL, "amount"))
    assert isinstance(rec, MagRecord) and rec.n == 11
    scores = scores_for(t, {"amount": 1e9})
    assert [s.score for s in scores] == [0.0]
    ev = scores[0].evidence
    assert ev["cold"] is True and ev["prior_used"] is False
    assert ev["n"] == 11 and ev["direction"] == "up"


def test_cold_without_record_evidence_has_n_zero() -> None:
    t = build([])
    for _ in range(10):  # warm the type only; no record exists yet
        t.ti.observe(AGENT, TOOL, "amount", (5.0,))
    assert t.store.get(key_mag(AGENT, TOOL, "amount")) is None
    (s,) = scores_for(t, {"amount": 5.0})
    assert s.score == 0.0
    assert dict(s.evidence) == {"cold": True, "n": 0, "direction": "up", "prior_used": False}


class _FakePriors:
    def __init__(self, ref: MagRecord | None) -> None:
        self.ref = ref
        self.calls: list[tuple[str, str, str]] = []

    def magnitude_ref(self, tool: str, path: str, direction: str) -> MagRecord | None:
        self.calls.append((tool, path, direction))
        return self.ref


def test_cold_key_scores_against_prior_with_inflated_iqr() -> None:
    cfg = Z  # this test pins the §6.3 formula
    pooled = build(_lognormal(seed=3), cfg=cfg)
    ref = pooled.store.get(key_mag(AGENT, TOOL, "amount"))
    assert isinstance(ref, MagRecord)
    priors = _FakePriors(ref)
    t = build([30.0 + i for i in range(12)], cfg=cfg, priors=priors)  # cold per-agent key
    (s,) = scores_for(t, {"amount": 37.0 * 30})
    assert priors.calls[-1] == (TOOL, "amount", "up")
    assert s.evidence["prior_used"] is True and "cold" not in s.evidence
    # Same value against the pooled record directly (no inflation) scores strictly higher.
    (warm,) = scores_for(pooled, {"amount": 37.0 * 30})
    assert 0.0 < s.score < warm.score
    y = signed_log1p(37.0 * 30)
    scale = (ref.q.quantile(0.75) - ref.q.quantile(0.25)) / 1.349 * cfg.prior_iqr_inflate
    z = (y - ref.q.quantile(0.5)) / max(scale, cfg.mag_eps)
    assert s.score == pytest.approx(squash(z, cfg.z0, cfg.tau))


def test_cold_prior_returning_none_scores_zero() -> None:
    t = build([30.0 + i for i in range(12)], priors=_FakePriors(None))
    (s,) = scores_for(t, {"amount": 1e9})
    assert s.score == 0.0 and s.evidence["cold"] is True


def test_non_finite_values() -> None:
    t = build(_lognormal(200))
    key = key_mag(AGENT, TOOL, "amount")
    rec = t.store.get(key)
    assert isinstance(rec, MagRecord)
    n0 = rec.n
    assert scores_for(t, {"amount": float("nan")}) == []  # NaN: no score
    for bad in (float("inf"), float("-inf"), "1e999", 10**400):
        (s,) = scores_for(t, {"amount": bad})
        assert s.score == 1.0 and s.evidence["non_finite"] is True
        assert s.evidence["n"] == n0 and s.evidence["direction"] == "up"
        assert s.evidence["prior_used"] is False
        _learn(t, make_env(agent_id=AGENT, tool_id=TOOL, args={"amount": bad}))
    _learn(t, make_env(agent_id=AGENT, tool_id=TOOL, args={"amount": float("nan")}))
    assert rec.n == n0  # neither NaN nor inf is ever pushed
    MagRecord.from_dict(rec.to_dict())  # from_dict rejects non-finite estimator state


def test_nan_mixed_with_finite_scores_finite_part(trained_mag: Trained) -> None:
    e = make_env(agent_id=AGENT, tool_id=TOOL, args={"amount": 37.0})
    mixed = dataclasses.replace(e, flat_args={"amount": (float("nan"), 37.0)})
    (s,) = trained_mag.det.score(mixed, trained_mag.view())
    assert s.score < 0.1


def test_constant_field_is_finite() -> None:
    t = build([10.0] * 100)
    s_same = score_amount(t, 10.0)
    s_big = score_amount(t, 20.0)
    assert math.isfinite(s_same) and s_same < 0.05
    assert s_big > 0.9


def test_hint_sets_direction() -> None:
    rng = np.random.default_rng(2)
    vals = [float(v) for v in rng.uniform(30.0, 50.0, 100)]
    hinted = build(vals, path="latitude", hints={TOOL: ToolHint(field_hints={"latitude": "money"})})
    rec = hinted.store.get(key_mag(AGENT, TOOL, "latitude"))
    assert isinstance(rec, MagRecord) and rec.direction == "up"
    assert score_field(hinted, "latitude", 1.0) < 0.05


def test_mag_record_roundtrip(trained_mag: Trained) -> None:
    rec = trained_mag.store.get(key_mag(AGENT, TOOL, "amount"))
    assert isinstance(rec, MagRecord)
    d = orjson.loads(orjson.dumps(rec.to_dict()))
    back = MagRecord.from_dict(d)
    assert back.to_dict() == rec.to_dict()
    assert MagRecord.KIND == "mag"


# --- response metadata (§2.3) ---------------------------------------------------------------


SF = "salesforce.query"
RESP_KEY = key_mag(AGENT, SF, "__resp__.item_count")


def _step(t: Trained, e: ActionEnvelope, *, allowed: bool) -> list[DetectorScore]:
    """Engine order: score first; fold in (types + detector) only for allowed outcomes."""
    scores = t.det.score(e, t.view(e.tool_id))
    if allowed:
        _learn(t, e)
    return scores


def _resp_score(scores: list[DetectorScore]) -> DetectorScore:
    (s,) = [s for s in scores if s.field == "__resp__.item_count"]
    return s


def _train_session(cfg: EngineConfig = DEFAULT) -> tuple[Trained, SessionTracker]:
    """One session of 200 allowed SOQL queries, each returning 10-30 rows."""
    store = DictStore()
    t = Trained(MagnitudeDetector(store, cfg), TypeInference(store, cfg), store)
    tracker = SessionTracker()
    rng = np.random.default_rng(4)
    for i, c in enumerate(rng.integers(10, 31, 200)):
        e = make_env(
            tracker=tracker,
            agent_id=AGENT,
            tool_id=SF,
            seq_no=i,
            args={"soql": "select id"},
            response_meta={"item_count": int(c)},
        )
        _step(t, e, allowed=True)
        tracker.observe(e)
    return t, tracker


def test_response_item_count_learned_from_prev_response_on_next_call() -> None:
    t, _ = _train_session()
    rec = t.store.get(RESP_KEY)
    # Each count is pushed by the next call's update; a session's last response is never learned.
    assert isinstance(rec, MagRecord) and rec.n == 199 and rec.direction == "up"


def test_update_ignores_own_response_meta() -> None:
    t = build([])
    e = make_env(agent_id=AGENT, tool_id=SF, args={}, response_meta={"item_count": 7})
    assert e.prev_response is None
    t.det.update(e, t.view(SF))
    assert t.store.get(RESP_KEY) is None


def test_response_item_count_scored_on_next_call() -> None:
    t, tracker = _train_session()
    rec = t.store.get(RESP_KEY)
    assert isinstance(rec, MagRecord)

    def next_call(item_count: int) -> DetectorScore:
        tr = SessionTracker()
        tr.observe(
            make_env(
                session_id="s2",
                agent_id=AGENT,
                tool_id=SF,
                args={"soql": "select id"},
                response_meta={"item_count": item_count},
            )
        )
        e = make_env(session_id="s2", tracker=tr, agent_id=AGENT, tool_id="gmail.send_email")
        assert e.prev_response is not None
        return _resp_score(t.det.score(e, t.view("gmail.send_email")))

    typical = next_call(20)
    assert typical.score < 0.1
    spike = next_call(20 * 50)
    assert spike.detector == "magnitude" and spike.field == "__resp__.item_count"
    assert spike.score > 0.9 and spike.evidence["tool"] == SF


def test_blocked_next_call_never_moves_response_baseline() -> None:
    t, tracker = _train_session()
    rec = t.store.get(RESP_KEY)
    assert isinstance(rec, MagRecord)
    n_before = rec.n

    # k: an allowed query whose response is 50x the usual row count.
    k = make_env(
        tracker=tracker,
        agent_id=AGENT,
        tool_id=SF,
        seq_no=200,
        args={"soql": "select id"},
        response_meta={"item_count": 1000},
    )
    _step(t, k, allowed=True)  # pushes the *previous* (normal) count, not k's own 1000
    tracker.observe(k)
    assert rec.n == n_before + 1
    snapshot = orjson.dumps(rec.to_dict())

    # k+1: carries k's response; (a) it scores high, (b) blocked, so nothing is pushed.
    k1 = make_env(tracker=tracker, agent_id=AGENT, tool_id="gmail.send_email", seq_no=201)
    assert k1.prev_response is not None and k1.prev_response.item_count == 1000
    spike = _resp_score(_step(t, k1, allowed=False))
    assert spike.score > 0.9
    assert rec.n == n_before + 1
    assert orjson.dumps(rec.to_dict()) == snapshot
    assert rec.q.quantile(1.0) < signed_log1p(1000)  # 1000 never entered the baseline


# --- Task 1.14 sweep switches -----------------------------------------------------------------


def test_direction_default_up_applies_only_to_unhinted_unnamed_fields() -> None:
    assert direction_for("latitude", {}) == "both"
    assert direction_for("latitude", {}, default="up") == "up"
    assert direction_for("amount", {}, default="both") == "up"  # name rule still wins
    up = EngineConfig(mag_direction_default="up")
    store = DictStore()
    t = Trained(MagnitudeDetector(store, up), TypeInference(store, up), store)
    for v in np.random.default_rng(3).normal(50.0, 2.0, 200):
        _learn(t, make_env(agent_id=AGENT, tool_id=TOOL, args={"latitude": float(v)}))
    rec = store.get(key_mag(AGENT, TOOL, "latitude"))
    assert isinstance(rec, MagRecord) and rec.direction == "up"
    assert score_field(t, "latitude", 1.0) < 0.05  # far below: one-sided up ignores it


def test_response_signal_off_neither_scores_nor_learns_response_counts() -> None:
    store = DictStore()
    off = EngineConfig(mag_resp_signal=False)
    t = Trained(MagnitudeDetector(store, off), TypeInference(store, off), store)
    tracker = SessionTracker()
    for i in range(60):
        e = make_env(
            tracker=tracker,
            agent_id=AGENT,
            tool_id=SF,
            seq_no=i,
            args={"soql": "select id"},
            response_meta={"item_count": 20 if i < 59 else 5000},
        )
        scores = _step(t, e, allowed=True)
        assert all(s.field != "__resp__.item_count" for s in scores)
        tracker.observe(e)
    assert store.get(RESP_KEY) is None


# --- Amendment A1.2: tail score on a two-piece log scale (mag_score="tail") -------------------

Z75 = 0.6744897501960817  # inverse normal cdf at 0.75
Z95 = 1.6448536269514722  # inverse normal cdf at 0.95
PRIOR = 0.005  # tail_prior
PRE_A1_MAG_KEYS = {"q", "ewma", "ewma2", "direction", "n"}
TAIL_EVIDENCE_KEYS = {
    "observed",
    "z",
    "p_tail",
    "percentile",
    "median_raw",
    "iqr_raw",
    "n",
    "direction",
    "prior_used",
}


def test_default_is_tail() -> None:
    """A1.9 (Task 1.21): the tail score is the default; "z" stays selectable."""
    assert EngineConfig().mag_score == "tail" and EngineConfig().tail_prior == 0.005


def test_z_rule_is_unchanged_and_has_no_p_tail() -> None:
    """The pre-amendment engine (``PRE_A1_SWITCHES``) still scores magnitude by the §6.3 rule."""
    pre_a1 = build(_lognormal(), cfg=EngineConfig().replace(**PRE_A1_SWITCHES))
    explicit = build(_lognormal(), cfg=Z)
    for x in (37.0, 370.0, 740.0):
        (old,) = scores_for(pre_a1, {"amount": x})
        (same,) = scores_for(explicit, {"amount": x})
        assert old == same
        assert set(old.evidence) == TAIL_EVIDENCE_KEYS - {"p_tail"}
        z = old.evidence["z"]
        assert isinstance(z, float) and old.score == squash(z, Z.z0, Z.tau)


@pytest.mark.parametrize("value", ["Tail", "zscore", "", None, 1])
def test_unknown_mag_score_rejected(value: object) -> None:
    # A mistyped switch must not silently run the other rule (like vel_count_stat).
    cfg = EngineConfig().replace(mag_score=value)
    with pytest.raises(ValueError, match=r"unknown mag_score .*known: \('z', 'tail'\)"):
        MagnitudeDetector(DictStore(), cfg)
    with pytest.raises(ValueError, match="mag_score"):
        Engine(cfg)
    with pytest.raises(ValueError, match="mag_score"):
        Engine(cfg, enabled=frozenset({"velocity"}))  # the engine always builds the detector


def test_tail_score_matches_formula(trained_tail: Trained) -> None:
    # 500 lognormal(3.6, 0.7), field "amount" (up)
    (s,) = scores_for(trained_tail, {"amount": 370.0})
    q = record(trained_tail, "amount").q
    d = signed_log1p(370.0) - q.quantile(0.5)
    sigma = max(
        (q.quantile(0.75) - q.quantile(0.5)) / 0.6744897501960817,
        (q.quantile(0.95) - q.quantile(0.5)) / 1.6448536269514722,
    )
    z = d / max(sigma, 0.05)
    assert s.evidence["z"] == pytest.approx(z) and s.evidence["p_tail"] == pytest.approx(
        normal_sf(z)
    )
    assert s.score == pytest.approx(posterior(normal_sf(z), 0.005))
    assert set(s.evidence) == TAIL_EVIDENCE_KEYS  # today's keys plus p_tail
    assert s.evidence["observed"] == 370.0 and s.evidence["prior_used"] is False


def test_tail_scale_is_taken_on_the_side_the_value_falls_on() -> None:
    # 40 + Exp(5): the upper half is about twice as wide (in log units) as the lower half.
    rng = np.random.default_rng(6)
    t = build([40.0 + float(v) for v in rng.exponential(5.0, 2000)], path="latitude", cfg=TAIL)
    q = record(t, "latitude").q
    q05, q25, q50, q75, q95 = (q.quantile(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95))
    lower = max((q50 - q25) / Z75, (q50 - q05) / Z95)
    upper = max((q75 - q50) / Z75, (q95 - q50) / Z95)
    assert TAIL.mag_eps < lower < 0.7 * upper  # two different scales, both above the floor
    for x, sigma in ((35.0, lower), (67.0, upper)):
        (s,) = scores_for(t, {"latitude": x})
        z = abs(signed_log1p(x) - q50) / sigma
        assert s.evidence["z"] == pytest.approx(z)
        assert s.evidence["p_tail"] == pytest.approx(2.0 * normal_sf(z))  # direction both
        assert s.score == pytest.approx(posterior(2.0 * normal_sf(z), PRIOR))
        assert 0.05 < s.score < 0.999  # not saturated: the comparison above is informative


def test_tail_lower_scale_takes_the_larger_of_quartile_and_q05_terms() -> None:
    # 60 - Exp(5): the mirror image. Its heavy tail is below the median, where the q05 term
    # exceeds the quartile term (on 40 + Exp(5) the quartile term wins on the lower side).
    rng = np.random.default_rng(6)
    t = build([60.0 - float(v) for v in rng.exponential(5.0, 2000)], path="latitude", cfg=TAIL)
    q = record(t, "latitude").q
    q05, q25, q50 = (q.quantile(p) for p in (0.05, 0.25, 0.5))
    quartile_term, q05_term = (q50 - q25) / Z75, (q50 - q05) / Z95
    assert TAIL.mag_eps < quartile_term < 0.7 * q05_term  # 0.087 and 0.143
    x = 38.0
    assert signed_log1p(x) < q50
    (s,) = scores_for(t, {"latitude": x})
    z = (q50 - signed_log1p(x)) / max(quartile_term, q05_term)
    assert s.evidence["z"] == pytest.approx(z)
    assert s.score == pytest.approx(posterior(2.0 * normal_sf(z), PRIOR))
    assert 0.05 < s.score < 0.999  # not saturated: the quartile term alone gives a larger z


def test_tail_warm_spike_and_normal(trained_tail: Trained) -> None:
    assert score_amount(trained_tail, 37.0) < 0.05  # 0.011
    assert score_amount(trained_tail, 37.0 * 20) > 0.98  # 0.991 (the §6.3 score is 0.82)
    assert score_amount(trained_tail, 37.0 * 100) > 0.9999
    assert score_amount(trained_tail, 37.0 / 20) < 0.05  # direction=up ignores small


def test_tail_direction_up_clamps_z_at_zero_below_the_median(trained_tail: Trained) -> None:
    # z = max(d / scale, 0): every value below the median scores exactly as the median does.
    q50 = record(trained_tail, "amount").q.quantile(0.5)
    for x in (37.0 / 20, 0.0, 20.0):
        assert signed_log1p(x) < q50
        (s,) = scores_for(trained_tail, {"amount": x})
        assert s.evidence["direction"] == "up"
        assert s.evidence["z"] == 0.0 and s.evidence["p_tail"] == 0.5
        assert s.score == pytest.approx(posterior(0.5, PRIOR))


@pytest.mark.parametrize(
    ("sigma", "x10", "x100"),
    [
        (0.5, (0.99, 1.0), (0.9999, 1.0)),
        (0.86, (0.40, 0.72), (0.999, 1.0)),
        (2.3, (0.0, 0.08), (0.05, 0.35)),
    ],
)
def test_tail_is_relative_to_the_fields_log_dispersion(
    sigma: float, x10: tuple[float, float], x100: tuple[float, float]
) -> None:
    # 5,000 lognormal(3.6, sigma) values; k x exp(3.6).
    # Exact log-normal values: 1.000/1.000, 0.575/1.000, 0.031/0.182
    rng = np.random.default_rng(0)
    t = build([float(v) for v in rng.lognormal(3.6, sigma, 5000)], cfg=TAIL)
    median = math.exp(3.6)
    assert x10[0] <= score_amount(t, 10.0 * median) <= x10[1]
    assert x100[0] <= score_amount(t, 100.0 * median) <= x100[1]


def test_tail_zero_negative_and_non_finite() -> None:
    # field "balance_delta" ~ normal(-200, 20), direction both
    rng = np.random.default_rng(5)
    path = "balance_delta"
    t = build([float(v) for v in rng.normal(-200.0, 20.0, 500)], path=path, cfg=TAIL)
    rec = record(t, path)
    assert rec.direction == "both"

    def score(x: float) -> float:
        return score_field(t, path, x)

    assert score(-200.0) < 0.02 and score(-20000.0) > 0.999
    assert score(0.0) > 0.999 and score(200.0) > 0.999
    # y = sign(x) * ln(1 + |x|): zero is y = 0 and lies above this field's (negative) median,
    # so it is scaled by the upper markers; -20000 lies below and is scaled by the lower ones.
    q05, q25, q50, q75, q95 = (rec.q.quantile(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95))
    assert q95 < 0.0
    upper = max((q75 - q50) / Z75, (q95 - q50) / Z95)
    lower = max((q50 - q25) / Z75, (q50 - q05) / Z95)
    assert min(upper, lower) > TAIL.mag_eps
    for x, y, sigma in ((0.0, 0.0, upper), (-20000.0, -math.log1p(20000.0), lower)):
        (s,) = scores_for(t, {path: x})
        assert s.evidence["z"] == pytest.approx(abs(y - q50) / sigma)
        assert s.evidence["p_tail"] == 0.0 and s.score == 1.0  # p = 0 gives exactly 1.0
    n0 = rec.n
    assert scores_for(t, {path: float("nan")}) == []  # NaN: no score
    for bad in (float("inf"), float("-inf")):
        (s,) = scores_for(t, {path: bad})
        assert s.score == 1.0 and s.evidence["non_finite"] is True
        assert "p_tail" not in s.evidence and s.evidence["n"] == n0
        _learn(t, make_env(agent_id=AGENT, tool_id=TOOL, args={path: bad}))
    _learn(t, make_env(agent_id=AGENT, tool_id=TOOL, args={path: float("nan")}))
    assert rec.n == n0  # neither NaN nor inf is ever pushed


def test_tail_constant_field() -> None:
    t = build([10.0] * 100, cfg=TAIL)
    assert score_amount(t, 10.0) == pytest.approx(0.00995025, abs=1e-6)  # posterior(0.5, 0.005)
    assert score_amount(t, 11.5) == pytest.approx(0.487, abs=5e-3)  # scale = mag_eps = 0.05
    assert score_amount(t, 20.0) > 0.9999


def test_tail_constant_field_floor_is_read_from_mag_eps() -> None:
    wide = TAIL.replace(mag_eps=0.5)
    t = build([10.0] * 100, cfg=wide)
    (s,) = scores_for(t, {"amount": 11.5})
    z = (math.log1p(11.5) - math.log1p(10.0)) / 0.5
    assert s.evidence["z"] == pytest.approx(z)  # 0.256; the default floor 0.05 gives 2.56
    assert s.score == pytest.approx(posterior(normal_sf(z), PRIOR))


def test_tail_reads_tail_prior_and_not_cal_alpha() -> None:
    """D-023: ``tail_prior`` is a field of its own so that a ``cal_alpha`` sweep moves thresholds
    and never scores. A wiring check on one constant field, not a sweep of the fixed constant."""
    values = [10.0] * 100
    other = build(values, cfg=TAIL.replace(tail_prior=0.05))
    assert score_amount(other, 10.0) == pytest.approx(posterior(0.5, 0.05))
    assert posterior(0.5, 0.05) == pytest.approx(0.0952381, abs=1e-6)
    (s,) = scores_for(build(values, cfg=TAIL.replace(tail_prior=0.02)), {"amount": 11.5})
    p = s.evidence["p_tail"]
    assert isinstance(p, float) and 0.0 < p < 0.5
    assert s.score == pytest.approx(posterior(p, 0.02))  # 0.795; 0.487 at the default
    swept = build(values, cfg=TAIL.replace(cal_alpha=0.05))  # the bench "alpha" sweep
    assert score_amount(swept, 10.0) == pytest.approx(posterior(0.5, PRIOR))
    assert score_amount(swept, 11.5) == score_amount(build(values, cfg=TAIL), 11.5)


@pytest.mark.parametrize(
    "value", [-1.0, -0.01, 0.0, 1.0, 1.5, float("nan"), float("inf"), float("-inf")]
)
def test_tail_prior_outside_the_open_unit_interval_rejected(value: float) -> None:
    # A prior outside (0, 1) gives scores outside [0, 1] (or a division by zero at -1.0).
    cfg = TAIL.replace(tail_prior=value)
    with pytest.raises(ValueError, match=r"tail_prior must be in \(0, 1\)"):
        MagnitudeDetector(DictStore(), cfg)
    with pytest.raises(ValueError, match="tail_prior"):
        Engine(cfg)
    with pytest.raises(ValueError, match="tail_prior"):
        Engine(EngineConfig().replace(tail_prior=value))  # checked under either rule


@pytest.mark.parametrize("value", [1e-6, 0.005, 0.5, 0.999])
def test_tail_prior_inside_the_open_unit_interval_accepted(value: float) -> None:
    t = build([10.0] * 100, cfg=TAIL.replace(tail_prior=value))
    for x in (10.0, 11.5, 1e6):
        assert 0.0 <= score_amount(t, x) <= 1.0


def test_tail_two_sided_for_direction_both(trained_both_tail: Trained) -> None:
    # uniform(30, 50), field "latitude"
    assert score_field(trained_both_tail, "latitude", 40.0) < 0.01  # p = min(1, 2 Q(z)) ~ 1
    assert score_field(trained_both_tail, "latitude", 2.0) > 0.999
    assert score_field(trained_both_tail, "latitude", 800.0) > 0.999
    (at_median,) = scores_for(
        trained_both_tail,
        {"latitude": math.expm1(record(trained_both_tail, "latitude").q.quantile(0.5))},
    )
    assert at_median.evidence["p_tail"] == pytest.approx(1.0)  # 2 Q(0), clipped to at most 1
    assert at_median.score == pytest.approx(PRIOR)


def test_tail_small_integer_field_pinned_limitation() -> None:
    """PINS a known limitation (spec A1.12 item 8): the upper markers of a tied integer field
    coincide."""
    t = build([1, 2, 3, 4] * 100, path="qty", cfg=TAIL)
    assert score_field(t, "qty", 4) < 0.1
    assert 0.2 < score_field(t, "qty", 5) < 0.5  # 0.327 on the prototype; §6.3 gives 0.10
    assert score_field(t, "qty", 8) > 0.99


@given(st.floats(0, 1e7), st.floats(0, 1e7))
def test_tail_monotone_in_x_for_direction_up(trained_tail: Trained, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert score_amount(trained_tail, lo) <= score_amount(trained_tail, hi)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_tail_scores_stay_in_the_unit_interval(trained_both_tail: Trained, x: float) -> None:
    (s,) = scores_for(trained_both_tail, {"latitude": x})
    assert 0.0 <= s.score <= 1.0
    p = s.evidence["p_tail"]
    assert isinstance(p, float) and 0.0 <= p <= 1.0


def test_tail_cold_and_prior() -> None:
    cold = build([30.0 + i for i in range(12)], cfg=TAIL)  # n < mag_cold_n, no prior
    (c,) = scores_for(cold, {"amount": 1e9})
    assert c.score == 0.0
    assert dict(c.evidence) == {"cold": True, "n": 3, "direction": "up", "prior_used": False}

    pooled = build(_lognormal(seed=3), cfg=TAIL)
    ref = record(pooled, "amount")
    priors = _FakePriors(ref)
    t = build([30.0 + i for i in range(12)], cfg=TAIL, priors=priors)  # cold per-agent key
    (s,) = scores_for(t, {"amount": 370.0})
    assert priors.calls[-1] == (TOOL, "amount", "up")
    assert s.evidence["prior_used"] is True and "cold" not in s.evidence
    assert s.evidence["n"] == ref.n
    d = signed_log1p(370.0) - ref.q.quantile(0.5)
    sigma = max(
        (ref.q.quantile(0.75) - ref.q.quantile(0.5)) / Z75,
        (ref.q.quantile(0.95) - ref.q.quantile(0.5)) / Z95,
    )
    z_inflated = d / max(sigma * TAIL.prior_iqr_inflate, TAIL.mag_eps)  # scale x 1.5
    assert s.evidence["z"] == pytest.approx(z_inflated)
    assert s.score == pytest.approx(posterior(normal_sf(z_inflated), 0.005))
    # The same value against the pooled record directly (no inflation) scores strictly higher.
    (warm,) = scores_for(pooled, {"amount": 370.0})
    assert warm.evidence["z"] == pytest.approx(z_inflated * TAIL.prior_iqr_inflate)
    assert 0.0 < s.score < warm.score < 1.0


def test_tail_prior_inflation_is_applied_inside_the_mag_eps_floor() -> None:
    # scale = max(sigma * inflate, mag_eps): a constant pooled prior (sigma = 0) is scaled by the
    # floor itself, not by the floor times prior_iqr_inflate.
    pooled = build([10.0] * 100, cfg=TAIL)
    ref = record(pooled, "amount")
    t = build([30.0 + i for i in range(12)], cfg=TAIL, priors=_FakePriors(ref))  # cold key
    (s,) = scores_for(t, {"amount": 11.5})
    assert s.evidence["prior_used"] is True and s.evidence["n"] == ref.n
    z = (math.log1p(11.5) - math.log1p(10.0)) / TAIL.mag_eps
    assert s.evidence["z"] == pytest.approx(z)  # 2.56; 1.71 if the floor were inflated
    assert s.score == pytest.approx(posterior(normal_sf(z), PRIOR))
    assert s.score == pytest.approx(score_amount(pooled, 11.5))  # same floor, warm or prior


def test_tail_score_is_idempotent_and_pure(trained_tail: Trained) -> None:
    store = trained_tail.store
    before, n_keys = _dump(store), len(store)
    tracker = SessionTracker()
    prev = make_env(agent_id=AGENT, tool_id="never.seen", args={}, response_meta={"item_count": 5})
    tracker.observe(prev)
    e = make_env(tracker=tracker, agent_id=AGENT, tool_id=TOOL, args={"amount": 9999.0})
    first = trained_tail.det.score(e, trained_tail.view())
    second = trained_tail.det.score(e, trained_tail.view())
    assert first == second
    assert {s.field for s in first} == {"amount", "__resp__.item_count"}
    assert "p_tail" in first[0].evidence
    assert _dump(store) == before and len(store) == n_keys  # no mutation, no new (cold) keys


def test_tail_response_item_count_scored_on_next_call() -> None:
    t, _ = _train_session(TAIL)

    def next_call(item_count: int) -> DetectorScore:
        tr = SessionTracker()
        tr.observe(
            make_env(
                session_id="s2",
                agent_id=AGENT,
                tool_id=SF,
                args={"soql": "select id"},
                response_meta={"item_count": item_count},
            )
        )
        e = make_env(session_id="s2", tracker=tr, agent_id=AGENT, tool_id="gmail.send_email")
        return _resp_score(t.det.score(e, t.view("gmail.send_email")))

    assert next_call(20).score < 0.05
    spike = next_call(20 * 50)
    assert spike.score > 0.999 and spike.evidence["tool"] == SF and "p_tail" in spike.evidence


def test_pre_a1_mag_state_scores_immediately(tmp_path: Path, trained_tail: Trained) -> None:
    old = build(_lognormal(), cfg=Z)  # the §6.3 rule: the engine before the amendment
    path, again = tmp_path / "pre-a1.jsonl", tmp_path / "reloaded.jsonl"
    old.store.dump(path)
    rows = {r["key"]: r["data"] for r in map(orjson.loads, path.read_bytes().splitlines())}
    assert set(rows[key_mag(AGENT, TOOL, "amount")]) == PRE_A1_MAG_KEYS  # no new record field
    store = DictStore()
    store.load(path)
    new = Trained(MagnitudeDetector(store, TAIL), TypeInference(store, TAIL), store)
    (first,) = scores_for(new, {"amount": 740.0})  # the first call after loading is scored warm
    assert "cold" not in first.evidence and first.evidence["n"] == 491
    assert first.evidence["prior_used"] is False and "p_tail" in first.evidence
    # scored by the new rule from the old state: the same as a record trained under "tail"
    (same,) = scores_for(trained_tail, {"amount": 740.0})
    assert first == same and first.score > 0.98
    (z_rule,) = scores_for(old, {"amount": 740.0})
    assert z_rule.score < first.score  # 0.82 under the §6.3 rule
    store.dump(again)
    assert again.read_bytes() == path.read_bytes()  # scoring under "tail" changed no state
