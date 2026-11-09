from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import iter_fields, signed_log1p, squash
from agentfw.core.detectors.magnitude import MagnitudeDetector, MagRecord, direction_for
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.hints import ToolHint
from agentfw.core.normalize import SessionTracker
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_mag
from tests.factories import make_env

AGENT = "a1"
TOOL = "stripe.create_refund"


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


def build(values: Iterable[Any], path: str = "amount", **det_kw: Any) -> Trained:
    store = DictStore()
    cfg = EngineConfig()
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


def _lognormal(n: int = 500, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.lognormal(3.6, 0.7, n)]


@pytest.fixture(scope="module")
def trained_mag() -> Trained:
    return build(_lognormal())


@pytest.fixture(scope="module")
def trained_both() -> Trained:
    rng = np.random.default_rng(1)
    return build([float(v) for v in rng.uniform(30.0, 50.0, 300)], path="latitude")


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


def test_direction_for() -> None:
    assert direction_for("amount", {}) == "up"
    assert direction_for("meta.item_count", {}) == "up"
    assert direction_for("lines.__len__", {}) == "up"
    assert direction_for("Total", {}) == "up"
    assert direction_for("latitude", {}) == "both"
    assert direction_for("latitude", {"latitude": "money"}) == "up"
    assert direction_for("x", {"x": "count"}) == "up"


# --- scoring --------------------------------------------------------------------------------


def test_warm_spike_and_normal(trained_mag: Trained) -> None:  # 500 lognormal(3.6,0.7) refunds
    assert score_amount(trained_mag, 37.0) < 0.1
    # Brief asked 20x > 0.9; under the spec formula 20x gives z~4.2-4.9 (s~0.83-0.93 by seed).
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
    assert scores[0].evidence["cold"] is True
    assert scores[0].evidence["prior_used"] is False


class _FakePriors:
    def __init__(self, ref: MagRecord | None) -> None:
        self.ref = ref
        self.calls: list[tuple[str, str, str]] = []

    def magnitude_ref(self, tool: str, path: str, direction: str) -> MagRecord | None:
        self.calls.append((tool, path, direction))
        return self.ref


def test_cold_key_scores_against_prior_with_inflated_iqr() -> None:
    cfg = EngineConfig()
    pooled = build(_lognormal(seed=3))
    ref = pooled.store.get(key_mag(AGENT, TOOL, "amount"))
    assert isinstance(ref, MagRecord)
    priors = _FakePriors(ref)
    t = build([30.0 + i for i in range(12)], priors=priors)  # cold per-agent key
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


def test_response_item_count_scored_on_next_call() -> None:
    store = DictStore()
    cfg = EngineConfig()
    t = Trained(MagnitudeDetector(store, cfg), TypeInference(store, cfg), store)
    rng = np.random.default_rng(4)
    for c in rng.integers(10, 31, 200):
        q = make_env(
            agent_id=AGENT,
            tool_id="salesforce.query",
            args={"soql": "select id"},
            response_meta={"item_count": int(c)},
        )
        _learn(t, q)
    rec = store.get(key_mag(AGENT, "salesforce.query", "__resp__.item_count"))
    assert isinstance(rec, MagRecord) and rec.n == 200 and rec.direction == "up"

    def next_call_score(item_count: int) -> DetectorScore:
        tracker = SessionTracker()
        tracker.observe(
            make_env(
                agent_id=AGENT,
                tool_id="salesforce.query",
                args={"soql": "select id"},
                response_meta={"item_count": item_count},
            )
        )
        e = make_env(tracker=tracker, agent_id=AGENT, tool_id="gmail.send_email", args={})
        assert e.prev_response is not None
        (s,) = t.det.score(e, t.view("gmail.send_email"))
        return s

    typical = next_call_score(20)
    assert typical.field == "__resp__.item_count" and typical.score < 0.1
    spike = next_call_score(20 * 50)
    assert spike.detector == "magnitude" and spike.field == "__resp__.item_count"
    assert spike.score > 0.9
    assert spike.evidence["tool"] == "salesforce.query"
