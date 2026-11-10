from __future__ import annotations

import dataclasses
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.velocity import VelocityDetector, VelocityRecord, _roll
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.hints import ToolHint
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_vel
from tests.factories import make_env

AGENT = "a1"
TOOL = "stripe.create_refund"
MIN = 60_000
T0 = 1_700_000_000_000 - 1_700_000_000_000 % (24 * 60 * MIN)  # midnight UTC
CFG = EngineConfig()
TYPES = TypeView(TypeInference(DictStore(), CFG), AGENT, TOOL)  # velocity ignores types


def env(ts: int, args: dict[str, Any] | None = None, tool: str = TOOL) -> ActionEnvelope:
    return make_env(
        agent_id=AGENT, tool_id=tool, ts_unix_ms=ts, args={"amount": 37.5} if args is None else args
    )


def at(e: ActionEnvelope, ts: int) -> ActionEnvelope:
    """Same call, different time (cheap: skips normalization)."""
    return dataclasses.replace(e, ts_unix_ms=ts)


def new_det(**kw: Any) -> VelocityDetector:
    return VelocityDetector(DictStore(), CFG, **kw)


def step(det: VelocityDetector, e: ActionEnvelope) -> DetectorScore:
    """Engine order: score, then update (the call is allowed)."""
    [s] = det.score(e, TYPES)
    det.update(e, TYPES)
    return s


def record(det: VelocityDetector, path: str | None = None) -> VelocityRecord:
    rec = det._store.get(key_vel(AGENT, TOOL, path))
    assert isinstance(rec, VelocityRecord)
    return rec


def hist_n(rec: VelocityRecord) -> int:
    assert rec.hist_count is not None
    return rec.hist_count.n


def snapshot(det: VelocityDetector) -> bytes:
    rows = sorted((k, r.to_dict()) for k, r in det._store.items())
    return orjson.dumps(rows, option=orjson.OPT_SORT_KEYS)


def ev_int(s: DetectorScore, key: str) -> int:
    v = s.evidence[key]
    assert isinstance(v, int)
    return v


def cadence(start: int, minutes: int, every: int) -> Iterator[int]:
    """Timestamps one call per ``every`` minutes (at second 7 of the minute)."""
    for m in range(0, minutes, every):
        yield start + m * MIN + 7_000


# -- the headline scenario -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_days() -> tuple[VelocityDetector, list[DetectorScore], float, int]:
    """One refund every 5 minutes for 3 days (864 calls) with narrow-band amounts.

    Returns (detector, day-3 scores, median amount, last call ts).
    """
    rng = np.random.default_rng(7)
    amounts = [round(float(a), 2) for a in rng.uniform(30.0, 45.0, 3 * 288)]
    templates = {a: env(T0, {"amount": a}) for a in set(amounts)}
    det = new_det()
    day3: list[DetectorScore] = []
    ts = 0
    for i, (ts, a) in enumerate(zip(cadence(T0, 3 * 24 * 60, 5), amounts, strict=True)):
        s = step(det, at(templates[a], ts))
        if i >= 2 * 288:
            day3.append(s)
    return det, day3, float(np.median(amounts)), ts


def test_normal_cadence_scores_low(
    three_days: tuple[VelocityDetector, list[DetectorScore], float, int],
) -> None:
    _, day3, _, _ = three_days
    assert len(day3) == 288
    assert max(s.score for s in day3) < 0.2
    assert all(ev_int(s, "n_hist") >= CFG.vel_cold_n for s in day3)  # warm, not cold zeros


def test_burst_of_median_sized_refunds_is_caught(
    three_days: tuple[VelocityDetector, list[DetectorScore], float, int],
) -> None:
    det, _, median_amount, last = three_days
    e = env(last, {"amount": median_amount})
    burst_start = last + 2 * MIN
    scores = [step(det, at(e, burst_start + i * 3_000)) for i in range(40)]  # 40 calls in 2 min
    s15 = scores[14]
    assert s15.score > 0.95
    assert ev_int(s15, "window_count") >= 15
    assert ev_int(s15, "n_hist") >= CFG.vel_cold_n
    # Normal: 2 calls per 10-minute window, i.e. ~2 median refunds of money. With
    # median-sized amounts both signals fire; whichever wins reports its own baseline.
    if s15.evidence["signal"] == "count":
        assert s15.field is None
        assert s15.evidence["baseline_median"] == pytest.approx(2.0, abs=0.5)
    else:
        assert s15.field == "amount"
        assert s15.evidence["baseline_median"] == pytest.approx(2 * median_amount, rel=0.1)
    assert scores[-1].score > 0.95


def test_count_signal_alone_catches_burst_after_three_days() -> None:
    """A tool with no money field: the burst must be caught by the count signal alone."""
    e = env(T0, {"customer": "cus_1"})
    det = new_det()
    last = 0
    for last in cadence(T0, 3 * 24 * 60, 5):  # 864 calls
        step(det, at(e, last))
    assert det._store.get(key_vel(AGENT, TOOL, "customer")) is None
    burst = [step(det, at(e, last + 2 * MIN + i * 3_000)) for i in range(40)]
    s15 = burst[14]
    assert s15.score > 0.95
    assert s15.field is None
    assert s15.evidence["signal"] == "count"
    assert s15.evidence["baseline_median"] == pytest.approx(2.0, abs=0.5)


# -- count vs. sum signal ------------------------------------------------------------------------


def _warm(det: VelocityDetector, amount: float = 37.5, hours: int = 6) -> int:
    e = env(T0, {"amount": amount})
    ts = 0
    for ts in cadence(T0, hours * 60, 5):
        det.update(at(e, ts), TYPES)
    return ts


def test_count_signal_reports_no_field() -> None:
    det = new_det()
    e = env(T0, {"customer": "cus_1"})  # no money field: only the count key exists
    last = 0
    for last in cadence(T0, 6 * 60, 5):
        det.update(at(e, last), TYPES)
    for i in range(14):
        det.update(at(e, last + MIN + i * 1_000), TYPES)
    [s] = det.score(at(e, last + MIN + 20_000), TYPES)
    assert s.detector == "velocity"
    assert s.score > 0.95
    assert s.field is None
    assert s.evidence["signal"] == "count"


def test_sum_signal_wins_for_normal_count_but_huge_window_sum() -> None:
    det = new_det()
    last = _warm(det, amount=37.5)
    [s] = det.score(env(last + 5 * MIN, {"amount": 37_500.0}), TYPES)
    assert s.field == "amount"
    assert s.evidence["signal"] == "sum"
    assert s.score > 0.95
    assert s.evidence["window_count"] == 2  # count is normal


def test_cold_scores_zero() -> None:
    det = new_det()
    [s] = det.score(env(T0), TYPES)
    assert s.score == 0.0
    assert s.evidence["n_hist"] == 0
    assert s.evidence["window_count"] == 1
    for ts in cadence(T0, 20, 1):  # 20 active minutes < vel_cold_n history samples
        det.update(at(env(T0), ts), TYPES)
    e = env(T0 + 21 * MIN)
    for i in range(30):
        det.update(at(e, T0 + 21 * MIN + i), TYPES)
    [s] = det.score(e, TYPES)
    assert ev_int(s, "n_hist") < CFG.vel_cold_n
    assert s.score == 0.0


# -- clock ---------------------------------------------------------------------------------------


def test_idle_gap_adds_at_most_window_plus_one_samples() -> None:
    det = new_det()
    last = _warm(det)
    rec = record(det)
    n_before = hist_n(rec)
    e = env(last + 12 * 60 * MIN)
    [s] = det.score(e, TYPES)
    assert s.evidence["window_count"] == 1  # window fully drained by the gap
    det.update(e, TYPES)
    added = hist_n(rec) - n_before
    assert 1 <= added <= CFG.vel_short_min + 1


def test_century_gap_is_constant_time_and_prunes_long() -> None:
    det = new_det()
    last = _warm(det)
    rec = record(det)
    n_before = hist_n(rec)
    later = last + 100 * 365 * 24 * 60 * MIN  # ~52M minutes; a per-minute loop would hang
    det.update(env(later), TYPES)
    assert hist_n(rec) - n_before <= CFG.vel_short_min + 1
    assert rec.last_minute == later // MIN
    assert list(rec.long) == [later // MIN]


def test_out_of_order_timestamp_counts_as_current_minute() -> None:
    det = new_det()
    last = _warm(det, hours=1)
    rec = record(det)
    det.update(env(last), TYPES)
    before_count = sum(c for c, _ in rec.short)
    [s] = det.score(env(last - 30 * MIN), TYPES)  # earlier than last_minute
    assert s.evidence["window_count"] == before_count + 1
    det.update(env(last - 30 * MIN), TYPES)
    assert rec.last_minute == last // MIN  # never rewinds
    assert sum(c for c, _ in rec.short) == before_count + 1
    assert rec.long[last // MIN][0] == 3


def test_score_is_idempotent_and_pure() -> None:
    det = new_det()
    last = _warm(det)
    before = snapshot(det)
    e = env(last + 3 * MIN, {"amount": 41.0})  # pending virtual advance of 3 minutes
    first = det.score(e, TYPES)
    assert det.score(e, TYPES) == first
    assert snapshot(det) == before


def test_score_matches_post_advance_state() -> None:
    """The virtual advance in score() equals what update() then commits."""
    det = new_det()
    last = _warm(det)
    e = env(last + 4 * MIN + 30_000)
    for i in range(5):
        det.update(at(e, last + 4 * MIN + i), TYPES)
    probe = at(e, last + 7 * MIN)
    [virtual] = det.score(probe, TYPES)
    det.update(probe, TYPES)  # commits the same advance (+ this call)
    rec = record(det)
    assert virtual.evidence["window_count"] == sum(c for c, _ in rec.short)
    assert virtual.evidence["n_hist"] == hist_n(rec)


# -- money keys ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "money"),
    [
        ("amount", True),
        ("total_amount", True),
        ("refund.amount", True),
        ("items[].amount", True),
        ("total", True),
        ("account", False),
        ("discount", False),
        ("count", False),
        ("fee", False),
    ],
)
def test_money_keys(path: str, money: bool) -> None:
    det = new_det()
    det.update(env(T0, {path: 12.5}), TYPES)
    assert det._store.get(key_vel(AGENT, TOOL)) is not None
    assert (det._store.get(key_vel(AGENT, TOOL, path)) is not None) is money


def test_money_hint_adds_sum_key() -> None:
    det = new_det(hints={TOOL: ToolHint(field_hints={"fee": "money"})})
    det.update(env(T0, {"fee": 12.5}), TYPES)
    assert det._store.get(key_vel(AGENT, TOOL, "fee")) is not None


def test_sums_use_numeric_strings_and_skip_non_finite() -> None:
    det = new_det()
    for i, v in enumerate(["37.50", math.nan, math.inf, -math.inf, "abc", 1e308, 1e308]):
        det.update(env(T0 + i, {"amount": v}), TYPES)
    rec = record(det, "amount")
    count, total = rec.short[(T0 // MIN) % CFG.vel_short_min]
    assert count == 3  # "37.50" and the two 1e308s
    assert math.isfinite(total)
    assert record(det).short[(T0 // MIN) % CFG.vel_short_min][0] == 7  # every call counts
    orjson.dumps(rec.to_dict())  # JSON-finite
    [s] = det.score(env(T0 + 9, {"amount": math.nan}), TYPES)
    assert 0.0 <= s.score <= 1.0


# -- state bounds and snapshots ------------------------------------------------------------------


@pytest.fixture(scope="module")
def dense_week() -> VelocityDetector:
    """A call every minute, around the clock, for 7 days (worst case for the sparse long window)."""
    templates = [env(T0, {"amount": a}) for a in (37.51, 112.37, 9.99, 1234.56)]
    det = new_det()
    for i, ts in enumerate(cadence(T0, 7 * 24 * 60, 1)):
        det.update(at(templates[i % 4], ts), TYPES)
    return det


def test_long_window_is_bounded(dense_week: VelocityDetector) -> None:
    for path in (None, "amount"):
        assert len(record(dense_week, path).long) <= CFG.vel_long_min + 1


def test_state_bytes_per_key_under_30kb(dense_week: VelocityDetector) -> None:
    keys = list(dense_week._store.items())
    assert len(keys) == 2
    for _, rec in keys:
        assert len(orjson.dumps(rec.to_dict(), option=orjson.OPT_SORT_KEYS)) < 30_000


def test_snapshot_round_trip(dense_week: VelocityDetector, tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    dense_week._store.dump(path)
    restored = new_det()
    restored._store.load(path)
    assert snapshot(restored) == snapshot(dense_week)
    probe = env(T0 + 7 * 24 * 60 * MIN + 30_000, {"amount": 50.0})
    assert restored.score(probe, TYPES) == dense_week.score(probe, TYPES)


def test_from_dict_rejects_inconsistent_state() -> None:
    det = new_det()
    det.update(env(T0), TYPES)
    d = record(det).to_dict()
    with pytest.raises(ValueError):
        VelocityRecord.from_dict({**d, "short_count": []})


def test_unused_history_and_sums_are_not_kept() -> None:
    det = new_det()
    det.update(env(T0, {"amount": 12.5}), TYPES)
    count_rec, money_rec = record(det), record(det, "amount")
    assert count_rec.hist_count is not None and count_rec.hist_sum is None
    assert money_rec.hist_count is None and money_rec.hist_sum is not None
    cd, md = count_rec.to_dict(), money_rec.to_dict()
    assert "short_sum" not in cd and "long_sum" not in cd
    assert "short_sum" in md and "long_sum" in md
    for rec, d in ((count_rec, cd), (money_rec, md)):
        assert VelocityRecord.from_dict(orjson.loads(orjson.dumps(d))).to_dict() == d
        assert rec == VelocityRecord.from_dict(d)


def test_from_dict_requires_exactly_one_history() -> None:
    det = new_det()
    det.update(env(T0, {"amount": 12.5}), TYPES)
    cd, md = record(det).to_dict(), record(det, "amount").to_dict()
    with pytest.raises(ValueError):
        VelocityRecord.from_dict({**cd, "hist_sum": md["hist_sum"]})
    with pytest.raises(ValueError):
        VelocityRecord.from_dict({**md, "hist_sum": None})


def test_nonpositive_long_window_does_not_raise() -> None:
    det = VelocityDetector(DictStore(), CFG.replace(vel_long_min=0))
    for i in range(3):
        det.update(env(T0 + i * MIN), TYPES)
    assert len(record(det).long) <= 1


def _roll_reference(ring: list[tuple[int, float]], last: int, to: int) -> list[tuple[int, float]]:
    """Brute force: recompute the window total from scratch at every step."""
    w, out, m = len(ring), [], last
    while m < to and sum(c for c, _ in ring) > 0:
        total = sum(s for _, s in ring)
        out.append(
            (sum(c for c, _ in ring), max(-sys.float_info.max, min(sys.float_info.max, total)))
        )
        m += 1
        ring[m % w] = (0, 0.0)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_roll_running_total_matches_recompute(seed: int) -> None:
    rng = np.random.default_rng(seed)
    w = CFG.vel_short_min
    ring = [
        (int(c), round(float(rng.uniform(1, 500)) * int(c), 2)) if c else (0, 0.0)
        for c in rng.integers(0, 4, w)
    ]
    if seed == 0:  # overflowing buckets must not leave a stale infinite running total
        ring[3], ring[4] = (1, sys.float_info.max), (1, sys.float_info.max)
    last = int(rng.integers(1_000, 2_000))
    to = last + int(rng.integers(0, 2 * w))
    expected = _roll_reference(list(ring), last, to)
    got_ring = list(ring)
    samples, (cnt, total) = _roll(got_ring, last, to, with_sum=True)
    assert [c for c, _ in samples] == [c for c, _ in expected]
    for (_, a), (_, b) in zip(samples, expected, strict=True):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-6)
    assert cnt == sum(c for c, _ in got_ring)
    assert total == pytest.approx(min(sum(s for _, s in got_ring), sys.float_info.max), abs=1e-6)
    count_only, (cnt2, total2) = _roll(list(ring), last, to, with_sum=False)
    assert [c for c, _ in count_only] == [c for c, _ in expected] and total2 == 0.0
    assert cnt2 == cnt


# -- Task 1.14 sweep switch ----------------------------------------------------------------------


def test_sum_signal_off_scores_count_only_and_keeps_no_money_keys() -> None:
    det = VelocityDetector(DictStore(), dataclasses.replace(CFG, vel_sum_signal=False))
    rng = np.random.default_rng(5)
    for ts in cadence(T0, 2 * 24 * 60, 5):
        step(det, env(ts, {"amount": round(float(rng.uniform(30.0, 45.0)), 2)}))
    huge = step(det, env(T0 + 2 * 24 * 60 * MIN + 30_000, {"amount": 1e7}))
    assert huge.evidence["signal"] == "count" and huge.field is None
    assert huge.score < 0.5  # one call at normal cadence; the money total is not a signal
    assert det._store.get(key_vel(AGENT, TOOL, "amount")) is None
