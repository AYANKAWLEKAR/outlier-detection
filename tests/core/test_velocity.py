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
from hypothesis import given
from hypothesis import strategies as st

from agentfw.core.config import EngineConfig
from agentfw.core.detectors.base import squash
from agentfw.core.detectors.velocity import VelocityDetector, VelocityRecord, _count_z, _roll
from agentfw.core.engine import Engine
from agentfw.core.envelope import ActionEnvelope, DetectorScore
from agentfw.core.hints import ToolHint
from agentfw.core.schema_infer import TypeInference, TypeView
from agentfw.core.state import DictStore, key_vel
from agentfw.core.stats.p2 import P2Quantiles
from tests.factories import make_env

AGENT = "a1"
TOOL = "stripe.create_refund"
MIN = 60_000
T0 = 1_700_000_000_000 - 1_700_000_000_000 % (24 * 60 * MIN)  # midnight UTC
# Spec §6.4: refused calls are not counted. vel_count_attempts (default on by user decision,
# decision 008) is pinned off here and switched on explicitly by the tests that cover it.
CFG = EngineConfig().replace(vel_count_attempts=False)
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


# -- vel_count_attempts (default on, user decision): refused attempts count in the window only --


def test_refused_attempts_count_in_the_window_but_not_in_history() -> None:
    det = VelocityDetector(DictStore(), CFG.replace(vel_count_attempts=True))
    last = _warm(det)
    e = env(T0)
    n0 = hist_n(record(det))
    [base] = det.score(at(e, last + MIN), TYPES)
    for i in range(12):  # refused (blocked / denied) attempts within the same minute
        det.observe_refused(at(e, last + MIN + i * 1_000))
    [s] = det.score(at(e, last + MIN + 20_000), TYPES)
    assert ev_int(s, "window_count") == ev_int(base, "window_count") + 12
    assert s.score > base.score
    # the baseline never sees them: after the window has rolled past, history is as without them
    ref = new_det()
    _warm(ref)
    later = at(e, last + 30 * MIN)
    det.update(later, TYPES)
    ref.update(later, TYPES)
    assert hist_n(record(det)) == hist_n(record(ref)) > n0
    assert record(det).hist_count.to_dict() == record(ref).hist_count.to_dict()  # type: ignore[union-attr]
    [a] = det.score(at(e, last + 31 * MIN), TYPES)
    [b] = ref.score(at(e, last + 31 * MIN), TYPES)
    assert a == b  # refused attempts older than the short window are forgotten
    assert record(det).to_dict() == record(ref).to_dict()


def test_refused_attempts_ignored_when_switch_off_or_key_cold() -> None:
    off = new_det()
    last = _warm(off)
    before = snapshot(off)
    off.observe_refused(at(env(T0), last + MIN))
    assert snapshot(off) == before
    on = VelocityDetector(DictStore(), CFG.replace(vel_count_attempts=True))
    on.observe_refused(env(T0))  # no record yet: nothing to count against, nothing created
    assert snapshot(on) == orjson.dumps([], option=orjson.OPT_SORT_KEYS)


def test_refused_attempts_roundtrip_through_to_dict() -> None:
    det = VelocityDetector(DictStore(), CFG.replace(vel_count_attempts=True))
    last = _warm(det)
    det.observe_refused(at(env(T0), last + MIN))
    rec = record(det)
    assert VelocityRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()
    assert "refused" not in record(_warm_det()).to_dict()  # absent unless used: goldens unchanged


def _warm_det() -> VelocityDetector:
    det = new_det()
    _warm(det)
    return det


# -- Amendment A1.1: the count signal as a Poisson tail deviate (vel_count_stat) ----------------

POISSON = EngineConfig().replace(vel_count_stat="poisson")
LOGZ = EngineConfig().replace(vel_count_stat="logz")
EVIDENCE_KEYS = {"signal", "window_count", "baseline_median", "n_hist", "z", "cold"}
# Every key a velocity record or its P² history had before the amendment (A1.1: no state changes).
PRE_A1_COUNT_KEYS = {
    "short_count",
    "last_minute",
    "long_ago",
    "long_count",
    "hist_count",
    "hist_sum",
}
PRE_A1_MONEY_KEYS = PRE_A1_COUNT_KEYS | {"short_sum", "long_sum"}
PRE_A1_P2_KEYS = {"probs", "n", "q", "pos", "buf"}
NO_MONEY = {"customer": "cus_1"}  # no money field: only the count key exists


def flat_key(cfg: EngineConfig) -> tuple[VelocityDetector, ActionEnvelope, int]:
    """One call every 15 minutes for 3 days: every 10-minute window holds exactly one call.

    Returns (detector, call template, a timestamp 15 minutes after the last call: empty window).
    """
    det = VelocityDetector(DictStore(), cfg)
    e = env(T0, NO_MONEY)
    last = 0
    for last in cadence(T0, 3 * 24 * 60, 15):
        det.update(at(e, last), TYPES)
    return det, e, last + 15 * MIN


def burst(cfg: EngineConfig, n: int) -> DetectorScore:
    """The ``n``-th of ``n`` calls within one minute on the flat key (the first n-1 allowed)."""
    det, e, start = flat_key(cfg)
    for i in range(n - 1):
        det.update(at(e, start + i * 1_000), TYPES)
    [s] = det.score(at(e, start + (n - 1) * 1_000), TYPES)
    assert s.evidence["window_count"] == n
    return s


def burst_score(cfg: EngineConfig, n: int) -> float:
    return burst(cfg, n).score


N_WRITTEN = 401


def key_with_history(cfg: EngineConfig, heights: list[float]) -> VelocityDetector:
    """A count key (empty window at T0) whose P² history is written directly.

    ``heights`` are the markers (min, q25, q50, q75, max) of ``ln(1 + window count)``, placed at
    their ideal positions ``1 + (n - 1) p``, so each quartile is its marker height exactly.
    """
    hist = P2Quantiles.from_dict(
        {
            "probs": [0.25, 0.5, 0.75],
            "n": N_WRITTEN,
            "q": heights,
            "pos": [1 + i * (N_WRITTEN - 1) // 4 for i in range(5)],
            "buf": [],
        }
    )
    assert [hist.quantile(p) for p in (0.25, 0.5, 0.75)] == heights[1:4]
    det = VelocityDetector(DictStore(), cfg)
    rec = VelocityRecord.new(cfg, T0 // MIN, money=False)
    rec.hist_count = hist
    det._store.put(key_vel(AGENT, TOOL), rec)
    return det


def quartile_key_score(cfg: EngineConfig, c: int) -> float:
    """Score of the ``c``-th call in one minute on a key whose window-count quartiles are 2/3/5.

    The window is empty before the burst, so no sample is pending and the history is scored
    exactly as written.
    """
    det = key_with_history(cfg, [math.log1p(k) for k in (1, 2, 3, 5, 12)])
    e = env(T0, NO_MONEY)
    start = T0 + 30 * MIN
    for i in range(c - 1):
        det.update(at(e, start + i * 1_000), TYPES)
    [s] = det.score(at(e, start + (c - 1) * 1_000), TYPES)
    assert s.evidence["window_count"] == c and s.evidence["n_hist"] == N_WRITTEN
    return s.score


def hist_state(det: VelocityDetector) -> dict[str, Any]:
    h = record(det).hist_count
    assert h is not None
    return h.to_dict()


def count_quartiles(det: VelocityDetector) -> tuple[float, float, float]:
    """Quartiles of the committed window-count history, in calls."""
    h = record(det).hist_count
    assert h is not None
    k25, k50, k75 = (math.expm1(h.quantile(p)) for p in (0.25, 0.5, 0.75))
    return k25, k50, k75


def test_count_z_reference_values() -> None:
    assert _count_z(11, 1.0, 1.0, 1.0) == pytest.approx(5.375466, abs=1e-6)  # exact Poisson: 5.61
    assert _count_z(21, 3.0, 3.0, 3.0) == pytest.approx(6.488254, abs=1e-6)  # exact 6.68
    # disp = 1.1524 > 1
    assert _count_z(21, 2.0, 3.0, 5.0) == pytest.approx(6.488254 / 1.152411, abs=1e-3)


def test_count_z_edges() -> None:
    assert _count_z(0, 1.0, 1.0, 1.0) == 0.0 and _count_z(-3, 1.0, 1.0, 1.0) == 0.0
    # not above the rate
    assert _count_z(3, 3.0, 3.0, 3.0) == 0.0 and _count_z(2, 3.0, 3.0, 3.0) == 0.0
    assert _count_z(1, 1.0, 1.0, 1.0) == 0.0
    # rate floor: a median window of 0 calls never divides by zero
    assert math.isfinite(_count_z(5, 0.0, 0.0, 0.0)) and _count_z(5, 0.0, 0.0, 0.0) > 0.0
    # ... the floor is 1e-9 (A1.1), and a negative median is floored like a zero one
    floored = 3 * math.sqrt(5) * (1 - 1 / 45 - (1e-9 / 5) ** (1 / 3))
    assert _count_z(5, 0.0, 0.0, 0.0) == pytest.approx(floored, abs=1e-9)
    assert _count_z(5, 1.0, -1.0, 1.0) == _count_z(5, 1.0, 0.0, 1.0)
    # inverted quartiles: disp floors at 1
    assert _count_z(10, 5.0, 3.0, 1.0) == _count_z(10, 3.0, 3.0, 3.0)
    # Anscombe clamp A(k) = 2 sqrt(max(k, 0) + 3/8): a negative quartile counts as 0 calls
    assert _count_z(5, -1.0, 1.0, 1.0) == _count_z(5, 0.0, 1.0, 1.0)
    disp = (2 * math.sqrt(8 + 0.375) - 2 * math.sqrt(0.375)) / 1.349  # 3.383: the clamp matters
    assert _count_z(20, -1.0, 3.0, 8.0) == _count_z(20, 0.0, 3.0, 8.0)
    assert _count_z(20, -1.0, 3.0, 8.0) == pytest.approx(_count_z(20, 3.0, 3.0, 3.0) / disp)


@given(st.integers(1, 500), st.integers(1, 500), st.floats(0.0, 50.0))
def test_count_z_monotone_in_count(a: int, b: int, rate: float) -> None:
    lo, hi = sorted((a, b))
    assert _count_z(lo, rate, rate, rate) <= _count_z(hi, rate, rate, rate) + 1e-12


def test_poisson_flat_key_needs_ten_calls() -> None:
    # one call every 15 min for 3 days (every window holds 1 call), then n calls in one minute
    s6 = burst(POISSON, n=6)
    assert s6.score == pytest.approx(squash(3.168, 3.0, 0.75), abs=2e-3)  # 0.556
    assert burst_score(POISSON, n=10) > 0.9 and burst_score(POISSON, n=9) < 0.9
    logz6 = burst(LOGZ, n=6)
    assert logz6.score > 0.9  # §6.4 over-scores it
    # the evidence keeps its keys; ``z`` is the deviate, the baseline is still the median count
    assert set(s6.evidence) == EVIDENCE_KEYS == set(logz6.evidence)
    assert s6.field is None and s6.evidence["signal"] == "count" and s6.evidence["cold"] is False
    assert s6.evidence["z"] == pytest.approx(3.168, abs=2e-3)
    assert s6.evidence["baseline_median"] == pytest.approx(1.0)
    assert s6.evidence["baseline_median"] == logz6.evidence["baseline_median"]
    assert s6.evidence["n_hist"] == logz6.evidence["n_hist"]


def test_poisson_bursty_key_needs_eighteen_calls() -> None:
    # history whose window quartiles are 2 / 3 / 5 calls (helpdesk-triage-b update_status, 008 §1)
    assert quartile_key_score(POISSON, c=21) > 0.95  # 0.971
    assert quartile_key_score(LOGZ, c=21) == pytest.approx(0.604, abs=2e-3)  # §6.4 scored 0.604
    assert quartile_key_score(POISSON, c=19) > 0.9 > quartile_key_score(POISSON, c=15)
    # A1.1 reference table: with exact quartiles the first count scoring >= 0.90 is 18
    assert quartile_key_score(POISSON, c=18) >= 0.9 > quartile_key_score(POISSON, c=17)


def test_poisson_cold_key_scores_zero() -> None:
    det = VelocityDetector(DictStore(), POISSON)
    e = env(T0, NO_MONEY)
    [s] = det.score(e, TYPES)  # no record at all
    assert s.score == 0.0 and s.evidence["cold"] is True and s.evidence["z"] is None
    assert s.evidence["n_hist"] == 0
    for ts in cadence(T0, 20, 1):  # 20 active minutes: fewer than vel_cold_n minute-samples
        det.update(at(e, ts), TYPES)
    for i in range(30):  # then a burst that would score ~1.0 on a warm key
        det.update(at(e, T0 + 21 * MIN + i), TYPES)
    [s] = det.score(at(e, T0 + 21 * MIN + 40), TYPES)
    assert 0 < ev_int(s, "n_hist") < POISSON.vel_cold_n
    assert ev_int(s, "window_count") > 30
    assert s.score == 0.0 and s.evidence["cold"] is True and s.evidence["z"] is None
    assert set(s.evidence) == EVIDENCE_KEYS


def test_poisson_history_heights_beyond_expm1_range_do_not_raise() -> None:
    # A state file may hold any finite marker height: exp() of it must saturate, never raise.
    det = key_with_history(POISSON, [0.0, 800.0, 900.0, 1000.0, 1e308])
    [s] = det.score(env(T0 + 30 * MIN, NO_MONEY), TYPES)
    assert s.evidence["z"] == 0.0 and s.evidence["cold"] is False  # 1 call, far below that rate
    assert s.score == squash(0.0, 3.0, 0.75)
    assert s.evidence["baseline_median"] == sys.float_info.max


def test_poisson_negative_history_heights_do_not_raise() -> None:
    # A count history never holds a negative height, but a state file may (P2Quantiles.from_dict
    # accepts any finite one). expm1 of it is a negative count: the rate floors at 1e-9 and the
    # Anscombe transform clamps at 0, so the call is scored, not refused with an exception.
    det = key_with_history(POISSON, [-3.0, -2.0, -1.0, 0.5, 1.0])
    [s] = det.score(env(T0 + 30 * MIN, NO_MONEY), TYPES)
    assert s.evidence["cold"] is False and s.evidence["window_count"] == 1
    # c = 1 at rate 1e-9 (cube root 1e-3); disp = 1 (the Anscombe IQR of 0 .. 0.65 calls is 0.59)
    assert s.evidence["z"] == pytest.approx(3 * (1 - 1 / 9 - 1e-3), abs=1e-9)
    assert s.evidence["z"] == _count_z(1, 0.0, 0.0, math.expm1(0.5))
    assert s.score == pytest.approx(squash(2.663667, 3.0, 0.75), abs=1e-6)  # 0.390


def test_poisson_folds_pending_minute_samples_into_the_quartiles() -> None:
    # A1.1: "pending minute-samples folded in on a copy, as today". A short history (41 samples
    # of about 2 calls), an 8-call burst in one minute, then one call 8 minutes later: the 8
    # minute-samples the burst left pending move every quartile, and z must read the moved ones.
    det = VelocityDetector(DictStore(), POISSON)
    e = env(T0, NO_MONEY)
    last = 0
    for last in cadence(T0, 45, 5):
        det.update(at(e, last), TYPES)
    for i in range(8):
        det.update(at(e, last + MIN + i * 1_000), TYPES)
    committed, n_committed = count_quartiles(det), hist_n(record(det))
    late = at(e, last + 9 * MIN)
    before = snapshot(det)
    [s] = det.score(late, TYPES)
    assert snapshot(det) == before  # folded in on a copy
    det.update(late, TYPES)  # commits exactly the samples score() folded in
    folded = count_quartiles(det)
    assert ev_int(s, "window_count") == 10  # the call 5 min before the burst, the burst, this one
    assert ev_int(s, "n_hist") == hist_n(record(det)) == n_committed + 8
    assert all(f > c + 0.04 for f, c in zip(folded, committed, strict=True))  # every quartile moved
    assert s.evidence["cold"] is False
    assert s.evidence["z"] == pytest.approx(_count_z(10, *folded), abs=1e-12)
    assert s.evidence["baseline_median"] == pytest.approx(folded[1], abs=1e-12)
    # the history without the pending samples gives a visibly different deviate
    assert _count_z(10, *committed) > _count_z(10, *folded) + 0.1


@pytest.mark.parametrize("value", ["Poisson", "poison", "", None, 1])
def test_unknown_vel_count_stat_rejected(value: object) -> None:
    # A mistyped switch must not silently run the other rule (like cal_censor in Engine).
    cfg = EngineConfig().replace(vel_count_stat=value)
    with pytest.raises(ValueError, match=r"unknown vel_count_stat .*known: \('logz', 'poisson'\)"):
        VelocityDetector(DictStore(), cfg)
    with pytest.raises(ValueError, match="vel_count_stat"):
        Engine(cfg)  # also with the detector disabled: the engine always builds it
    with pytest.raises(ValueError, match="vel_count_stat"):
        Engine(cfg, enabled=frozenset({"magnitude"}))


def test_poisson_sum_signal_is_unchanged() -> None:
    scores: list[DetectorScore] = []
    for cfg in (LOGZ, POISSON):
        det = VelocityDetector(DictStore(), cfg)
        last = _warm(det, amount=37.5)  # every window: 2 calls, 75.0 of money
        [s] = det.score(env(last + 5 * MIN, {"amount": 167.5}), TYPES)
        assert s.field == "amount" and s.evidence["signal"] == "sum"
        assert s.evidence["window_count"] == 2  # a normal count: only the money total is high
        scores.append(s)
    logz, poisson = scores
    assert 0.5 < logz.score < 0.99  # not saturated, so equality below is a real comparison
    assert poisson.score == logz.score  # bit-equal: the money sum keeps the §6.4 rule
    assert poisson == logz


def test_poisson_refused_attempts_count() -> None:
    assert POISSON.vel_count_attempts is True
    det, e, start = flat_key(POISSON)
    hist_before = hist_state(det)
    [base] = det.score(at(e, start), TYPES)
    assert ev_int(base, "window_count") == 1 and base.score < 0.1
    for i in range(9):  # nine refused (blocked / denied) attempts within the minute
        det.observe_refused(at(e, start + i * 1_000))
    [s] = det.score(at(e, start + 9_000), TYPES)
    assert ev_int(s, "window_count") == 10  # refused attempts raise c ...
    assert s.score > 0.9
    assert s.score == pytest.approx(burst_score(POISSON, n=10), abs=1e-9)
    assert s.evidence["z"] == pytest.approx(_count_z(10, 1.0, 1.0, 1.0), abs=1e-9)
    assert hist_state(det) == hist_before  # ... and never the history
    assert s.evidence["n_hist"] == base.evidence["n_hist"]
    assert s.evidence["baseline_median"] == base.evidence["baseline_median"]
    off, e, start = flat_key(POISSON.replace(vel_count_attempts=False))
    for i in range(9):
        off.observe_refused(at(e, start + i * 1_000))
    assert off.score(at(e, start + 9_000), TYPES) == [base]


def test_poisson_score_is_pure_and_idempotent(tmp_path: Path) -> None:
    det = VelocityDetector(DictStore(), POISSON)
    last = _warm(det)  # a count key and a money key
    for i in range(3):
        det.observe_refused(at(env(T0), last + MIN + i))
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    det._store.dump(before)
    e = env(last + 3 * MIN, {"amount": 41.0})  # pending virtual advance of 3 minutes
    first = det.score(e, TYPES)
    assert det.score(e, TYPES) == first
    det._store.dump(after)
    assert after.read_bytes() == before.read_bytes()
    assert first[0].evidence["cold"] is False
    assert first[0].evidence["window_count"] == 6  # 2 allowed calls in the window + 3 refused + 1


def test_pre_a1_velocity_state_scores_immediately(tmp_path: Path) -> None:
    old = VelocityDetector(DictStore(), LOGZ)  # the §6.4 rule: the engine before the amendment
    last = _warm(old)
    path, again = tmp_path / "pre-a1.jsonl", tmp_path / "reloaded.jsonl"
    old._store.dump(path)
    new = VelocityDetector(DictStore(), POISSON)
    new._store.load(path)
    new._store.dump(again)
    assert again.read_bytes() == path.read_bytes()  # to_dict() is byte-equal to the dump
    rows = {r["key"]: r["data"] for r in map(orjson.loads, path.read_bytes().splitlines())}
    count, money = rows[key_vel(AGENT, TOOL)], rows[key_vel(AGENT, TOOL, "amount")]
    assert len(rows) == 2  # no key the old code lacks
    assert set(count) == PRE_A1_COUNT_KEYS and set(count["hist_count"]) == PRE_A1_P2_KEYS
    assert set(money) == PRE_A1_MONEY_KEYS and set(money["hist_sum"]) == PRE_A1_P2_KEYS
    e = env(T0, NO_MONEY)
    start = last + 15 * MIN
    [first] = new.score(at(e, start), TYPES)  # the first call after loading is scored warm
    assert first.evidence["cold"] is False and first.evidence["z"] == 0.0
    scores = [step(new, at(e, start + i * 1_000)) for i in range(25)]
    assert ev_int(scores[-1], "window_count") == 25
    assert scores[-1].score > 0.99 and scores[-1].evidence["signal"] == "count"
    # scored by the new rule from the old history (every warm window held 2 calls)
    assert scores[-1].evidence["z"] == pytest.approx(_count_z(25, 2.0, 2.0, 2.0), abs=1e-2)


def test_default_is_still_logz() -> None:
    assert EngineConfig().vel_count_stat == "logz"
