from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import orjson
import pytest
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from agentfw.bench.metrics import (
    GATE_BENIGN_FPR,
    GATE_FAMILIES,
    GATE_REASON_MIN,
    GATE_SHARE,
    STRENGTH_BUCKETS,
    among_reasons_share,
    auprc,
    benign_fpr,
    bootstrap_ci,
    cold_start_curve,
    compute_all,
    control_fpr,
    drift_delay,
    fpr_at_recall,
    gate_preview,
    latency_summary,
    per_agent_fpr_at_recall,
    recall_by_family,
    top_reason_share,
)
from agentfw.bench.replay import ScoreRow
from agentfw.core.config import EngineConfig

_ids = itertools.count()


def _row(**kw: Any) -> ScoreRow:
    """ScoreRow factory: a warm, test-split benign row unless overridden."""
    i = next(_ids)
    base: dict[str, Any] = {
        "envelope_id": f"e{i:06d}",
        "session_id": f"s{i}",
        "agent_id": "A",
        "tool_id": "stripe.create_refund",
        "ts_unix_ms": 1_000 + i,
        "split": "test",
        "fused_score": 0.0,
        "per_detector_json": "[]",
        "label": "benign",
        "attack_family": None,
        "attack_id": None,
        "injection_level": None,
        "strength": None,
        "step": None,
        "control_type": None,
        "is_warm": True,
        "n_key_obs": 100,
        "latency_us": 10.0,
        "threshold": None,
    }
    base.update(kw)
    if base["label"] == "attack":
        if base["attack_family"] is None:
            base["attack_family"] = "magnitude_spike"
        if base["attack_id"] is None:
            base["attack_id"] = f"atk-{base['envelope_id']}"
    return ScoreRow(**base)


def _atk(score: float, **kw: Any) -> ScoreRow:
    return _row(label="attack", fused_score=score, **kw)


def _per(*pairs: tuple[str, float]) -> str:
    return orjson.dumps(
        [{"detector": d, "field": None, "score": s, "evidence": {}} for d, s in pairs]
    ).decode()


# --- AUPRC / global operating point ------------------------------------------------------------


def test_perfect_scores() -> None:
    y = np.array([0, 0, 1, 0, 1], bool)
    s = y.astype(float)
    assert auprc(s, y) == 1.0 and fpr_at_recall(s, y)[1] == 0.0


def test_random_scores_auprc_near_base_rate() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(20_000) < 0.05
    assert abs(auprc(rng.random(20_000), y) - y.mean()) < 0.01


def test_all_tied_scores_equal_base_rate() -> None:
    y = np.array([1, 0, 0, 0], bool)
    assert auprc(np.zeros(4), y) == 0.25


def test_auprc_matches_sklearn_reference_with_ties() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(5, 400))
        y = rng.random(n) < 0.2
        y[0] = True
        s = rng.integers(0, 8, n).astype(float) + 0.5 * y  # heavy ties, some signal
        assert auprc(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_auprc_handles_known_tie_example() -> None:
    # Groups: {0.9: +}, {0.5: +,-,-}, {0.1: +}. AP = 1/3*1 + 1/3*(2/4) + 1/3*(3/5).
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 1, 0, 0, 1], bool)
    assert auprc(s, y) == pytest.approx((1 + 0.5 + 0.6) / 3)


def test_auprc_and_fpr_ignore_nan_scores() -> None:
    s = np.array([0.9, np.nan, 0.1, np.nan])
    y = np.array([1, 0, 0, 1], bool)
    assert auprc(s, y) == 1.0
    assert fpr_at_recall(s, y) == (0.9, 0.0)


def test_auprc_without_positives_is_nan() -> None:
    assert math.isnan(auprc(np.array([0.1, 0.2]), np.array([0, 0], bool)))


def test_operating_point_is_highest_threshold_meeting_recall() -> None:
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    y = np.array([1, 1, 0, 1, 0], bool)
    assert fpr_at_recall(s, y, 0.95) == (0.6, 0.5)


def test_operating_point_flags_whole_tie_group() -> None:
    # Recall 0.5 is reached at 0.8; recall 1.0 needs 0.4, where both tied benigns are flagged too.
    s = np.array([0.8, 0.4, 0.4, 0.4, 0.1])
    y = np.array([1, 1, 0, 0, 0], bool)
    assert fpr_at_recall(s, y, 0.5) == (0.8, 0.0)
    thr, fpr = fpr_at_recall(s, y, 0.95)
    assert thr == 0.4 and fpr == pytest.approx(2 / 3)


def test_operating_point_exact_recall_boundary() -> None:
    # 19 of 20 attacks above 0.5 -> recall exactly 0.95 at threshold 0.5.
    s = np.concatenate([np.full(19, 0.5), [0.0], [0.2, 0.6]])
    y = np.concatenate([np.ones(20, bool), np.zeros(2, bool)])
    assert fpr_at_recall(s, y, 0.95) == (0.5, 0.5)


# --- thresholds, family/strength recall ---------------------------------------------------------


def test_threshold_modes_float_mapping_and_row() -> None:
    rows = [
        _row(agent_id="A", fused_score=0.5, threshold=0.4),
        _row(agent_id="A", fused_score=0.3, threshold=None),
        _row(agent_id="B", fused_score=0.9, threshold=0.95),
        _row(agent_id="Z", fused_score=0.99, threshold=0.1),
        _row(agent_id="A", fused_score=float("nan"), threshold=0.0),
    ]
    assert benign_fpr(rows, 0.5) == pytest.approx(3 / 4)  # >= is flagged; NaN row excluded
    # Mapping: agent Z missing -> never flagged.
    assert benign_fpr(rows, {"A": 0.3, "B": 0.95}) == pytest.approx(2 / 4)
    # "row": A(0.5>=0.4) flagged, None not flagged, B not, Z flagged.
    assert benign_fpr(rows, "row") == pytest.approx(2 / 4)


def test_attack_level_recall_counts_any_envelope() -> None:
    rows = [
        _atk(0.9, attack_id="a1", ts_unix_ms=1, strength=20.0),
        _atk(0.1, attack_id="a1", ts_unix_ms=2, strength=20.0),
        _atk(0.1, attack_id="a1", ts_unix_ms=3, strength=20.0),
        _atk(0.2, attack_id="a2", ts_unix_ms=4, strength=5.0),
        _row(fused_score=0.0),
    ]
    env = recall_by_family(rows, 0.5, "envelope")["magnitude_spike"]
    atk = recall_by_family(rows, 0.5, "attack")["magnitude_spike"]
    assert env["all"].recall == pytest.approx(1 / 4)
    assert (env["all"].n_detected, env["all"].n_total) == (1, 4)
    assert atk["all"].recall == pytest.approx(1 / 2)
    assert (atk["all"].n_detected, atk["all"].n_total) == (1, 2)
    assert atk["10-100"].recall == 1.0 and atk["3-10"].recall == 0.0


def test_attack_family_and_strength_from_first_row_by_ts() -> None:
    rows = [
        _atk(0.9, attack_id="a1", ts_unix_ms=20, strength=50.0),
        _atk(0.1, attack_id="a1", ts_unix_ms=10, strength=4.0),
    ]
    cells = recall_by_family(rows, 0.5, "attack")["magnitude_spike"]
    assert cells["3-10"].n_total == 1 and cells["3-10"].recall == 1.0
    assert cells["10-100"].n_total == 0 and math.isnan(cells["10-100"].recall)


def test_strength_bucket_keys_and_edges() -> None:
    assert STRENGTH_BUCKETS["magnitude_spike"] == (3, 10, 100)
    rows = [
        _atk(0.9, strength=3.0),  # lower edge -> 3-10
        _atk(0.9, strength=10.0),  # inner edge belongs to the upper bucket
        _atk(0.9, strength=100.0),  # last bucket is upper-inclusive
        _atk(0.9, strength=2.5),  # below range -> clamped into first bucket
        _atk(0.9, strength=140.0),  # above range -> clamped into last bucket
        _atk(0.9, strength=None),  # no strength -> only "all"
        _atk(0.9, attack_family="slow_drift", strength=0.1),
        _atk(0.9, attack_family="novel_target", strength=None),
    ]
    out = recall_by_family(rows, 0.5, "envelope")
    ms = out["magnitude_spike"]
    assert set(ms) == {"3-10", "10-100", "all"}
    assert ms["3-10"].n_total == 2 and ms["10-100"].n_total == 3 and ms["all"].n_total == 6
    assert set(out["slow_drift"]) == {"0.03-0.06", "0.06-0.1", "all"}
    assert out["slow_drift"]["0.06-0.1"].n_total == 1
    assert set(out["novel_target"]) == {"all"}


def test_benign_and_control_fpr() -> None:
    rows = [
        _row(fused_score=0.9, control_type="benign_unusual"),
        _row(fused_score=0.1, control_type="benign_unusual"),
        _row(fused_score=0.1),
        _row(fused_score=0.1),
        _atk(0.9),
    ]
    assert benign_fpr(rows, 0.5) == pytest.approx(1 / 4)
    assert control_fpr(rows, 0.5) == pytest.approx(1 / 2)
    assert math.isnan(control_fpr(rows[2:], 0.5))


def test_top_reason_share() -> None:
    rows = [
        # a1 detected; its highest-scoring flagged row's top detector is magnitude.
        _atk(0.9, attack_id="a1", per_detector_json=_per(("magnitude", 0.9), ("velocity", 0.2))),
        _atk(0.2, attack_id="a1", per_detector_json=_per(("novelty", 0.8))),
        # a2 detected via velocity.
        _atk(0.7, attack_id="a2", per_detector_json=_per(("magnitude", 0.1), ("velocity", 0.7))),
        # a3 undetected -> excluded.
        _atk(0.1, attack_id="a3", per_detector_json=_per(("magnitude", 0.1))),
        # other family -> excluded.
        _atk(0.9, attack_id="a4", attack_family="novel_target", per_detector_json=_per(("x", 1))),
    ]
    assert top_reason_share(rows, "magnitude_spike", "magnitude", 0.5) == pytest.approx(0.5)
    assert top_reason_share(rows, "magnitude_spike", "velocity", 0.5) == pytest.approx(0.5)
    assert math.isnan(top_reason_share(rows, "magnitude_spike", "magnitude", 0.95))


def test_top_reason_share_skips_null_and_non_numeric_scores() -> None:
    bad = '[{"detector": "a", "score": null}, {"detector": "b", "score": "x"}, '
    rows = [
        _atk(0.9, attack_id="a1", per_detector_json=bad + '{"detector": "c", "score": 0.4}]'),
        _atk(0.9, attack_id="a2", per_detector_json='[{"detector": "a", "score": null}]'),
    ]
    assert top_reason_share(rows, "magnitude_spike", "c", 0.5) == 0.5
    assert top_reason_share(rows, "magnitude_spike", "a", 0.5) == 0.0


# --- among-the-reasons share (spec A1.6, D-027) ---------------------------------------------------


def _ra(aid: str, fused: float, *pairs: tuple[str, float], **kw: Any) -> ScoreRow:
    """An attack row with live threshold 0.9 (``"row"``) and the given per-detector scores."""
    kw.setdefault("attack_family", "velocity_burst")
    kw.setdefault("threshold", 0.9)
    return _atk(fused, attack_id=aid, per_detector_json=_per(*pairs), **kw)


def test_among_reasons_counts_a_targeted_reason_on_any_flagged_row() -> None:
    rows = [
        _ra("A1", 0.95, ("sequence", 0.95), ("velocity", 0.31)),  # flagged, velocity a reason
        _ra("A2", 0.95, ("sequence", 0.95), ("velocity", 0.30)),  # 0.30 is not > reason_min
        _ra("A3", 0.50, ("velocity", 0.50)),  # the reason sits only on an unflagged row
        _ra("A3", 0.95, ("sequence", 0.95)),
        _ra("A4", 0.10, ("velocity", 0.10)),  # not detected
        _ra("B1", 0.95, ("velocity", 0.95), attack_family="novel_target"),  # other family
    ]
    assert among_reasons_share(rows, "velocity_burst", "velocity", "row") == pytest.approx(1 / 3)
    assert top_reason_share(rows, "velocity_burst", "velocity", "row") == 0.0
    # reason_min is the strict lower bound: lowering it admits A2's 0.30.
    got = among_reasons_share(rows, "velocity_burst", "velocity", "row", reason_min=0.2)
    assert got == pytest.approx(2 / 3)


def test_among_reasons_respects_the_gate_bucket() -> None:
    # magnitude spikes with strength 5 (attributed) and 20 (not attributed), both detected
    rows = [
        _ra("M5", 0.95, ("magnitude", 0.95), attack_family="magnitude_spike", strength=5.0),
        _ra("M20", 0.95, ("sequence", 0.95), ("magnitude", 0.2),
            attack_family="magnitude_spike", strength=20.0),
    ]  # fmt: skip
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row", bucket="10-100") == 0.0
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row", bucket="3-10") == 1.0
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row") == 0.5


def test_among_reasons_takes_family_and_strength_from_the_first_row() -> None:
    rows = [  # same attack: the later row's strength (20) must not move it into 10-100
        _ra("M", 0.95, ("magnitude", 0.95), attack_family="magnitude_spike", strength=5.0,
            ts_unix_ms=1),
        _ra("M", 0.95, ("magnitude", 0.95), attack_family="magnitude_spike", strength=20.0,
            ts_unix_ms=2),
    ]  # fmt: skip
    assert math.isnan(
        among_reasons_share(rows, "magnitude_spike", "magnitude", "row", bucket="10-100")
    )
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row", bucket="3-10") == 1.0


def test_among_reasons_nan_without_detection() -> None:
    rows = [
        _ra("A1", 0.10, ("velocity", 0.95)),  # targeted reason but never flagged
        _ra("A2", math.nan, ("velocity", 0.95), threshold=0.0),  # unscored rows are excluded
    ]
    assert math.isnan(among_reasons_share(rows, "velocity_burst", "velocity", "row"))
    assert math.isnan(among_reasons_share([], "velocity_burst", "velocity", "row"))
    assert math.isnan(among_reasons_share(rows, "novel_target", "novelty", "row"))


def test_among_reasons_skips_null_and_non_numeric_scores() -> None:
    bad = (
        '[{"detector": "velocity", "score": null}, {"detector": "velocity", "score": "0.9"}, '
        '{"detector": "velocity", "score": true}, {"detector": "velocity"}]'
    )
    rows = [
        _atk(0.95, attack_id="A1", attack_family="velocity_burst", threshold=0.9,
             per_detector_json=bad),
        _ra("A2", 0.95, ("velocity", 0.9)),
    ]  # fmt: skip
    assert among_reasons_share(rows, "velocity_burst", "velocity", "row") == 0.5


def test_gate_reason_min_matches_engine_config() -> None:
    assert GATE_REASON_MIN == EngineConfig().reason_min == 0.3
    assert (GATE_SHARE, GATE_BENIGN_FPR) == (0.90, 0.02)


# --- gate_preview: the amended criterion (spec A1.7) --------------------------------------------


def _gate_atk(family: str, *pairs: tuple[str, float], fused: float = 0.9) -> ScoreRow:
    return _atk(
        fused,
        attack_id=f"{family}-{next(_ids)}",
        attack_family=family,
        strength=20.0 if family == "magnitude_spike" else None,
        threshold=0.5,
        per_detector_json=_per(*pairs),
    )


def _gate_rows(
    n_benign: int = 100, n_flagged: int = 1, **families: list[ScoreRow]
) -> list[ScoreRow]:
    """One attributed, detected attack per gated family (unless replaced) plus benign rows."""
    rows: list[ScoreRow] = []
    for fam, (_, det) in GATE_FAMILIES.items():
        rows += families.get(fam, [_gate_atk(fam, (det, 0.9))])
    rows += [
        _row(fused_score=0.9 if i < n_flagged else 0.1, threshold=0.5) for i in range(n_benign)
    ]
    return rows


def test_gate_preview_passed_is_the_amended_criterion() -> None:
    g = gate_preview(_gate_rows())
    assert g.passed is True and g.failures == () and g.missing == ()
    assert all(c.among_reasons_share == 1.0 for c in g.cells.values())

    exfil = [_gate_atk("exfil_volume", ("magnitude", 0.9)) for _ in range(8)]
    exfil.append(_gate_atk("exfil_volume", ("sequence", 0.95), ("magnitude", 0.3)))
    g = gate_preview(_gate_rows(exfil_volume=exfil))
    assert g.cells["exfil_volume"].recall == 1.0
    assert g.cells["exfil_volume"].among_reasons_share == pytest.approx(8 / 9)
    assert g.passed is False and g.failures == ("among:exfil_volume",)

    g = gate_preview(_gate_rows(n_flagged=3))
    assert g.benign_fpr == pytest.approx(0.03)
    assert g.passed is False and g.failures == ("benign_fpr",)

    vel = [_gate_atk("velocity_burst", ("velocity", 0.9)) for _ in range(3)]
    vel += [_gate_atk("velocity_burst", ("sequence", 0.95), ("velocity", 0.5)) for _ in range(7)]
    g = gate_preview(_gate_rows(velocity_burst=vel))
    assert g.passed is True  # top-1 is reported, not gated
    assert g.cells["velocity_burst"].top_reason_share == pytest.approx(0.3)
    assert g.cells["velocity_burst"].among_reasons_share == 1.0


def test_gate_preview_thresholds_are_inclusive() -> None:
    vel = [_gate_atk("velocity_burst", ("velocity", 0.9)) for _ in range(9)]
    vel.append(_gate_atk("velocity_burst", ("sequence", 0.9)))  # among = 0.90 exactly
    nov = [_gate_atk("novel_target", ("novelty", 0.9)) for _ in range(9)]
    nov.append(_gate_atk("novel_target", ("novelty", 0.9), fused=0.1))  # recall = 0.90 exactly
    g = gate_preview(_gate_rows(n_flagged=2, velocity_burst=vel, novel_target=nov))  # FPR 0.02
    assert g.cells["velocity_burst"].among_reasons_share == pytest.approx(0.9)
    assert g.cells["novel_target"].recall == pytest.approx(0.9)
    assert g.benign_fpr == pytest.approx(0.02)
    assert g.passed is True and g.failures == ()


def test_gate_preview_nan_benign_fpr_and_nan_share_fail() -> None:
    undetected = [_gate_atk("novel_target", ("novelty", 0.9), fused=0.1)]
    g = gate_preview(_gate_rows(n_benign=0, novel_target=undetected))
    assert math.isnan(g.benign_fpr)
    assert math.isnan(g.cells["novel_target"].among_reasons_share)
    assert g.passed is False
    assert g.failures == ("recall:novel_target", "among:novel_target", "benign_fpr")


def test_gate_preview_gates_among_inside_the_gate_bucket() -> None:
    # A detected magnitude spike of strength 5 (bucket 3-10, outside the gate bucket 10-100)
    # whose only reason is sequence: family-wide the share would be 1/2, in the gate bucket it is 1.
    off_bucket = _atk(
        0.9,
        attack_id="magnitude_spike-s5",
        attack_family="magnitude_spike",
        strength=5.0,
        threshold=0.5,
        per_detector_json=_per(("sequence", 0.95), ("magnitude", 0.3)),
    )
    mag = [_gate_atk("magnitude_spike", ("magnitude", 0.9)), off_bucket]
    rows = _gate_rows(magnitude_spike=mag)
    assert among_reasons_share(rows, "magnitude_spike", "magnitude", "row") == 0.5  # family-wide
    g = gate_preview(rows)
    cell = g.cells["magnitude_spike"]
    assert (cell.bucket, cell.n_detected, cell.n_total) == ("10-100", 1, 1)
    assert cell.among_reasons_share == 1.0
    assert "among:magnitude_spike" not in g.failures
    assert g.passed is True and g.failures == ()


def test_gate_preview_lists_missing_families_as_failures() -> None:
    g = gate_preview(_gate_rows(prereq_skip=[]))
    assert g.missing == ("prereq_skip",)
    assert g.passed is False and g.failures == ("missing:prereq_skip",)


# --- cold start, drift, latency -----------------------------------------------------------------


def test_cold_start_curve_buckets() -> None:
    rows = [
        _atk(0.9, n_key_obs=0),
        _atk(0.1, n_key_obs=9),
        _row(fused_score=0.9, n_key_obs=5),
        _atk(0.9, n_key_obs=10),
        _row(fused_score=0.1, n_key_obs=29),
        _row(fused_score=0.1, n_key_obs=30),
        _atk(0.9, n_key_obs=100),
        _row(fused_score=0.9, n_key_obs=5000),
        _row(fused_score=0.1, n_key_obs=100),
    ]
    curve = cold_start_curve(rows, 0.5)
    assert [(c.lo, c.hi) for c in curve] == [(0, 10), (10, 30), (30, 100), (100, math.inf)]
    b0, b1, b2, b3 = curve
    assert (b0.recall, b0.fpr, b0.n_attack, b0.n_benign) == (0.5, 1.0, 2, 1)
    assert (b1.recall, b1.fpr) == (1.0, 0.0)
    assert math.isnan(b2.recall) and b2.fpr == 0.0
    assert (b3.recall, b3.fpr, b3.n_attack, b3.n_benign) == (1.0, 0.5, 1, 2)


def test_drift_delay_counts_sessions_and_misses() -> None:
    rows = [
        *[
            _atk(0.9 if k in (2, 4) else 0.1, attack_id="d1", attack_family="slow_drift", step=k)
            for k in range(5)
        ],
        *[_atk(0.1, attack_id="d2", attack_family="slow_drift", step=k) for k in range(6)],
        _atk(0.9, attack_id="d3", attack_family="slow_drift", step=0),
        _atk(0.9, attack_id="m1", attack_family="magnitude_spike", step=None),
    ]
    d = drift_delay(rows, 0.5)
    assert d.n == 3
    assert d.miss_rate == pytest.approx(1 / 3)
    assert d.median == 2.0  # delays [2, 5 (miss -> k), 0]
    assert d.p90 == pytest.approx(4.4)


def test_drift_delay_without_drift_attacks() -> None:
    d = drift_delay([_atk(0.9), _row()], 0.5)
    assert d.n == 0 and math.isnan(d.median) and math.isnan(d.p90) and math.isnan(d.miss_rate)


def test_latency_summary_excludes_unscored_rows() -> None:
    rows = [_row(latency_us=float(v), fused_score=0.1) for v in range(1, 101)]
    rows.append(_row(latency_us=0.0, fused_score=float("nan")))
    p50, p99 = latency_summary(rows)
    assert p50 == pytest.approx(50.5) and p99 == pytest.approx(99.01)


# --- per-agent calibration ----------------------------------------------------------------------


def _scaled_agents() -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    for agent, lo, atk in (("A", 0.0, 0.3), ("B", 0.5, 0.8)):
        rows += [
            _row(agent_id=agent, split="train", fused_score=v)
            for v in np.linspace(lo, lo + 0.2, 101)
        ]
        rows += [
            _row(agent_id=agent, fused_score=v) for v in np.linspace(lo + 0.001, lo + 0.199, 50)
        ]
        rows += [_atk(atk, agent_id=agent) for _ in range(10)]
    return rows


def test_per_agent_beats_global_on_scaled_agents() -> None:
    # agent A benign ~U(0,0.2) attacks 0.3; agent B benign ~U(0.5,0.7) attacks 0.8
    rows = _scaled_agents()
    test = [r for r in rows if r.split == "test"]
    s = np.array([r.fused_score for r in test])
    y = np.array([r.label == "attack" for r in test])
    res = per_agent_fpr_at_recall(rows)
    assert res.fpr == 0.0 and res.recall == 1.0 and res.alpha == 0.0
    assert res.thresholds == {"A": pytest.approx(0.2), "B": pytest.approx(0.7)}
    assert fpr_at_recall(s, y)[1] > 0.4


def _linear_agent() -> list[ScoreRow]:
    return [
        # train benign: (1-alpha) quantile == 1-alpha
        *[_row(split="train", fused_score=v) for v in np.linspace(0.0, 1.0, 10_001)],
        _atk(0.0, split="train"),  # train attacks never move thresholds
        _row(split="test", fused_score=0.05),
        _row(split="test", fused_score=0.85),
        _row(split="test", fused_score=0.92),
        _row(split="test", fused_score=0.6),
        *[_atk(v) for v in (0.99, 0.95, 0.91, 0.5)],
    ]


def _g(train: NDArray[np.float64], score: float) -> float:
    """Exceedance fraction: the smallest alpha at which ``score`` is flagged."""
    return float(np.sum(train > score) / train.size)


_LIN = np.linspace(0.0, 1.0, 10_001)


def test_per_agent_picks_smallest_alpha_meeting_target() -> None:
    res = per_agent_fpr_at_recall(_linear_agent(), recall=0.75)
    # third-highest attack (0.91) sets the operating point: highest thresholds meeting the target
    assert res.alpha == _g(_LIN, 0.91) and res.alpha == pytest.approx(0.09, abs=2e-4)
    assert res.recall == 0.75 and res.fpr == 0.25  # only the 0.92 benign is flagged
    assert res.thresholds["A"] == float(_LIN[_LIN <= 0.91].max())


def test_per_agent_alpha_can_exceed_half() -> None:
    res = per_agent_fpr_at_recall([*_linear_agent(), _atk(0.2)], recall=1.0)
    assert res.alpha == _g(_LIN, 0.2) and res.alpha == pytest.approx(0.8, abs=2e-4)
    assert res.recall == 1.0 and res.fpr == 0.75


def test_per_agent_unreachable_target_gives_nan_fpr() -> None:
    train = np.linspace(0.5, 1.0, 501)
    rows = [
        *[_row(split="train", fused_score=v) for v in train],
        _row(fused_score=0.7),
        _atk(0.9),
        _atk(0.4),  # below every train benign score: never flagged, even at alpha = 1
    ]
    res = per_agent_fpr_at_recall(rows, recall=1.0)
    assert math.isnan(res.fpr)
    assert res.recall == 0.5 and res.alpha == _g(train, 0.9)  # best achievable point


def test_per_agent_unseen_agent_uses_pooled_train_quantile() -> None:
    rows = [*_linear_agent(), _atk(0.95, agent_id="C"), _row(agent_id="C", fused_score=0.3)]
    res = per_agent_fpr_at_recall(rows, recall=0.75)
    assert res.thresholds["C"] == res.thresholds["A"]


def _random_rows(n: int = 20_000, seed: int = 0) -> list[ScoreRow]:
    rng = np.random.default_rng(seed)
    return [
        _row(
            agent_id="ABCD"[i % 4],
            split="train" if i < 0.7 * n else "test",
            fused_score=float(rng.random()),
            label="attack" if rng.random() < 0.05 else "benign",
        )
        for i in range(n)
    ]


def test_per_agent_random_scores_cost_about_target_fpr() -> None:
    # Uninformative scores: any operating point with recall >= 0.95 flags ~95% of benigns.
    # ~300 test attacks -> sd of the 0.95 order statistic ~ sqrt(.95*.05/300) ~ 0.013, so a 0.05
    # tolerance is ~4 sd. A grid capped at alpha = 0.5 would report ~0.5 here.
    m = compute_all(_random_rows(), warm_only=False, n_boot=5)
    assert m.per_agent_recall >= 0.95
    assert abs(m.fpr95_per_agent - 0.95) < 0.05
    assert abs(m.fpr95_global - 0.95) < 0.05


def test_compute_all_unreachable_per_agent_is_nan_with_nan_ci() -> None:
    train = np.linspace(0.5, 1.0, 501)
    rows = [
        *[_row(split="train", fused_score=v) for v in train],
        _row(fused_score=0.7),
        _atk(0.9),
        _atk(0.4),
    ]
    m = compute_all(rows, warm_only=False, n_boot=20)
    assert math.isnan(m.fpr95_per_agent)
    assert all(math.isnan(v) for v in m.fpr95_per_agent_ci)
    assert m.per_agent_recall == 0.5


def test_recall_target_must_be_in_unit_interval() -> None:
    s = np.array([0.9, 0.1])
    y = np.array([1, 0], bool)
    assert fpr_at_recall(s, y, 1.0) == (0.9, 0.0)
    for bad in (0.0, -0.1, 1.5, math.nan):
        with pytest.raises(ValueError):
            fpr_at_recall(s, y, bad)
        with pytest.raises(ValueError):
            per_agent_fpr_at_recall(_linear_agent()[-6:], recall=bad)


# --- bootstrap ----------------------------------------------------------------------------------


def _sessions(n_sess: int = 30, per: int = 3, seed: int = 1) -> list[ScoreRow]:
    rng = np.random.default_rng(seed)
    rows: list[ScoreRow] = []
    for k in range(n_sess):
        attack = k % 3 == 0
        for _ in range(per):
            score = float(rng.random() + (0.5 if attack else 0.0))
            rows.append(
                _row(session_id=f"S{k}", fused_score=score, label="attack" if attack else "benign")
            )
    return rows


def _auprc_rows(rs: Sequence[ScoreRow]) -> float:
    return auprc(np.array([r.fused_score for r in rs]), np.array([r.label == "attack" for r in rs]))


def test_bootstrap_ci_contains_point_and_resamples_sessions() -> None:
    rows = _sessions()
    seen: list[Counter[str]] = []

    def spy(rs: Sequence[ScoreRow]) -> float:
        seen.append(Counter(r.session_id for r in rs))
        return _auprc_rows(rs)

    lo, hi = bootstrap_ci(rows, spy, n=200, seed=3)
    point = _auprc_rows(rows)
    assert lo <= point <= hi and lo < hi
    assert len(seen) == 200
    for c in seen:
        assert sum(c.values()) == len(rows)  # n_sessions drawn, each with all its rows
        assert all(v % 3 == 0 for v in c.values())  # whole sessions, never single envelopes
    assert any(max(c.values()) > 3 for c in seen)  # with replacement
    assert len({tuple(sorted(c.items())) for c in seen}) > 1


def test_bootstrap_ci_deterministic_and_skips_nan() -> None:
    rows = _sessions()
    assert bootstrap_ci(rows, _auprc_rows, n=100, seed=5) == bootstrap_ci(
        rows, _auprc_rows, n=100, seed=5
    )
    assert bootstrap_ci(rows, _auprc_rows, n=100, seed=5) != bootstrap_ci(
        rows, _auprc_rows, n=100, seed=6
    )

    def nan_unless_s0(rs: Sequence[ScoreRow]) -> float:
        return 1.0 if any(r.session_id == "S0" for r in rs) else math.nan

    assert bootstrap_ci(rows, nan_unless_s0, n=100) == (1.0, 1.0)
    lo, hi = bootstrap_ci(rows, lambda rs: math.nan, n=10)
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_ci_level_controls_width() -> None:
    rows = _sessions()
    lo95, hi95 = bootstrap_ci(rows, _auprc_rows, n=300, seed=0, level=0.95)
    lo50, hi50 = bootstrap_ci(rows, _auprc_rows, n=300, seed=0, level=0.5)
    assert lo95 <= lo50 <= hi50 <= hi95


# --- compute_all --------------------------------------------------------------------------------


def _dataset(row_thr: float | None) -> list[ScoreRow]:
    rows = _scaled_agents()
    rows = [replace(r, threshold=row_thr) if r.split == "test" else r for r in rows]
    rows += [
        _row(fused_score=float("nan"), label="attack", attack_family="magnitude_spike"),
        _row(fused_score=0.99, is_warm=False, n_key_obs=3),  # cold benign, flagged
        _row(fused_score=0.9, split="train", label="attack", attack_family="novel_target"),
    ]
    return rows


def test_compute_all_test_split_warm_filter_and_fields() -> None:
    rows = _dataset(row_thr=None)
    all_m = compute_all(rows, warm_only=False, n_boot=50)
    warm = compute_all(rows, warm_only=True, n_boot=50)
    # test split, NaN excluded: 2 agents x (50 benign + 10 attacks) + 1 cold benign
    assert all_m.n_rows == 121 and warm.n_rows == 120
    assert all_m.n_attack_rows == 20 == warm.n_attack_rows
    # pooled: B attacks (0.8) rank first, then all 50 B benigns outrank A attacks (0.3)
    assert warm.auprc == pytest.approx((1 + 20 / 70) / 2)
    assert warm.auprc_ci[0] <= warm.auprc <= warm.auprc_ci[1]
    assert warm.thr95_global == pytest.approx(0.3) and warm.recall_at_global_thr == 1.0
    assert warm.fpr95_global == pytest.approx(0.5)
    assert warm.fpr95_per_agent == 0.0 and warm.per_agent_alpha == 0.0
    assert warm.per_agent_recall == 1.0 and warm.breakdown_mode == "per_agent"
    assert warm.fpr95_per_agent_ci[0] == 0.0 <= warm.fpr95_per_agent_ci[1] < 0.1
    assert set(warm.per_family_envelope) == {"magnitude_spike"}  # train-split family excluded
    assert warm.per_family_attack["magnitude_spike"]["all"].recall == 1.0
    assert math.isnan(warm.control_fpr)
    assert len(warm.cold_start) == 4 and warm.drift.n == 0
    assert warm.latency_p50_us == 10.0 and warm.latency_p99_us == 10.0
    # the cold flagged benign only shows up without the warm filter
    assert all_m.cold_start[0].fpr == 1.0 and warm.cold_start[0].n_benign == 0


def test_compute_all_breakdown_threshold_modes() -> None:
    rows = _dataset(row_thr=0.95)  # every attack (0.3 / 0.8) is below its row threshold
    row = compute_all(rows, warm_only=True, breakdown_thresholds="row", n_boot=10)
    per_agent = compute_all(rows, warm_only=True, n_boot=10)
    assert row.breakdown_mode == "row" and per_agent.breakdown_mode == "per_agent"
    assert row.per_family_envelope["magnitude_spike"]["all"].recall == 0.0
    assert per_agent.per_family_envelope["magnitude_spike"]["all"].recall == 1.0
    # "row" never silently switches mode: rows without a threshold are simply not flagged
    none = compute_all(
        _dataset(row_thr=None), warm_only=True, breakdown_thresholds="row", n_boot=10
    )
    assert none.breakdown_mode == "row"
    assert none.per_family_attack["magnitude_spike"]["all"].recall == 0.0


def test_bootstrap_records_dropped_nan_replicates() -> None:
    from agentfw.bench.metrics import bootstrap_ci_counted

    rows = _sessions()

    def nan_unless_s0(rs: Sequence[ScoreRow]) -> float:
        return 1.0 if any(r.session_id == "S0" for r in rs) else math.nan

    ci = bootstrap_ci_counted(rows, nan_unless_s0, n=100, seed=0)
    assert 0 < ci.n_dropped < 100
    assert (ci.lo, ci.hi) == bootstrap_ci(rows, nan_unless_s0, n=100, seed=0) == (1.0, 1.0)
    assert bootstrap_ci_counted(rows, lambda rs: math.nan, n=10).n_dropped == 10
    assert bootstrap_ci_counted(rows, _auprc_rows, n=10).n_dropped == 0


def test_metrics_and_report_carry_the_dropped_counts() -> None:
    from agentfw.bench.report import _headline

    # one attack session among 40: resamples that miss it have no attack (AUPRC NaN)
    rows = [_row(session_id=f"B{k}", fused_score=0.1 * (k % 7)) for k in range(40)]
    rows.append(_row(session_id="ATK", fused_score=0.9, label="attack"))
    m = compute_all(rows, warm_only=False, n_boot=200)
    assert 0 < m.auprc_ci_n_dropped < 200
    h = _headline(m, warm_only=False)
    assert h["auprc_ci_n_dropped"] == m.auprc_ci_n_dropped
    assert h["fpr95_per_agent_ci_n_dropped"] == m.fpr95_per_agent_ci_n_dropped
