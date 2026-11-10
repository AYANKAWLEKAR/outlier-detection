from __future__ import annotations

import math

import pytest

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import BaselineSummary, DetectorScore
from agentfw.core.fusion import FusionResult, fuse, reason_from


def ds(
    score: float, detector: str = "magnitude", field: str | None = "f", **ev: object
) -> DetectorScore:
    return DetectorScore(detector, field, score, ev)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.9, 0.6], 1.0),
        ([0.9, 0.5], 0.9),
        ([0.7, 0.6, 0.6], 0.85),
        ([], 0.0),
        ([0.2], 0.2),
    ],
)
def test_fusion_v1(scores: list[float], expected: float) -> None:
    res = fuse([ds(s) for s in scores], EngineConfig())
    assert isinstance(res, FusionResult)
    assert res.fused == pytest.approx(expected)


def test_reasons_only_above_03_and_ordered() -> None:
    scores = [ds(0.2, "a"), ds(0.8, "b"), ds(0.3, "c"), ds(0.5, "d"), ds(0.31, "e")]
    res = fuse(scores, EngineConfig())
    assert [r.detector for r in res.reasons] == ["b", "d", "e"]


def test_nan_scores_treated_as_zero() -> None:
    res = fuse([ds(math.nan), ds(0.4)], EngineConfig())
    assert res.fused == pytest.approx(0.4)
    assert math.isfinite(fuse([ds(math.nan)], EngineConfig()).fused)
    assert fuse([ds(math.nan)], EngineConfig()).reasons == ()


def test_fused_clipped_to_unit_interval() -> None:
    assert fuse([ds(5.0), ds(5.0)], EngineConfig()).fused == 1.0
    assert fuse([ds(-1.0)], EngineConfig()).fused == 0.0


def test_reason_from_magnitude() -> None:
    s = ds(
        0.9,
        observed=500.0,
        median_raw=40.0,
        iqr_raw=10.0,
        n=120,
        percentile=0.999,
        prior_used=True,
        z=7.0,
    )
    r = reason_from(s)
    assert r.detector == "magnitude" and r.field == "f"
    assert r.observed == 500.0
    assert r.baseline_summary == BaselineSummary(40.0, 10.0, 120)
    assert r.percentile == 0.999
    assert r.prior_used is True
    assert r.prereq_missing is None and r.blast_radius is None


def test_reason_from_novelty_never_leaks_raw_value() -> None:
    r = reason_from(ds(0.8, "novelty", p_unseen=0.1, n_total=50, value_hash="abc123"))
    assert r.observed is None
    assert r.baseline_summary is None
    assert r.percentile is None
    assert r.prior_used is False


def test_reason_from_sequence() -> None:
    r = reason_from(ds(1.0, "sequence", None, fired="prereq", prereq_missing="crm.lookup"))
    assert r.field is None
    assert r.prereq_missing == "crm.lookup"
    assert r.baseline_summary is None


def test_reason_from_velocity_window_count() -> None:
    r = reason_from(ds(0.7, "velocity", window_count=40, baseline_median=3.0, n_hist=100))
    assert r.observed == 40
    assert r.baseline_summary is None


def test_baseline_summary_requires_all_three() -> None:
    assert reason_from(ds(0.5, median_raw=1.0, n=5)).baseline_summary is None
