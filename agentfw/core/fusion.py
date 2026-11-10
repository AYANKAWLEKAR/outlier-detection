"""Fusion v1 (spec §6.7): ``top1 + fusion_bonus * [top2 > fusion_bonus_gate]``, clipped to [0, 1].

A NaN detector score is treated as 0 so the fused score is always finite. Every score above
``reason_min`` becomes a :class:`Reason`, ordered by descending score (stable on ties).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from agentfw.core.config import EngineConfig
from agentfw.core.envelope import BaselineSummary, DetectorScore, Reason, Scalar

__all__ = ["FusionResult", "fuse", "reason_from"]


@dataclass(slots=True, frozen=True)
class FusionResult:
    fused: float
    reasons: tuple[Reason, ...]


def _num(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v)


def _scalar(v: object) -> Scalar:
    return v if v is None or isinstance(v, str | int | float | bool) else None


def reason_from(s: DetectorScore) -> Reason:
    """Map a detector score's evidence onto a :class:`Reason`; absent keys stay ``None``.

    ``observed`` is taken from evidence ``observed`` (magnitude), else ``window_count``
    (velocity). Novelty evidence carries only a value hash, so its ``observed`` stays ``None``
    and raw ids never enter a reason. ``BaselineSummary`` needs median, iqr and n together.
    """
    ev = s.evidence
    observed = _scalar(ev["observed"]) if "observed" in ev else _scalar(ev.get("window_count"))
    median, iqr, n = _num(ev.get("median_raw")), _num(ev.get("iqr_raw")), ev.get("n")
    baseline = (
        BaselineSummary(median, iqr, n)
        if median is not None and iqr is not None and isinstance(n, int) and not isinstance(n, bool)
        else None
    )
    pm = ev.get("prereq_missing")
    return Reason(
        detector=s.detector,
        field=s.field,
        observed=observed,
        baseline_summary=baseline,
        percentile=_num(ev.get("percentile")),
        blast_radius=None,
        prereq_missing=pm if isinstance(pm, str) else None,
        prior_used=ev.get("prior_used") is True,
    )


def fuse(scores: Sequence[DetectorScore], cfg: EngineConfig) -> FusionResult:
    top1 = top2 = 0.0
    cleaned: list[tuple[float, DetectorScore]] = []
    for s in scores:
        v = 0.0 if math.isnan(s.score) else s.score
        cleaned.append((v, s))
        if v > top1:
            top1, top2 = v, top1
        elif v > top2:
            top2 = v
    fused = top1 + (cfg.fusion_bonus if top2 > cfg.fusion_bonus_gate else 0.0)
    fused = min(1.0, max(0.0, fused))
    hits = sorted((c for c in cleaned if c[0] > cfg.reason_min), key=lambda c: -c[0])
    return FusionResult(fused, tuple(reason_from(s) for _, s in hits))
