"""Benchmark metrics over the score table (spec 5.4; D-007, D-008, D-009).

Every published number comes from this module. numpy only.

Conventions shared by every function:

* Rows whose ``fused_score`` is NaN (not sampled by the detector) are excluded from every metric.
  The array-level functions (:func:`auprc`, :func:`fpr_at_recall`) drop NaN scores the same way.
* A row is **flagged** when ``fused_score >= threshold``. :data:`Thresholds` is one of

  - a ``float``: one global threshold;
  - a ``Mapping[str, float]`` of ``agent_id -> threshold``: agents missing from the map are never
    flagged (equivalent to a ``+inf`` threshold);
  - ``"row"``: each row's own ``ScoreRow.threshold`` (the engine's live per-agent threshold); rows
    whose threshold is ``None`` are never flagged.
* Rates with an empty denominator are ``NaN``.
* Recall comparisons against a target allow ``1e-12`` of floating-point slack.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Literal

import numpy as np
import orjson
from numpy.typing import NDArray

from agentfw.bench.replay import ScoreRow

__all__ = [
    "BOOTSTRAP_N",
    "COLD_START_BUCKETS",
    "RECALL_TARGET",
    "STRENGTH_BUCKETS",
    "BucketCell",
    "DriftDelay",
    "Metrics",
    "PerAgentResult",
    "RecallCell",
    "Thresholds",
    "auprc",
    "benign_fpr",
    "bootstrap_ci",
    "cold_start_curve",
    "compute_all",
    "control_fpr",
    "drift_delay",
    "fpr_at_recall",
    "latency_summary",
    "per_agent_fpr_at_recall",
    "recall_by_family",
    "top_reason_share",
]

RECALL_TARGET = 0.95
BOOTSTRAP_N = 1000
_EPS = 1e-12

#: Strength bucket edges per family (D-009). Bucket ``"lo-hi"`` is ``[lo, hi)`` except the last,
#: which is ``[lo, hi]``. Strengths below the first edge are clamped into the first bucket and
#: strengths above the last edge into the last bucket (realized strengths can fall a hair outside
#: the sampling range); attacks with no strength, or of a family not listed here, count only in
#: the ``"all"`` bucket. Keys are formatted with ``:g`` (``"3-10"``, ``"0.06-0.1"``).
STRENGTH_BUCKETS: dict[str, tuple[float, ...]] = {
    "magnitude_spike": (3, 10, 100),
    "scope_blowup": (100, 1000, 10000),
    "velocity_burst": (10, 30, 60),
    "slow_drift": (0.03, 0.06, 0.10),
    "exfil_volume": (10, 30, 100),
}

#: Cold-start buckets on ``n_key_obs``: ``[lo, hi)``.
COLD_START_BUCKETS: tuple[tuple[float, float], ...] = (
    (0, 10),
    (10, 30),
    (30, 100),
    (100, math.inf),
)

Thresholds = float | Mapping[str, float] | Literal["row"]
BreakdownMode = Literal["per_agent", "row"]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True, frozen=True)
class RecallCell:
    recall: float
    n_detected: int
    n_total: int


@dataclass(slots=True, frozen=True)
class BucketCell:
    lo: float
    hi: float
    recall: float
    fpr: float
    n_attack: int
    n_benign: int


@dataclass(slots=True, frozen=True)
class DriftDelay:
    median: float
    p90: float
    miss_rate: float
    n: int


@dataclass(slots=True, frozen=True)
class PerAgentResult:
    alpha: float
    fpr: float
    recall: float
    thresholds: Mapping[str, float]


@dataclass(slots=True)
class Metrics:
    auprc: float
    auprc_ci: tuple[float, float]
    thr95_global: float
    fpr95_global: float
    recall_at_global_thr: float
    fpr95_per_agent: float
    fpr95_per_agent_ci: tuple[float, float]
    per_agent_alpha: float
    per_agent_recall: float
    breakdown_mode: str
    per_family_envelope: dict[str, dict[str, RecallCell]]
    per_family_attack: dict[str, dict[str, RecallCell]]
    control_fpr: float
    cold_start: list[BucketCell]
    drift: DriftDelay
    latency_p50_us: float
    latency_p99_us: float
    n_rows: int
    n_attack_rows: int


# --- helpers ------------------------------------------------------------------------------------


def _check_recall(recall: float) -> None:
    if not 0.0 < recall <= 1.0:  # also rejects NaN
        raise ValueError(f"recall target must be in (0, 1], got {recall!r}")


def _rate(num: int, den: int) -> float:
    return num / den if den else math.nan


def _valid(rows: Sequence[ScoreRow]) -> list[ScoreRow]:
    return [r for r in rows if not math.isnan(r.fused_score)]


def _is_attack(r: ScoreRow) -> bool:
    return r.label == "attack"


def _flag(r: ScoreRow, thresholds: Thresholds) -> bool:
    if isinstance(thresholds, str):
        return r.threshold is not None and r.fused_score >= r.threshold
    if isinstance(thresholds, Mapping):
        t = thresholds.get(r.agent_id)
        return t is not None and r.fused_score >= t
    return r.fused_score >= thresholds


def _clean(scores: FloatArray, y: BoolArray) -> tuple[FloatArray, BoolArray]:
    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y, dtype=np.bool_)
    keep = ~np.isnan(s)
    return s[keep], yy[keep]


def _tie_groups(s: FloatArray, y: BoolArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Descending distinct thresholds with cumulative TP / FP at each (``score >= t``)."""
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    yy = y[order]
    tp = np.cumsum(yy, dtype=np.float64)
    fp = np.cumsum(~yy, dtype=np.float64)
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), ss.size - 1]
    return ss[ends], tp[ends], fp[ends]


# --- global curve metrics -----------------------------------------------------------------------


def auprc(scores: FloatArray, y: NDArray[np.bool_]) -> float:
    """Average precision ``sum_k (R_k - R_{k-1}) P_k`` with each tie group as one threshold step.

    All-tied scores give the base rate. NaN when there are no positives.
    """
    s, yy = _clean(scores, y)
    n_pos = int(yy.sum())
    if n_pos == 0:
        return math.nan
    _, tp, fp = _tie_groups(s, yy)
    precision = tp / (tp + fp)
    d_recall = np.diff(np.r_[0.0, tp]) / n_pos
    return float(np.sum(d_recall * precision))


def fpr_at_recall(
    scores: FloatArray, y: NDArray[np.bool_], recall: float = RECALL_TARGET
) -> tuple[float, float]:
    """D-007: the **highest** threshold whose recall >= ``recall``; returns ``(threshold, fpr)``.

    ``(nan, nan)`` without positives; the FPR is NaN without negatives. Raises ``ValueError``
    unless ``0 < recall <= 1``.
    """
    _check_recall(recall)
    s, yy = _clean(scores, y)
    n_pos = int(yy.sum())
    n_neg = int(yy.size - n_pos)
    if n_pos == 0:
        return math.nan, math.nan
    thr, tp, fp = _tie_groups(s, yy)
    # Recall is non-decreasing as the threshold descends; full recall is reached at the minimum.
    i = int(np.argmax(tp / n_pos >= recall - _EPS))
    return float(thr[i]), _rate(int(fp[i]), n_neg)


# --- thresholded rates --------------------------------------------------------------------------


def benign_fpr(rows: Sequence[ScoreRow], thresholds: Thresholds) -> float:
    """FPR over all benign rows (controls included)."""
    benign = [r for r in _valid(rows) if not _is_attack(r)]
    return _rate(sum(_flag(r, thresholds) for r in benign), len(benign))


def control_fpr(rows: Sequence[ScoreRow], thresholds: Thresholds) -> float:
    """FPR over benign rows with ``control_type == "benign_unusual"``."""
    ctrl = [r for r in _valid(rows) if not _is_attack(r) and r.control_type == "benign_unusual"]
    return _rate(sum(_flag(r, thresholds) for r in ctrl), len(ctrl))


def _bucket_key(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def _bucket_of(family: str, strength: float | None) -> str | None:
    edges = STRENGTH_BUCKETS.get(family)
    if edges is None or strength is None or math.isnan(strength):
        return None
    # Index of the bucket [e_i, e_{i+1}); clamp to the first/last bucket (last is upper-inclusive).
    i = int(np.searchsorted(edges, strength, side="right")) - 1
    i = min(max(i, 0), len(edges) - 2)
    return _bucket_key(edges[i], edges[i + 1])


def _attacks(rows: Sequence[ScoreRow]) -> dict[str, list[ScoreRow]]:
    """Valid attack rows grouped by ``attack_id`` (fallback: ``envelope_id``), each sorted by ts."""
    groups: dict[str, list[ScoreRow]] = defaultdict(list)
    for r in _valid(rows):
        if _is_attack(r):
            groups[r.attack_id or r.envelope_id].append(r)
    for g in groups.values():
        g.sort(key=lambda r: (r.ts_unix_ms, r.envelope_id))
    return dict(groups)


def recall_by_family(
    rows: Sequence[ScoreRow],
    thresholds: Thresholds,
    level: Literal["envelope", "attack"],
) -> dict[str, dict[str, RecallCell]]:
    """Recall per attack family and strength bucket (D-008, D-009).

    ``level="envelope"``: each attack row counts on its own. ``level="attack"``: an ``attack_id``
    is detected if any of its rows is flagged; its family and strength come from its first row by
    ``(ts_unix_ms, envelope_id)``. Every family present gets all of its defined buckets (empty ones
    have ``recall=NaN``) plus ``"all"``.
    """
    units: list[tuple[str, float | None, bool]] = []
    if level == "envelope":
        for r in _valid(rows):
            if _is_attack(r):
                units.append((r.attack_family or "unknown", r.strength, _flag(r, thresholds)))
    else:
        for g in _attacks(rows).values():
            first = g[0]
            detected = any(_flag(r, thresholds) for r in g)
            units.append((first.attack_family or "unknown", first.strength, detected))

    counts: dict[str, dict[str, list[int]]] = {}
    for family, strength, hit in units:
        if family not in counts:
            edges = STRENGTH_BUCKETS.get(family, ())
            keys = [_bucket_key(a, b) for a, b in zip(edges, edges[1:], strict=False)]
            counts[family] = {k: [0, 0] for k in [*keys, "all"]}
        bucket = _bucket_of(family, strength)
        for key in ("all", bucket):
            if key is not None:
                counts[family][key][0] += hit
                counts[family][key][1] += 1
    return {
        fam: {k: RecallCell(_rate(d, n), d, n) for k, (d, n) in cells.items()}
        for fam, cells in counts.items()
    }


def _top_detector(per_detector_json: str) -> str | None:
    """Detector with the highest numeric score; null / non-numeric / NaN scores are skipped."""
    best: str | None = None
    best_score = -math.inf
    for item in orjson.loads(per_detector_json):  # first in list order wins ties
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float) or math.isnan(score):
            continue
        if best is None or score > best_score:
            best, best_score = str(item.get("detector")), float(score)
    return best


def top_reason_share(
    rows: Sequence[ScoreRow], family: str, detector: str, thresholds: Thresholds
) -> float:
    """Share of detected ``family`` attacks whose top per-detector score came from ``detector``.

    Attack level: per detected ``attack_id`` the representative row is its flagged row with the
    highest ``fused_score`` (earliest on ties); its top detector is the ``per_detector_json`` entry
    with the highest score (first in list order on ties). NaN when no attack was detected.
    """
    hits = 0
    detected = 0
    for g in _attacks(rows).values():
        if (g[0].attack_family or "unknown") != family:
            continue
        flagged = [r for r in g if _flag(r, thresholds)]
        if not flagged:
            continue
        detected += 1
        rep = max(flagged, key=lambda r: r.fused_score)  # max() keeps the first maximum
        hits += _top_detector(rep.per_detector_json) == detector
    return _rate(hits, detected)


def cold_start_curve(rows: Sequence[ScoreRow], thresholds: Thresholds) -> list[BucketCell]:
    """Envelope-level recall and FPR per ``n_key_obs`` bucket (:data:`COLD_START_BUCKETS`)."""
    valid = _valid(rows)
    out: list[BucketCell] = []
    for lo, hi in COLD_START_BUCKETS:
        sel = [r for r in valid if lo <= r.n_key_obs < hi]
        atk = [_flag(r, thresholds) for r in sel if _is_attack(r)]
        ben = [_flag(r, thresholds) for r in sel if not _is_attack(r)]
        out.append(
            BucketCell(
                lo, hi, _rate(sum(atk), len(atk)), _rate(sum(ben), len(ben)), len(atk), len(ben)
            )
        )
    return out


def drift_delay(rows: Sequence[ScoreRow], thresholds: Thresholds) -> DriftDelay:
    """Detection delay in sessions for ``slow_drift`` attacks, counted by ``Label.step``.

    Per attack: delay = smallest ``step`` among its flagged rows; an undetected attack counts as
    ``k`` = its max ``step`` and as a miss. Rows with ``step=None`` are ignored; an attack with no
    stepped rows is skipped. ``n`` is the number of attacks measured; NaN stats when ``n == 0``.
    """
    delays: list[float] = []
    misses = 0
    for g in _attacks(rows).values():
        if g[0].attack_family != "slow_drift":
            continue
        stepped = [r for r in g if r.step is not None]
        if not stepped:
            continue
        hit_steps = [r.step for r in stepped if r.step is not None and _flag(r, thresholds)]
        if hit_steps:
            delays.append(float(min(hit_steps)))
        else:
            delays.append(float(max(r.step for r in stepped if r.step is not None)))
            misses += 1
    if not delays:
        return DriftDelay(math.nan, math.nan, math.nan, 0)
    arr = np.asarray(delays)
    return DriftDelay(
        float(np.median(arr)), float(np.percentile(arr, 90)), misses / len(delays), len(delays)
    )


def latency_summary(rows: Sequence[ScoreRow]) -> tuple[float, float]:
    """``(p50, p99)`` of ``score()`` wall time in microseconds over scored rows."""
    lat = np.asarray([r.latency_us for r in _valid(rows)], dtype=np.float64)
    if lat.size == 0:
        return math.nan, math.nan
    p50, p99 = np.percentile(lat, [50, 99])
    return float(p50), float(p99)


# --- per-agent calibration ----------------------------------------------------------------------


def _exceedance(train_sorted: FloatArray, scores: FloatArray) -> FloatArray:
    """Smallest alpha at which each score is flagged against one agent's train-benign ECDF.

    With ``q_p`` the inverted-CDF quantile (smallest train value ``b`` with ``F(b) >= p``),
    ``score >= q_{1-alpha}`` iff ``score >= min(train)`` and ``alpha >= #(train > score) / n``.
    Scores below every train value are never flagged (``+inf``), even at ``alpha = 1``.
    """
    n = train_sorted.size
    above = n - np.searchsorted(train_sorted, scores, side="right")
    out = above / n
    out[scores < train_sorted[0]] = math.inf
    return out


def _threshold_at(train_sorted: FloatArray, alpha: float) -> float:
    """``q_{1-alpha}``: the smallest train value ``b`` with ``#(train > b) / n <= alpha``.

    Flagging ``score >= q_{1-alpha}`` reproduces ``_exceedance(score) <= alpha`` exactly.
    """
    g = _exceedance(train_sorted, train_sorted)
    return float(train_sorted[int(np.argmax(g <= alpha + _EPS))])


def per_agent_fpr_at_recall(
    rows: Sequence[ScoreRow],
    recall: float = RECALL_TARGET,
) -> PerAgentResult:
    """Per-agent operating point: one ``alpha`` shared by all agents (spec 5.4, D-007, D-021).

    Each agent's threshold is the ``1 - alpha`` inverted-CDF quantile of that agent's
    **train-split benign** scores; an agent with no train benign rows (e.g. a held-out agent in the
    by-agent split) uses the pooled train-benign ECDF. Recall and FPR are pooled over the **test**
    split. Exact, no grid: every test row gets its flag-alpha (:func:`_exceedance`), and the
    operating point is the **smallest** candidate alpha in ``[0, 1]`` (highest thresholds) whose
    test recall >= ``recall``. ``thresholds`` holds every agent's score threshold at that alpha.

    If the target is unreachable even at ``alpha = 1`` (attacks scoring below every train-benign
    score of their agent are never flagged), ``fpr`` is NaN while ``alpha`` / ``recall`` /
    ``thresholds`` describe the best achievable point (smallest alpha with maximal recall); if no
    test attack is flaggable at all, ``alpha`` is NaN, ``recall`` 0 and ``thresholds`` empty.
    All-NaN result when there are no train benign rows or no test attack rows.
    """
    _check_recall(recall)
    valid = _valid(rows)
    train_by_agent: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        if r.split == "train" and not _is_attack(r):
            train_by_agent[r.agent_id].append(r.fused_score)
    test = [r for r in valid if r.split == "test"]
    pooled = np.sort(np.asarray(list(chain.from_iterable(train_by_agent.values()))))
    is_atk = np.asarray([_is_attack(r) for r in test], dtype=np.bool_)
    n_atk = int(is_atk.sum())
    if pooled.size == 0 or n_atk == 0:
        return PerAgentResult(math.nan, math.nan, math.nan, {})

    train: dict[str, FloatArray] = {}
    for agent in sorted({r.agent_id for r in valid}):
        own = train_by_agent.get(agent)
        train[agent] = np.sort(np.asarray(own)) if own else pooled

    flag_alpha = np.empty(len(test), dtype=np.float64)
    by_agent: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(test):
        by_agent[r.agent_id].append(i)
    scores = np.asarray([r.fused_score for r in test], dtype=np.float64)
    for agent, idx in by_agent.items():
        ix = np.asarray(idx)
        flag_alpha[ix] = _exceedance(train[agent], scores[ix])

    atk_alpha = np.sort(flag_alpha[is_atk])
    cand = np.unique(atk_alpha[np.isfinite(atk_alpha)])
    if cand.size == 0:
        return PerAgentResult(math.nan, math.nan, 0.0, {})
    rec = np.searchsorted(atk_alpha, cand + _EPS, side="right") / n_atk
    meets = rec >= recall - _EPS
    j = int(np.argmax(meets)) if meets.any() else int(np.argmax(rec))
    alpha = float(cand[j])
    ben_alpha = flag_alpha[~is_atk]
    fpr = _rate(int(np.sum(ben_alpha <= alpha + _EPS)), ben_alpha.size) if meets.any() else math.nan
    return PerAgentResult(
        alpha=alpha,
        fpr=fpr,
        recall=float(rec[j]),
        thresholds={agent: _threshold_at(t, alpha) for agent, t in train.items()},
    )


# --- bootstrap ----------------------------------------------------------------------------------


def bootstrap_ci(
    rows: Sequence[ScoreRow],
    metric: Callable[[Sequence[ScoreRow]], float],
    n: int = BOOTSTRAP_N,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over **sessions** (spec 5.4).

    Each resample draws as many sessions as exist, with replacement (``np.random.default_rng``),
    and keeps every row of each drawn session. Resamples where ``metric`` is NaN are skipped;
    ``(nan, nan)`` if all are.
    """
    groups: dict[str, list[ScoreRow]] = defaultdict(list)
    for r in rows:
        groups[r.session_id].append(r)
    sessions = [groups[k] for k in sorted(groups)]
    if not sessions:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n):
        pick = rng.integers(0, len(sessions), size=len(sessions))
        v = metric(list(chain.from_iterable(sessions[int(i)] for i in pick)))
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return math.nan, math.nan
    lo, hi = np.percentile(vals, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


# --- everything ---------------------------------------------------------------------------------


def _xy(rows: Sequence[ScoreRow]) -> tuple[FloatArray, BoolArray]:
    s = np.asarray([r.fused_score for r in rows], dtype=np.float64)
    y = np.asarray([_is_attack(r) for r in rows], dtype=np.bool_)
    return s, y


def _auprc_rows(rows: Sequence[ScoreRow]) -> float:
    return auprc(*_xy(rows))


def _per_agent_fpr(rows: Sequence[ScoreRow]) -> float:
    return per_agent_fpr_at_recall(rows).fpr


def compute_all(
    rows: Sequence[ScoreRow],
    *,
    warm_only: bool,
    breakdown_thresholds: BreakdownMode = "per_agent",
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Metrics:
    """All headline metrics on the **test** split (NaN-score rows excluded).

    ``warm_only`` keeps only ``is_warm`` rows -- for evaluation *and* for the train-split per-agent
    calibration, so thresholds are fit on the population they are judged on. The per-agent CI
    resamples sessions over train and test rows together, so it includes calibration variance.
    Thresholded breakdowns (family recall, control FPR, cold start, drift) use
    ``breakdown_thresholds``: ``"per_agent"`` (default) = the per-agent thresholds chosen by
    :func:`per_agent_fpr_at_recall`; ``"row"`` = each row's own live engine threshold (rows without
    one are never flagged). The mode is recorded in ``Metrics.breakdown_mode``; it is never chosen
    automatically. When the per-agent target is unreachable, ``fpr95_per_agent`` and its CI are
    NaN and ``per_agent_recall`` reports the best achievable recall.
    """
    pool = [r for r in _valid(rows) if r.is_warm or not warm_only]
    test = [r for r in pool if r.split == "test"]
    s, y = _xy(test)

    thr_g, fpr_g = fpr_at_recall(s, y, RECALL_TARGET)
    n_atk = int(y.sum())
    recall_g = _rate(int(np.sum(s[y] >= thr_g)), n_atk)

    pa = per_agent_fpr_at_recall(pool, RECALL_TARGET)
    thresholds: Thresholds = "row" if breakdown_thresholds == "row" else pa.thresholds
    pa_ci = (
        (math.nan, math.nan)
        if math.isnan(pa.fpr)
        else bootstrap_ci(pool, _per_agent_fpr, n=n_boot, seed=seed)
    )

    p50, p99 = latency_summary(test)
    return Metrics(
        auprc=auprc(s, y),
        auprc_ci=bootstrap_ci(test, _auprc_rows, n=n_boot, seed=seed),
        thr95_global=thr_g,
        fpr95_global=fpr_g,
        recall_at_global_thr=recall_g,
        fpr95_per_agent=pa.fpr,
        fpr95_per_agent_ci=pa_ci,
        per_agent_alpha=pa.alpha,
        per_agent_recall=pa.recall,
        breakdown_mode=breakdown_thresholds,
        per_family_envelope=recall_by_family(test, thresholds, "envelope"),
        per_family_attack=recall_by_family(test, thresholds, "attack"),
        control_fpr=control_fpr(test, thresholds),
        cold_start=cold_start_curve(test, thresholds),
        drift=drift_delay(test, thresholds),
        latency_p50_us=p50,
        latency_p99_us=p99,
        n_rows=len(test),
        n_attack_rows=n_atk,
    )
