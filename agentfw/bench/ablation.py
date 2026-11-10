"""Detector ablations (spec 5.6, Tier 1 exit): one replay per registry variant.

Every variant is replayed on the same dataset and split and summarised on the **warm test** rows
(the headline population): AUPRC, per-agent FPR@95 *with the recall it achieved*, attack-level
recall per family at the engine's live per-agent thresholds (``"row"``; the ``"all"`` strength
bucket), and the final ``state_bytes``. ``workers > 1`` replays variants in parallel processes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agentfw.bench.dataset import load_dir
from agentfw.bench.metrics import compute_all
from agentfw.bench.protocol import Detector
from agentfw.bench.registry import DETECTORS, make_detector
from agentfw.bench.replay import OutcomeMode, ScoreRow, SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.envelope import Label, RawCall

__all__ = ["AblationRow", "ablation_row", "parallel_map", "replay_detector", "run_ablation"]


@dataclass(slots=True, frozen=True)
class AblationRow:
    variant: str
    auprc: float
    fpr95_per_agent: float
    per_agent_recall: float  # the recall fpr95_per_agent was achieved at
    per_family_recall: dict[str, float]  # attack level, live thresholds, "all" bucket
    state_bytes: int


@lru_cache(maxsize=2)
def _load(traces_dir: str) -> tuple[list[RawCall], dict[str, Label]]:
    raws, labels, _ = load_dir(traces_dir)
    return raws, labels


def replay_detector(
    traces_dir: str | Path,
    detector: str,
    split: SplitSpec,
    cfg: EngineConfig | None = None,
    outcomes: OutcomeMode = "allow",
) -> tuple[list[ScoreRow], Detector]:
    """Replay one registry detector (an engine variant gets ``cfg``) over a dataset directory;
    ``outcomes`` is :func:`agentfw.bench.replay.replay`'s outcome mode (default allow-all)."""
    if detector not in DETECTORS:
        raise KeyError(f"unknown detector {detector!r}; available: {sorted(DETECTORS)}")
    raws, labels = _load(str(traces_dir))
    det = make_detector(detector) if cfg is None else make_detector(detector, cfg=cfg)
    return replay(raws, labels, det, split, outcomes=outcomes), det


def ablation_row(
    variant: str, rows: Sequence[ScoreRow], state_bytes: int, *, n_boot: int = 0, seed: int = 0
) -> AblationRow:
    m = compute_all(rows, warm_only=True, breakdown_thresholds="row", n_boot=n_boot, seed=seed)
    return AblationRow(
        variant=variant,
        auprc=m.auprc,
        fpr95_per_agent=m.fpr95_per_agent,
        per_agent_recall=m.per_agent_recall,
        per_family_recall={f: cells["all"].recall for f, cells in m.per_family_attack.items()},
        state_bytes=state_bytes,
    )


def parallel_map[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """``[fn(x) for x in items]``, in worker processes when ``workers > 1`` (order kept)."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def _one(job: tuple[str, str, SplitSpec, EngineConfig | None, int]) -> AblationRow:
    traces_dir, variant, split, cfg, n_boot = job
    rows, det = replay_detector(traces_dir, variant, split, cfg)
    return ablation_row(variant, rows, det.state_bytes(), n_boot=n_boot)


def run_ablation(
    traces_dir: str | Path,
    variants: Sequence[str],
    split: SplitSpec,
    *,
    cfg: EngineConfig | None = None,
    n_boot: int = 0,
    workers: int = 1,
) -> list[AblationRow]:
    """One :class:`AblationRow` per variant, in order. Unknown variants raise ``KeyError`` first."""
    unknown = [v for v in variants if v not in DETECTORS]
    if unknown:
        raise KeyError(f"unknown detectors {unknown}; available: {sorted(DETECTORS)}")
    jobs = [(str(traces_dir), v, split, cfg, n_boot) for v in variants]
    return parallel_map(_one, jobs, workers)
