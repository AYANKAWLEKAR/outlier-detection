"""Leave-one-group-out attribution (F10, decision T-13).

For each attack family, how many detected attacks does each group of detectors *carry*? An attack
is carried by group G when the full engine detects it and the engine with G switched off does not.
An attack that two groups carry at once (undetected without either) is counted under **each** of
them, so a group's count is exactly the number of attacks removing it loses (T-13); table totals
can therefore exceed the number of detected attacks. A detected attack that no single removal
loses is counted under ``"redundant"``.

"Detected" means the attack has a row whose decision is ``block`` or ``require_approval``. Only
**test-split** rows count (the population :func:`agentfw.bench.metrics.compute_all` evaluates
per-call metrics on), so the table covers the same attacks as test-split recall; calibration and
train rows are ignored. No warm filter is applied: it matches ``compute_all(warm_only=False)``.
Runaway attacks (``metrics.is_runaway``, T-10) are in no per-call metric, so they are never
"detected" here and never enter the table. ``splice`` attacks do enter it, as their own family:
the table reads the engine's decisions and fits no threshold, so they move no other row (T-40).

The engine is any registry engine variant (default ``"engine"``) built with the caller's config
and hints; a group is applied to the engine that variant builds (:func:`group_engine`): its
``cfg_off`` overrides the variant's effective config and its ``drop_detector`` leaves the
variant's enabled set, so ``engine-traj`` can lose ``vital`` or one of its switches (T-52).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agentfw.bench.ablation import parallel_map
from agentfw.bench.metrics import is_runaway
from agentfw.bench.registry import ENGINE_VARIANTS, make_detector
from agentfw.bench.replay import OutcomeMode, ScoreRow, SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS, DETECTOR_ORDER, Engine
from agentfw.core.envelope import Label, RawCall
from agentfw.core.hints import ToolHint

__all__ = [
    "DEFAULT_GROUPS",
    "REDUNDANT",
    "AttributionGroup",
    "AttributionRuns",
    "attribution_runs",
    "attribution_table",
    "detected_attacks",
    "group_engine",
    "tabulate",
]

REDUNDANT = "redundant"
_FLAGGED = frozenset({"block", "require_approval"})


@dataclass(slots=True, frozen=True)
class AttributionGroup:
    """A removable group: drop one detector, and/or replay with config fields overridden."""

    name: str
    drop_detector: str | None = None
    cfg_off: Mapping[str, object] = field(default_factory=dict)


DEFAULT_GROUPS: tuple[AttributionGroup, ...] = tuple(
    AttributionGroup(d, drop_detector=d) for d in DETECTOR_ORDER if d in ALL_DETECTORS
)


def detected_attacks(
    rows: Sequence[ScoreRow], *, splits: frozenset[str] = frozenset({"test"})
) -> set[str]:
    """Attack ids with an attack row in ``splits`` (default: test only; no warm filter) decided
    block or require_approval. Runaway attacks never count (T-10)."""
    return {
        r.attack_id or r.envelope_id
        for r in rows
        if r.label == "attack"
        and not is_runaway(r.attack_family)
        and r.split in splits
        and r.decision in _FLAGGED
    }


@dataclass(slots=True, frozen=True)
class AttributionRuns:
    """The detected-attack sets behind a table: the full engine's and, per group, the engine's
    without that group (same order as the groups)."""

    full: frozenset[str]
    without: tuple[frozenset[str], ...]


_Job = tuple[
    list[RawCall],
    dict[str, Label],
    AttributionGroup | None,
    EngineConfig,
    OutcomeMode,
    SplitSpec,
    str,
    Mapping[str, ToolHint],
]


def group_engine(
    detector: str, cfg: EngineConfig, group: AttributionGroup | None, hints: Mapping[str, ToolHint]
) -> Engine:
    """``detector`` (a registry engine variant) built with ``cfg``, then the group applied to the
    engine it builds: ``cfg_off`` on top of the variant's effective config (so a variant's own
    switches, e.g. ``engine-traj``'s, can be turned off) and ``drop_detector`` removed from its
    enabled detectors (so ``vital`` can be dropped from a variant that enables it). For
    ``"engine"`` this is the registry's ``engine-no-{drop}`` with ``cfg_off`` applied."""
    base = make_detector(detector, cfg=cfg, hints=hints)
    if not isinstance(base, Engine):
        raise TypeError(f"{detector!r} is not an engine variant")
    if group is None:
        return base
    drop = group.drop_detector
    if drop is not None and drop not in base.enabled:
        raise ValueError(f"{detector!r} does not enable {drop!r}")
    eff = base.cfg.replace(**group.cfg_off) if group.cfg_off else base.cfg
    return Engine(eff, enabled=base.enabled - {drop} if drop else base.enabled, hints=hints)


def _run(job: _Job) -> frozenset[str]:
    raws, labels, group, cfg, outcomes, split, detector, hints = job
    det = group_engine(detector, cfg, group, hints)
    rows = replay(raws, labels, det, split, outcomes=outcomes)
    return frozenset(detected_attacks(rows))


def attribution_runs(
    raws: Sequence[RawCall],
    labels: dict[str, Label],
    *,
    groups: Sequence[AttributionGroup],
    cfg: EngineConfig,
    outcomes: OutcomeMode,
    split: SplitSpec,
    workers: int = 1,
    detector: str = "engine",
    hints: Mapping[str, ToolHint] | None = None,
) -> AttributionRuns:
    """One full replay plus one per group of ``detector`` (a registry engine variant, built with
    ``cfg`` and ``hints``; ``None`` = no hints, the default engine's). A group is applied to the
    engine the variant builds (:func:`group_engine`; Task 4.1, T-52), so it can turn off one of
    the variant's switches or drop any detector it enables, ``vital`` included."""
    if detector not in ENGINE_VARIANTS:
        raise KeyError(f"{detector!r} is not a registry engine variant")
    shared = list(raws)
    h: Mapping[str, ToolHint] = {} if hints is None else hints
    jobs: list[_Job] = [
        (shared, labels, g, cfg, outcomes, split, detector, h) for g in (None, *groups)
    ]
    full, *without = parallel_map(_run, jobs, workers)
    return AttributionRuns(full, tuple(without))


def tabulate(
    runs: AttributionRuns, labels: dict[str, Label], groups: Sequence[AttributionGroup]
) -> dict[str, dict[str, int]]:
    """``family -> (group name | "redundant") -> attacks carried`` from :func:`attribution_runs`
    (module docstring). Families with no detected attack are absent."""
    family = {
        lab.attack_id: lab.attack_family or "unknown"
        for lab in labels.values()
        if lab.label == "attack" and lab.attack_id is not None and not is_runaway(lab.attack_family)
    }
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for atk in sorted(runs.full):
        fam = family.get(atk, "unknown")
        carriers = [g.name for g, det in zip(groups, runs.without, strict=True) if atk not in det]
        for name in carriers or [REDUNDANT]:
            table[fam][name] += 1
    return {f: dict(v) for f, v in table.items()}


def attribution_table(
    raws: Sequence[RawCall],
    labels: dict[str, Label],
    *,
    groups: Sequence[AttributionGroup],
    cfg: EngineConfig,
    outcomes: OutcomeMode,
    split: SplitSpec,
    workers: int = 1,
    detector: str = "engine",
    hints: Mapping[str, ToolHint] | None = None,
) -> dict[str, dict[str, int]]:
    """``family -> (group name | "redundant") -> attacks carried``; one full replay plus one per
    group (:func:`attribution_runs`, then :func:`tabulate`). Families with no detected attack
    are absent."""
    runs = attribution_runs(
        raws,
        labels,
        groups=groups,
        cfg=cfg,
        outcomes=outcomes,
        split=split,
        workers=workers,
        detector=detector,
        hints=hints,
    )
    return tabulate(runs, labels, groups)
