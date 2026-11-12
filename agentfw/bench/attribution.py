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
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agentfw.bench.ablation import parallel_map
from agentfw.bench.metrics import is_runaway
from agentfw.bench.registry import make_detector
from agentfw.bench.replay import OutcomeMode, ScoreRow, SplitSpec, replay
from agentfw.core.config import EngineConfig
from agentfw.core.engine import ALL_DETECTORS, DETECTOR_ORDER
from agentfw.core.envelope import Label, RawCall

__all__ = [
    "DEFAULT_GROUPS",
    "REDUNDANT",
    "AttributionGroup",
    "attribution_table",
    "detected_attacks",
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


def _run(
    job: tuple[
        list[RawCall],
        dict[str, Label],
        AttributionGroup | None,
        EngineConfig,
        OutcomeMode,
        SplitSpec,
    ],
) -> set[str]:
    raws, labels, group, cfg, outcomes, split = job
    name = "engine"
    if group is not None:
        if group.cfg_off:
            cfg = cfg.replace(**group.cfg_off)
        if group.drop_detector is not None:
            name = f"engine-no-{group.drop_detector}"
    rows = replay(raws, labels, make_detector(name, cfg=cfg), split, outcomes=outcomes)
    return detected_attacks(rows)


def attribution_table(
    raws: Sequence[RawCall],
    labels: dict[str, Label],
    *,
    groups: Sequence[AttributionGroup],
    cfg: EngineConfig,
    outcomes: OutcomeMode,
    split: SplitSpec,
    workers: int = 1,
) -> dict[str, dict[str, int]]:
    """``family -> (group name | "redundant") -> attacks carried``; one full replay plus one per
    group. Families with no detected attack are absent."""
    shared = list(raws)
    jobs = [(shared, labels, g, cfg, outcomes, split) for g in (None, *groups)]
    full, *without = parallel_map(_run, jobs, workers)
    family = {
        lab.attack_id: lab.attack_family or "unknown"
        for lab in labels.values()
        if lab.label == "attack" and lab.attack_id is not None and not is_runaway(lab.attack_family)
    }
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for atk in sorted(full):
        fam = family.get(atk, "unknown")
        carriers = [g.name for g, det in zip(groups, without, strict=True) if atk not in det]
        for name in carriers or [REDUNDANT]:
            table[fam][name] += 1
    return {f: dict(v) for f, v in table.items()}
